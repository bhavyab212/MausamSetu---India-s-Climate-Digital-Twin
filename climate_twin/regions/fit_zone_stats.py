"""
regions/fit_zone_stats.py — fit per-zone stats on TRAIN YEARS ONLY.

Writes:
    regions/zone_stats.json     — per-zone norm + climatology + percentile
                                  thresholds, plus loss weights
    regions/zone_stats.sha256   — 12-hex signature bound to (zone_mask.sha256
                                  + manifest.sig + train_years)

Contract (from india_zones.yaml + the rebuild plan):
  * All stats are fit on TRAIN YEARS ONLY (default 1951-2022 from the cube's
    manifest). This is the second leakage surface — take it as seriously
    as the year-split rule.
  * Every per-cell contribution is weighted by the cell's SOFT membership
    for its zone (interior cells count as 1.0, boundary cells contribute
    partially to both adjacent zones).
  * Insufficient cells (< min_cells_per_split) → the zone is FLAGGED, not
    silently absorbed. Its stats are still written but ``insufficient=True``.

Fields per zone:
  n_cells                              hard-mask cells
  n_valid_cell_days_train_by_var       (excludes NaN target cells)
  norm.<var> = {mean, std}             z-score parameters, train-only
  climatology.<var>.day_of_year[366]   daily mean over the wet-season year
                                       (float32; NaN if that DOY has <10 obs)
  percentiles_rain_mm_day.{p50,p90,p95,p99}
  loss_weight                          inverse-variance normalized across zones
  imd_category_counts (train)          {category: fraction_of_days_with_any_cell}
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[1]        # climate_twin/
sys.path.insert(0, str(REPO))

import data_source as DS                          # noqa: E402
from regions.india_zones import get_zones          # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
HERE = Path(__file__).resolve().parent
STATS_PATH = HERE / "zone_stats.json"
STATS_HASH_PATH = HERE / "zone_stats.sha256"

VARIABLES = ("rain", "tmax", "tmin", "insat_lst")


# ---------------------------------------------------------------------------
def _train_year_slice(ds: xr.Dataset, train_years: tuple[int, int]) -> np.ndarray:
    years = np.asarray(ds["time.year"].values)
    return (years >= train_years[0]) & (years <= train_years[1])


def _per_cell_zone_membership(memb: np.ndarray, zone_k: int) -> np.ndarray:
    """Return (H, W) float32 = membership of zone_k for every cell."""
    return memb[..., zone_k].astype(np.float32)


def _fit_norm(vals: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    """Weighted mean + std of a 3-D (T, H, W) field.

    ``weights`` is (H, W). Values that are NaN are ignored; the effective
    denominator is the sum of ``weights`` broadcast over time × finite mask.
    """
    finite = np.isfinite(vals)
    # broadcast (H,W) weights over T
    w = np.broadcast_to(weights[None, :, :], vals.shape)
    # apply finite mask and weight
    w_eff = np.where(finite, w, 0.0)
    n_eff = w_eff.sum()
    if n_eff <= 0:
        return float("nan"), float("nan"), 0
    v = np.where(finite, vals, 0.0)
    mean = float((v * w_eff).sum() / n_eff)
    var = float(((v - mean) ** 2 * w_eff).sum() / n_eff)
    return mean, float(np.sqrt(max(var, 0.0))), int(finite.sum())


def _fit_climatology(vals: np.ndarray, weights: np.ndarray, doys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each DOY 1..366, weighted mean across all matching cells.

    Returns (mean_by_doy[366], n_by_doy[366]).
    """
    mean_by_doy = np.full(366, np.nan, dtype=np.float32)
    n_by_doy = np.zeros(366, dtype=np.int64)
    w = np.broadcast_to(weights[None, :, :], vals.shape)
    for d in range(1, 367):
        m = doys == d
        if not m.any():
            continue
        sub = vals[m]                    # (n_days_this_doy, H, W)
        finite = np.isfinite(sub)
        w_sub = np.where(finite, np.broadcast_to(weights[None, :, :], sub.shape), 0.0)
        n = float(w_sub.sum())
        if n <= 0:
            n_by_doy[d - 1] = 0
            continue
        v = np.where(finite, sub, 0.0)
        mean_by_doy[d - 1] = float((v * w_sub).sum() / n)
        n_by_doy[d - 1] = int(finite.sum())
    # If a DOY has <10 valid cell-observations pooled, drop it (too noisy)
    mean_by_doy[n_by_doy < 10] = np.nan
    return mean_by_doy, n_by_doy


def _fit_percentiles(vals: np.ndarray, weights: np.ndarray, qs=(50, 90, 95, 99)) -> dict[str, float]:
    """Weighted percentiles of a 3-D field on land cells.

    We approximate with an unweighted percentile over the samples where
    weight > 0.05, because true weighted percentiles are expensive and
    boundary cells (weight in (0.05, 0.95)) are few. For the interior
    (weight ≈ 1) this is exact.
    """
    finite = np.isfinite(vals)
    w = np.broadcast_to(weights[None, :, :], vals.shape)
    keep = finite & (w > 0.05)
    sub = vals[keep]
    if sub.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    pcts = np.percentile(sub, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, pcts)}


def _imd_category_share(rain_vals: np.ndarray, weights: np.ndarray, categories: dict) -> dict[str, float]:
    finite = np.isfinite(rain_vals)
    w = np.broadcast_to(weights[None, :, :], rain_vals.shape)
    keep = finite & (w > 0.05)
    total = int(keep.sum())
    out: dict[str, float] = {}
    if total == 0:
        return {k: 0.0 for k in categories}
    v = rain_vals[keep]
    for name, (lo, hi) in categories.items():
        n_in = int(((v >= lo) & (v <= hi)).sum())
        out[name] = round(100.0 * n_in / total, 3)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    print("[fit] loading zone registry + india cube…")
    Z = get_zones()
    ds = DS.load_region("india")

    manifest_sig = DS.manifest_sig() or "?"
    mani = DS.manifest_dict()
    train_years = tuple(mani.get("train_years", [1951, 2022]))
    print(f"  zone_mask_sig  = {Z.mask_signature}")
    print(f"  manifest_sig   = {manifest_sig}")
    print(f"  train_years    = {train_years}")

    tsel = _train_year_slice(ds, train_years)
    n_train_days = int(tsel.sum())
    print(f"  train days     = {n_train_days}")
    doys = np.asarray(ds["time.dayofyear"].values)[tsel]

    per_zone_var: dict[str, dict[str, Any]] = {}
    # First pass: fit norm+climatology+percentiles per zone × variable
    for zi, zone in enumerate(Z.zones):
        k = Z.zone_ids.index(zone.id)
        w = _per_cell_zone_membership(Z.membership, k)          # (H, W)
        hard_cells = int((Z.hard_mask == zone.id).sum())
        soft_cells = float(w.sum())
        print(f"\n[{zone.key}] hard={hard_cells}  soft={soft_cells:.1f}")

        zvar: dict[str, Any] = {
            "id": zone.id,
            "n_cells_hard": hard_cells,
            "soft_cell_mass": round(soft_cells, 2),
            "insufficient": hard_cells < Z.min_cells_per_split,
            "norm": {},
            "climatology": {},
            "percentiles_rain_mm_day": {},
            "n_valid_cell_days_train": {},
        }

        for var in VARIABLES:
            # Full 3-D slice for train years (T_train, H, W) — memory OK on 129x135
            arr = ds[var].values[tsel]  # float32
            mean, std, n_valid = _fit_norm(arr, w)
            zvar["norm"][var] = {"mean": mean, "std": std}
            zvar["n_valid_cell_days_train"][var] = n_valid
            if var == "rain":
                zvar["percentiles_rain_mm_day"] = _fit_percentiles(arr, w)
                zvar["imd_category_share_pct"] = _imd_category_share(
                    arr, w, Z.imd_categories
                )
            clim_mean, clim_n = _fit_climatology(arr, w, doys)
            zvar["climatology"][var] = {
                "doy_mean": [None if np.isnan(v) else float(v) for v in clim_mean.tolist()],
                "doy_n": [int(n) for n in clim_n.tolist()],
            }
            print(f"  {var:10s} mean={mean:+8.3f}  std={std:6.3f}  n_valid={n_valid:,}")

        per_zone_var[zone.key] = zvar

    # Loss weights: inverse variance of `rain` (the target metric of first
    # concern), clipped by india_zones.loss_weighting.clip_ratio.
    lw_mode = Z.loss_weighting_mode
    lo, hi = Z.loss_weighting_clip_ratio
    variances = np.array([per_zone_var[z.key]["norm"]["rain"]["std"] ** 2
                          for z in Z.zones])
    if lw_mode == "inverse_variance":
        w_raw = 1.0 / np.maximum(variances, 1e-6)
    elif lw_mode == "uniform":
        w_raw = np.ones_like(variances)
    else:
        w_raw = np.ones_like(variances)
    # Normalise so mean weight = 1, then clip
    w_raw = w_raw / w_raw.mean()
    w_clipped = np.clip(w_raw, lo, hi)
    # Re-normalise after clipping so mean = 1
    w_clipped = w_clipped / w_clipped.mean()
    for z, w_val in zip(Z.zones, w_clipped):
        per_zone_var[z.key]["loss_weight"] = round(float(w_val), 4)

    # Payload
    payload = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "zone_mask_sig": Z.mask_signature,
        "manifest_sig": manifest_sig,
        "train_years": list(train_years),
        "n_train_days": n_train_days,
        "loss_weighting_mode": lw_mode,
        "loss_weighting_clip_ratio": [lo, hi],
        "zones": per_zone_var,
    }
    text = json.dumps(payload, indent=2)
    STATS_PATH.write_text(text, encoding="utf-8")

    sig = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    STATS_HASH_PATH.write_text(sig)

    print()
    print("── zone stats summary ─────────────────────────────────────")
    print(f"  stats_signature = {sig}")
    print(f"  written to      = {STATS_PATH.name}")
    print()
    print("  Zone         │  cells │ rain μ (mm/d) │ rain σ │ rain p99 │ loss w")
    print("  ─────────────┼────────┼───────────────┼────────┼──────────┼───────")
    for z in Z.zones:
        r = per_zone_var[z.key]
        m = r["norm"]["rain"]["mean"]; s = r["norm"]["rain"]["std"]
        p99 = r["percentiles_rain_mm_day"].get("p99", float("nan"))
        print(f"  {z.key:12s} │ {r['n_cells_hard']:6d} │ {m:12.3f}  │ {s:6.3f} │ "
              f"{p99:8.2f} │ {r['loss_weight']:5.3f}"
              + ("  ⚠ insufficient" if r["insufficient"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
