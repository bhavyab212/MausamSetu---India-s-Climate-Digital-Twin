"""
train.eval.demo_tiers — Phase 3 end-to-end demonstrator.

Run:  python -m climate_twin.train.eval.demo_tiers

Loads real 2023 (validation year) rain from the india cube, builds
persistence + climatology as the "baselines", uses climatology as the
"model" for demonstration, and runs every tier so we can:
  * verify all four tiers execute against real data without error
  * profile per-tier cost (seconds) — plan §3c
  * generate a real Tier-3 heatmap PNG for the STOP report

The insufficient-data path fires naturally on the small zones × short
threshold combinations (e.g. zone_percentile "p99" during DJF when
almost nothing exceeds it).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from climate_twin.regions import get_zones
from climate_twin import data_source as DS
from climate_twin.train.eval import (
    run_tier1, run_tier2, run_tier3, run_tier4,
    render_tier3_heatmap,
    persistence_prediction, climatology_prediction,
    per_zone_baseline_skill,
)

IST = timezone(timedelta(hours=5, minutes=30))
OUT_DIR = REPO / "climate_twin" / "_phase0" / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_climatology(ds, train_years, variable="rain"):
    """Per-cell per-DOY climatology from train years only. (366, H, W)."""
    years = np.asarray(ds["time.year"].values)
    train_sel = (years >= train_years[0]) & (years <= train_years[1])
    doys = np.asarray(ds["time.dayofyear"].values)[train_sel]
    arr = ds[variable].values[train_sel]
    H, W = arr.shape[-2:]
    clim = np.full((366, H, W), np.nan, dtype=np.float32)
    for d in range(1, 367):
        m = doys == d
        if not m.any():
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            clim[d - 1] = np.nanmean(arr[m], axis=0)
    return clim


def main() -> int:
    Z = get_zones()
    mani = DS.manifest_dict()
    train_years = tuple(mani.get("train_years", [1951, 2022]))
    val_year = 2023
    print(f"[phase3] train {train_years}  val {val_year}")
    print(f"[phase3] zone signature = {Z.mask_signature}")

    ds = DS.load_region("india")
    years = np.asarray(ds["time.year"].values)
    val_sel = years == val_year
    val_idx = int(np.argmax(val_sel))
    truth = ds["rain"].values[val_sel]                 # (T_val, H, W)
    months = np.asarray(ds["time.month"].values)[val_sel]
    doys = np.asarray(ds["time.dayofyear"].values)[val_sel]

    print(f"[data] val_days={truth.shape[0]}  H={truth.shape[1]}  W={truth.shape[2]}")

    # Persistence + climatology on val
    if val_idx > 0:
        fallback = ds["rain"].values[val_idx - 1]      # day before val
    else:
        fallback = truth[0]
    persistence = persistence_prediction(truth, fallback_first_day=fallback)
    print("[bench] building per-cell climatology from train years…")
    clim366 = _build_climatology(ds, train_years, "rain")
    climatology = climatology_prediction(doys, clim366)
    ds.close()

    # For demo: pretend "model" == climatology (so vs-climatology skill ≈ 0,
    # vs-persistence skill should be positive for rain).
    pred = climatology.copy()

    # Loss-weight vector (from Phase 1) — used by Tier 1
    memb_avg = Z.membership.astype(np.float32).sum(axis=-1) / Z.n_zones()
    hard = Z.hard_mask
    mask = (hard > 0).astype(np.float32)
    weight = np.ones_like(mask)   # for Tier 1: unweighted global RMSE

    # IMD absolute categories (mm/day) — upper bounds
    IMD_ABSOLUTE_THRESHOLDS = {
        "moderate":       15.6,
        "heavy":          64.5,
        "very_heavy":    115.6,
        "extremely_heavy": 204.5,
    }
    # Zone-percentile thresholds (from Phase-1 zone_stats.json)
    zone_stats = json.loads(
        (REPO / "climate_twin/regions/zone_stats.json").read_text(encoding="utf-8")
    )
    zone_percentile_thresholds: dict[str, dict[str, float]] = {}
    for zkey, blk in zone_stats["zones"].items():
        p = blk["percentiles_rain_mm_day"]
        zone_percentile_thresholds[zkey] = {
            "p90": float(p["p90"]), "p95": float(p["p95"]), "p99": float(p["p99"]),
        }

    # ── Tier 1 ──
    t1 = run_tier1(pred, truth, weight, mask, min_cells=5)
    print(f"[T1] rmse={t1['rmse']:.4f}  runtime={t1['runtime_s']}s")

    # ── Tier 2 ──
    # First compute persistence RMSE per zone as the baseline reference
    tier2_baseline = {}
    for k, zone in enumerate(Z.zones):
        w = Z.membership[..., k].astype(np.float32)
        finite = np.isfinite(persistence) & np.isfinite(truth) & (mask[None] > 0)
        num = float(np.where(finite, (persistence - truth)**2 * w[None], 0.0).sum())
        den = float(np.where(finite, w[None], 0.0).sum())
        tier2_baseline[zone.key] = float(np.sqrt(num / den)) if den > 0 else float("nan")
    t2 = run_tier2(pred, truth, Z, mask,
                   baseline_rmses=tier2_baseline, min_cells=5)
    print(f"[T2] runtime={t2['runtime_s']}s")
    for zk, row in t2["per_zone"].items():
        skill = row.get("skill_vs_baseline", "—")
        if isinstance(skill, float):
            skill = f"{skill:+.3f}"
        print(f"     {zk:22s}  rmse={row['rmse']!s:>7}  skill={skill}")

    # ── Tier 3 ──
    t3 = run_tier3(pred, truth, months, Z, mask,
                   thresholds_absolute=IMD_ABSOLUTE_THRESHOLDS,
                   thresholds_zone_percentile=zone_percentile_thresholds,
                   min_cells=5)
    print(f"[T3] runtime={t3['runtime_s']}s")

    # Also compute the baselines' cross-product so heatmap can show skill
    baseline_cross = per_zone_baseline_skill(
        truth, persistence, climatology, months, Z, mask,
        IMD_ABSOLUTE_THRESHOLDS, zone_percentile_thresholds, min_cells=5,
    )
    # baseline_cross is {"persistence": <tier3-shape>, "climatology": <tier3-shape>}
    baseline_wrapper = {
        src: {"per_zone_season": table}
        for src, table in [
            ("persistence", baseline_cross["persistence"]),
            ("climatology", baseline_cross["climatology"]),
        ]
    }

    heatmap_path = OUT_DIR / "tier3_heatmap.png"
    render_tier3_heatmap(
        t3,
        # skill against persistence — climatology should mostly win
        {"persistence": {"per_zone_season": baseline_cross["persistence"]}},
        Z, out_path=heatmap_path,
        title="Phase-3 demo — climatology-as-model vs persistence",
        metric_key="rmse",
        baseline_source="persistence",
    )
    print(f"[T3] wrote heatmap {heatmap_path.name}")

    # ── Tier 4 ──
    t4 = run_tier4(pred, truth, persistence, climatology, Z, mask,
                   bootstrap_samples=200, seed=42)
    print(f"[T4] runtime={t4['runtime_s']}s")
    for zk, row in t4["per_zone"].items():
        if isinstance(row.get("rmse_ci"), dict):
            r = row["rmse_ci"]
            v_p = row["vs_persistence"].get("winner", "?") if isinstance(row.get("vs_persistence"), dict) else "?"
            v_c = row["vs_climatology"].get("winner", "?") if isinstance(row.get("vs_climatology"), dict) else "?"
            print(f"     {zk:22s}  rmse={r['point']:.3f} [{r['ci95_lower']:.3f},{r['ci95_upper']:.3f}]"
                  f"  vs_pers={v_p:>10s}  vs_clim={v_c}")

    # ── Persist artefacts ──
    (OUT_DIR / "tier1.json").write_text(json.dumps(t1, indent=2, default=str))
    (OUT_DIR / "tier2.json").write_text(json.dumps(t2, indent=2, default=str))
    (OUT_DIR / "tier3.json").write_text(json.dumps(t3, indent=2, default=str))
    (OUT_DIR / "tier4.json").write_text(json.dumps(t4, indent=2, default=str))

    # ── Tier-cost profile (plan §3c) ──
    profile = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "n_val_days": int(truth.shape[0]),
        "n_land_cells": int(mask.sum()),
        "runtimes_s": {
            "tier1": t1["runtime_s"],
            "tier2": t2["runtime_s"],
            "tier3": t3["runtime_s"],
            "tier4": t4["runtime_s"],
        },
    }
    (OUT_DIR / "tier_timing_profile.json").write_text(json.dumps(profile, indent=2))

    total = sum(profile["runtimes_s"].values())
    print()
    print("── Tier cost profile ─────────────────────────")
    for tier, dt in profile["runtimes_s"].items():
        share = 100 * dt / total if total > 0 else 0
        print(f"  {tier}  {dt:7.3f}s   {share:5.1f}% of val cycle")
    print(f"  TOTAL {total:.3f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
