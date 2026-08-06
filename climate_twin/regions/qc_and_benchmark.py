"""
regions/qc_and_benchmark.py — Phase 1 hard STOP-gate artefacts.

Produces:
  regions/qc/zones_hard.png              full-India hard mask overlay
  regions/qc/zones_membership_argmax.png soft-membership argmax (for sanity)
  regions/qc/zone_annual_rain.png        annual mean rainfall per zone
  regions/qc/split_coverage.md           table zone × split × cell count × years
  _phase0/benchmark_persistence_climatology.json
                                          per-zone persistence + climatology skill
                                          on 2024-2025 holdout — the number to beat
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import data_source as DS                                # noqa: E402
from regions.india_zones import get_zones                # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
HERE = Path(__file__).resolve().parent
QC_DIR = HERE / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)
PHASE0_DIR = REPO / "_phase0"
BENCHMARK_JSON = PHASE0_DIR / "benchmark_persistence_climatology.json"
COVERAGE_MD = QC_DIR / "split_coverage.md"


# ---------------------------------------------------------------------------
# QC maps
# ---------------------------------------------------------------------------
def _render_qc_maps(Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    mg = Z.master_grid
    lats = np.linspace(mg["extent_lat"][0], mg["extent_lat"][1], mg["n_lat"])
    lons = np.linspace(mg["extent_lon"][0], mg["extent_lon"][1], mg["n_lon"])

    palette = [
        "#0b0f1e",   # 0 unassigned (near-black)
        "#F4A34A", "#8AB4F8", "#7EE787", "#B392F0",
        "#F97583", "#79B8FF", "#FFEA7F", "#A5D6FF", "#F4C7C3",
    ]
    cmap = ListedColormap(palette[:Z.n_zones() + 1])
    bounds = list(range(Z.n_zones() + 2))
    norm = BoundaryNorm(bounds, cmap.N)

    # ── 1. hard mask ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(lons, lats, Z.hard_mask, cmap=cmap, norm=norm,
                        shading="nearest")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(20.0)))
    ax.set_title("Hard zone mask — India (0.25°, 129×135)\n"
                 f"signature {Z.mask_signature}",
                 color="white", fontsize=11)
    ax.set_xlabel("°E"); ax.set_ylabel("°N")
    ax.set_facecolor("#050912")
    fig.set_facecolor("#050912")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

    cbar = fig.colorbar(im, ax=ax, ticks=[i + 0.5 for i in range(Z.n_zones() + 1)])
    labels = ["0 unassigned"] + [f"{z.id} {z.key}" for z in Z.zones]
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(QC_DIR / "zones_hard.png", dpi=140,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ── 2. soft-membership argmax (sanity: should match hard mask
    #      except in boundary bands where argmax may flip) ──
    soft_arg = np.zeros_like(Z.hard_mask)
    zone_ids = list(Z.zone_ids)
    for i in range(Z.hard_mask.shape[0]):
        for j in range(Z.hard_mask.shape[1]):
            if Z.hard_mask[i, j] == 0:
                continue
            k = int(np.argmax(Z.membership[i, j]))
            soft_arg[i, j] = zone_ids[k]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.pcolormesh(lons, lats, soft_arg, cmap=cmap, norm=norm, shading="nearest")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(20.0)))
    ax.set_title("Soft membership argmax (sanity check)", color="white")
    ax.set_facecolor("#050912"); fig.set_facecolor("#050912")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    fig.tight_layout()
    fig.savefig(QC_DIR / "zones_membership_argmax.png", dpi=140,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ── 3. Per-cell annual mean rain (whole cube) — colored by zone
    #      Renders as land-cell scatter with per-zone median annual rainfall.
    ds = DS.load_region("india")
    rain = ds["rain"].values     # (T, H, W)
    years = np.asarray(ds["time.year"].values)
    # annual sum: for each year, sum daily rain (ignoring NaN)
    y_uniq = np.unique(years)
    ann = np.zeros((len(y_uniq), rain.shape[1], rain.shape[2]), dtype=np.float32)
    for k, y in enumerate(y_uniq):
        m = years == y
        ann[k] = np.nansum(rain[m], axis=0)
    mean_ann = ann.mean(axis=0)                          # mean annual rainfall
    mean_ann = np.where(Z.hard_mask > 0, mean_ann, np.nan)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(lons, lats, mean_ann, cmap="Blues",
                        vmin=0, vmax=np.nanpercentile(mean_ann, 99),
                        shading="nearest")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(20.0)))
    ax.set_title("Mean annual rainfall (1951-2025, mm/year)", color="white")
    ax.set_facecolor("#050912"); fig.set_facecolor("#050912")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    cb = fig.colorbar(im, ax=ax, label="mm/year")
    cb.ax.tick_params(colors="white")
    cb.ax.yaxis.label.set_color("white")
    fig.tight_layout()
    fig.savefig(QC_DIR / "zone_annual_rain.png", dpi=140,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    ds.close()

    return mean_ann


# ---------------------------------------------------------------------------
# Split coverage
# ---------------------------------------------------------------------------
def _split_coverage(Z, train_years, val_years, test_years):
    ds = DS.load_region("india")
    years = np.asarray(ds["time.year"].values)
    time_len = len(years)
    all_years = sorted(int(y) for y in np.unique(years))

    def _mk_slice(rng):
        return (years >= rng[0]) & (years <= rng[1])

    splits = {
        "train": _mk_slice(train_years),
        "val":   _mk_slice(val_years),
        "test":  _mk_slice(test_years),
    }

    lines = ["# Phase 1f — Split coverage table",
             "",
             f"master grid: {Z.hard_mask.shape}   cube time: {time_len} days",
             f"train years: {train_years}   val years: {val_years}   test years: {test_years}",
             "",
             "| Zone | id | hard-cells | soft-mass | train-days | val-days | test-days | rain μ (mm/d) | flag |",
             "|------|---:|-----------:|----------:|-----------:|---------:|----------:|--------------:|------|"]

    rain = ds["rain"].values
    for zone in Z.zones:
        k = Z.zone_ids.index(zone.id)
        m_cells = Z.membership[..., k].astype(np.float32)
        hard_cells = int((Z.hard_mask == zone.id).sum())
        soft_mass = float(m_cells.sum())

        # per-split valid cell-days (finite AND weight>0.05)
        def _valid(sel):
            arr = rain[sel]
            finite = np.isfinite(arr)
            w = np.broadcast_to(m_cells[None, :, :], arr.shape)
            return int(((finite & (w > 0.05))).sum())

        n_train = _valid(splits["train"])
        n_val = _valid(splits["val"])
        n_test = _valid(splits["test"])

        # mean rain over train years, weighted
        arr = rain[splits["train"]]
        finite = np.isfinite(arr)
        w = np.broadcast_to(m_cells[None, :, :], arr.shape)
        num = float(np.where(finite, arr * w, 0.0).sum())
        den = float(np.where(finite, w, 0.0).sum())
        mean_rain = num / den if den > 0 else float("nan")

        flag_bits = []
        if hard_cells < Z.min_cells_per_split:
            flag_bits.append(f"⚠ cells<{Z.min_cells_per_split}")
        if n_val < 100 * hard_cells:
            flag_bits.append("⚠ thin val")
        if n_test < 100 * hard_cells:
            flag_bits.append("⚠ thin test")
        flag = " ".join(flag_bits) or "✓"

        lines.append(
            f"| {zone.key} | {zone.id} | {hard_cells} | {soft_mass:.1f} | "
            f"{n_train:,} | {n_val:,} | {n_test:,} | {mean_rain:.3f} | {flag} |"
        )

    COVERAGE_MD.write_text("\n".join(lines), encoding="utf-8")
    ds.close()
    print(f"[qc] wrote {COVERAGE_MD}")


# ---------------------------------------------------------------------------
# Benchmark — per-zone persistence + climatology on 2024-2025 holdout
# ---------------------------------------------------------------------------
def _benchmark(Z, train_years, test_years):
    """For each zone, compute persistence & climatology skill on the test window."""
    ds = DS.load_region("india")
    years = np.asarray(ds["time.year"].values)
    doys = np.asarray(ds["time.dayofyear"].values)

    train_sel = (years >= train_years[0]) & (years <= train_years[1])
    test_sel  = (years >= test_years[0])  & (years <= test_years[1])
    train_doys = doys[train_sel]
    test_doys = doys[test_sel]

    out = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "zone_mask_sig": Z.mask_signature,
        "manifest_sig": DS.manifest_sig(),
        "train_years": list(train_years),
        "test_years": list(test_years),
        "notes": (
            "For each zone × variable × baseline, RMSE and MAE are computed "
            "over all (day × cell) pairs in the test window with soft-membership "
            "weight > 0.05. Persistence baseline = yesterday's value at same "
            "cell. Climatology baseline = per-DOY mean over train years at "
            "that cell. This is the honest number-to-beat for Phase 2."
        ),
        "zones": {},
    }

    variables = ("rain", "tmax", "tmin")   # skip insat (2020-2021 only)

    # Precompute per-cell per-DOY climatology from train years
    print("[bench] precomputing per-cell per-DOY climatology from train years…")
    H, W = ds.sizes["lat"], ds.sizes["lon"]
    clim = {v: np.full((366, H, W), np.nan, dtype=np.float32) for v in variables}
    for v in variables:
        arr_train = ds[v].values[train_sel]     # (T_train, H, W)
        for d in range(1, 367):
            m = train_doys == d
            if not m.any():
                continue
            with np.errstate(invalid="ignore"):
                clim[v][d - 1] = np.nanmean(arr_train[m], axis=0)
        print(f"   {v}: climatology filled")

    # Ground truth on the test window
    for var in variables:
        truth = ds[var].values[test_sel]        # (T_test, H, W)

        # Persistence = shift by 1 day. First day → yesterday from just before test.
        # We look up the last training day and prepend it.
        idx_test_start = int(np.argmax(test_sel))
        prev = ds[var].values[max(0, idx_test_start - 1):idx_test_start + truth.shape[0] - 1]
        persistence = np.concatenate([ds[var].values[idx_test_start - 1:idx_test_start], prev[:-1]], axis=0) if idx_test_start > 0 else np.roll(truth, 1, axis=0)
        if persistence.shape[0] != truth.shape[0]:
            persistence = persistence[:truth.shape[0]]

        # Climatology prediction = clim[var][doy-1] for each test day
        climatology = np.stack([clim[var][d - 1] for d in test_doys], axis=0)

        for zone in Z.zones:
            k = Z.zone_ids.index(zone.id)
            m_cells = Z.membership[..., k].astype(np.float32)
            w = np.broadcast_to(m_cells[None, :, :], truth.shape)
            finite = np.isfinite(truth) & np.isfinite(persistence) & np.isfinite(climatology) & (w > 0.05)

            def _rmse(pred):
                diff = pred - truth
                w_eff = np.where(finite, w, 0.0)
                num = float(np.where(finite, (diff ** 2) * w_eff, 0.0).sum())
                den = float(w_eff.sum())
                return float(np.sqrt(num / den)) if den > 0 else float("nan")

            def _mae(pred):
                w_eff = np.where(finite, w, 0.0)
                den = float(w_eff.sum())
                num = float(np.where(finite, np.abs(pred - truth) * w_eff, 0.0).sum())
                return float(num / den) if den > 0 else float("nan")

            zone_bucket = out["zones"].setdefault(zone.key, {"id": zone.id, "metrics": {}})
            n_valid = int(finite.sum())
            zone_bucket["metrics"][var] = {
                "n_valid_pairs": n_valid,
                "persistence": {"rmse": _rmse(persistence), "mae": _mae(persistence)},
                "climatology": {"rmse": _rmse(climatology), "mae": _mae(climatology)},
            }

        print(f"[bench] {var}: computed persistence+climatology skill for {Z.n_zones()} zones")

    ds.close()

    BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_JSON.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {BENCHMARK_JSON}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    Z = get_zones()
    mani = DS.manifest_dict()
    train_years = tuple(mani.get("train_years", [1951, 2022]))

    # The plan: train 1951-2022, val 2023, test 2024-2025
    val_years = (train_years[1] + 1, train_years[1] + 1)             # 2023
    test_years = (train_years[1] + 2, int(mani["years"][1]))          # 2024-2025

    print(f"[phase1] train {train_years}  val {val_years}  test {test_years}")

    _render_qc_maps(Z)
    print(f"[qc] wrote QC maps to {QC_DIR}")

    _split_coverage(Z, train_years, val_years, test_years)
    bench = _benchmark(Z, train_years, test_years)

    # Terse console table of the benchmark
    print()
    print("── Persistence vs Climatology skill on test years", test_years, "──")
    print("  zone         │ rain-RMSE (pers)  rain-RMSE (clim)  │ tmax-RMSE (pers)  tmax-RMSE (clim)")
    for zone in Z.zones:
        m = bench["zones"][zone.key]["metrics"]
        print(f"  {zone.key:12s} │  {m['rain']['persistence']['rmse']:7.3f}          "
              f"{m['rain']['climatology']['rmse']:7.3f}         │ "
              f" {m['tmax']['persistence']['rmse']:7.3f}          "
              f"{m['tmax']['climatology']['rmse']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
