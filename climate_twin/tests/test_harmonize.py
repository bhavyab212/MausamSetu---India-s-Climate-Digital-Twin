"""
Phase 3 QC runner — reads the 2020 readers, harmonizes to the master 0.25° grid,
runs physical consistency + missingness checks, and writes before/after PNGs.

Run:  cd climate_twin && ../venv/Scripts/python tests/test_harmonize.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CT   = REPO / "climate_twin"
sys.path.insert(0, str(CT))

from data.readers.imd_rainfall_nc import read_rainfall_year
from data.readers.imd_temperature import (read_tmax_year, read_tmin_year,
                                          TEMP_LAT0, TEMP_LAT1, TEMP_LON0, TEMP_LON1)
from data.readers.insat_lst import read_insat_day
from data import harmonize as HM

QC = CT / "data" / "qc"
QC.mkdir(parents=True, exist_ok=True)

def _plot(name, arr, extent, cmap, title, unit, vmin=None, vmax=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 4.8), facecolor="#0d1526")
    ax.set_facecolor("#0d1526")
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap,
                   aspect="auto", vmin=vmin, vmax=vmax)
    ax.tick_params(colors="#8899bb", labelsize=7)
    for s in ax.spines.values(): s.set_color("#22314f")
    ax.set_title(title, color="#e6edf7", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(unit, color="#8899bb"); cb.ax.tick_params(colors="#8899bb", labelsize=7)
    fig.tight_layout(); fig.savefig(QC / name, dpi=110); plt.close(fig)

def _pct(v): return f"{v:.2f}%"

def main():
    Y = 2020
    print(f"=== Phase 3 harmonize QC for {Y} ===")

    # ─── Read native ────────────────────────────────────────────────
    rain = read_rainfall_year(REPO / "data" / "Rainfall" / f"RF25_ind{Y}_rfp25.nc")
    tmax = read_tmax_year(REPO / "data" / "max_temp" / f"Maxtemp_MaxT_{Y}.GRD")
    tmin = read_tmin_year(REPO / "data" / "min_temp" / f"Mintemp_MinT_{Y}.GRD")

    # ─── 3a. Harmonize to master grid ──────────────────────────────
    rain_m = HM.rainfall_to_master(rain)
    tmax_m = HM.regrid_temp_to_master(tmax)
    tmin_m = HM.regrid_temp_to_master(tmin)
    HM.assert_grid_match(rain_m, tmax_m, tmin_m)
    print(f"[3a] rain shape {rain_m.shape}  tmax shape {tmax_m.shape}  tmin shape {tmin_m.shape}   (master byte-identical ✓)")

    # 3a QC: before/after temp regrid on a pre-monsoon day
    day = "2020-05-15"
    _plot(f"tmax_native_{day}.png", tmax.sel(time=day).values,
          (TEMP_LON0, TEMP_LON1, TEMP_LAT0, TEMP_LAT1), "turbo",
          f"Tmax native 1° 31×31 — {day}", "°C", vmin=15, vmax=45)
    _plot(f"tmax_master_{day}.png", tmax_m.sel(time=day).values,
          (HM.MASTER_LON[0], HM.MASTER_LON[-1], HM.MASTER_LAT[0], HM.MASTER_LAT[-1]),
          "turbo", f"Tmax bilinear→0.25° 129×135 — {day}", "°C", vmin=15, vmax=45)
    _plot(f"rain_master_{day}.png", rain_m.sel(time=day).values,
          (HM.MASTER_LON[0], HM.MASTER_LON[-1], HM.MASTER_LAT[0], HM.MASTER_LAT[-1]),
          "Blues", f"Rainfall native 0.25° — {day}", "mm/day", vmin=0, vmax=60)

    # ─── INSAT regrid ─────────────────────────────────────────────
    ip = REPO / "data" / "INSAT" / "2020" / "May" / "INSAT_LST_IndiaMean_01MAY2020.h5"
    if ip.exists():
        insat = read_insat_day(ip)
        insat_m = HM.regrid_insat_conservative(insat)
        HM.assert_grid_match(insat_m)
        print(f"[3a] INSAT native {insat.shape} → master {insat_m.shape}   (conservative area-average ✓)")
        _plot("insat_master_2020-05-01.png", insat_m.isel(time=0).values,
              (HM.MASTER_LON[0], HM.MASTER_LON[-1], HM.MASTER_LAT[0], HM.MASTER_LAT[-1]),
              "turbo", "INSAT LST conservative→0.25° — 2020-05-01", "°C", vmin=-5, vmax=45)
        print(f"     LST master values: min {np.nanmin(insat_m.values):.2f}  max {np.nanmax(insat_m.values):.2f}  nan% {_pct(100*np.isnan(insat_m.values).mean())}")
    else:
        print("[3a] INSAT sample missing — skipping regrid demo")

    # ─── 3b. Physical consistency ─────────────────────────────────
    HM.assert_no_sentinels(rain_m, tmax_m, tmin_m)
    print("[3b] no sentinels (-999 / 99.9) survive on any product ✓")

    physcheck = HM.check_tmax_ge_tmin(tmax_m, tmin_m)
    print(f"[3b] tmax >= tmin on master grid (2020):  violations = {physcheck['violations']:,} / {physcheck['valid_cell_days']:,} "
          f"({physcheck['violation_pct']:.4f}%)")
    if physcheck.get("first_violations"):
        for v in physcheck["first_violations"]:
            print(f"       - {v['time']} lat_idx={v['lat_idx']} lon_idx={v['lon_idx']}  tmax={v['tmax']:.2f}  tmin={v['tmin']:.2f}")

    lm = HM.compare_land_masks(("rain", rain_m), ("tmax", tmax_m), ("tmin", tmin_m))
    print(f"[3b] land-mask counts on master grid: {lm['counts']}")
    for k, v in lm.items():
        if k.startswith("rain vs"):
            other = k.split()[-1]
            only_other_key = f"only_{other}"
            print(f"       {k}: both={v['both']}  only_rain={v['only_rain']}  {only_other_key}={v[only_other_key]}")

    # ─── 3c. Missingness policy ───────────────────────────────────
    tmax_f, tmax_flag = HM.apply_missingness_policy(tmax_m, max_gap_days=3)
    tmin_f, tmin_flag = HM.apply_missingness_policy(tmin_m, max_gap_days=3)
    frac = HM.cell_missing_fraction(rain_m)
    print(f"[3c] tmax interpolated cells (≤3-day gaps): {int(tmax_flag.values.sum()):,}")
    print(f"[3c] tmin interpolated cells (≤3-day gaps): {int(tmin_flag.values.sum()):,}")
    print(f"[3c] rain cell-missing frac: min {frac.min():.3f}  mean {frac.mean():.3f}  max {frac.max():.3f}  "
          f"cells >20% missing: {int((frac.values > 0.20).sum())} / {frac.size}")

    # ─── 3d. Unit + attrs (already applied in readers) ────────────
    for da in (rain_m, tmax_m, tmin_m):
        assert "units" in da.attrs and "long_name" in da.attrs and "source" in da.attrs, \
            f"{da.name} missing canonical attrs"
    print(f"[3d] canonical attrs present on rain / tmax / tmin ✓")

    print(f"\nQC PNGs written to: {QC}")
    print("Files:")
    for f in sorted(QC.glob("*.png")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
