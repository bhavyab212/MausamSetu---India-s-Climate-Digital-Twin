"""
Reader tests — Phase 2. One test class per product; runs headless.
Also saves a sanity map to climate_twin/data/qc/ per product for eyeball QC.

Run:  cd climate_twin && ../venv/Scripts/python -m pytest tests/test_readers.py -q
Or:   cd climate_twin && ../venv/Scripts/python tests/test_readers.py
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]        # L:/MausamSetu
CT_ROOT   = REPO_ROOT / "climate_twin"
QC_DIR    = CT_ROOT / "data" / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

RAIN_YEAR = 2020
RAIN_FILE = REPO_ROOT / "data" / "Rainfall" / f"RF25_ind{RAIN_YEAR}_rfp25.nc"
TMAX_FILE = REPO_ROOT / "data" / "max_temp" / f"Maxtemp_MaxT_{RAIN_YEAR}.GRD"
TMIN_FILE = REPO_ROOT / "data" / "min_temp" / f"Mintemp_MinT_{RAIN_YEAR}.GRD"
INSAT_DIR = REPO_ROOT / "data" / "INSAT" / str(RAIN_YEAR)


def _save_qc(name: str, arr: np.ndarray, extent, cmap: str, title: str, unit: str):
    """Render one PNG for eyeball verification (does not import Streamlit)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor="#0d1526")
    ax.set_facecolor("#0d1526")
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="auto")
    ax.tick_params(colors="#8899bb", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#22314f")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(unit, color="#8899bb")
    cbar.ax.tick_params(colors="#8899bb", labelsize=7)
    ax.set_title(title, color="#e6edf7", fontsize=11)
    ax.set_xlabel("Longitude (°E)", color="#8899bb")
    ax.set_ylabel("Latitude (°N)", color="#8899bb")
    out = QC_DIR / name
    fig.tight_layout(); fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ─────────────────────────── Rainfall ─────────────────────────────────────

@pytest.mark.skipif(not RAIN_FILE.exists(), reason="rainfall file missing")
def test_rainfall_year_2020():
    import sys
    sys.path.insert(0, str(CT_ROOT))
    from data.readers.imd_rainfall_nc import read_rainfall_year, validate_rainfall
    da = read_rainfall_year(RAIN_FILE)
    stats = validate_rainfall(da)
    assert da.name == "rain"
    assert stats["shape"] == (366, 129, 135)                # 2020 is leap
    assert stats["lat_range"] == (6.5, 38.5)
    assert stats["lon_range"] == (66.5, 100.0)
    assert stats["nan_pct"] > 60.0                          # ocean is masked
    assert stats["out_of_range"] == 0
    # coord monotonicity
    assert bool((np.diff(da.lat.values) > 0).all())
    assert bool((np.diff(da.lon.values) > 0).all())
    # QC map: monsoon peak day
    day = da.sel(time="2020-07-15", method="nearest").values
    _save_qc("rain_2020-07-15.png", day,
             extent=(66.5, 100.0, 6.5, 38.5), cmap="Blues",
             title=f"IMD rainfall — 2020-07-15 (max {np.nanmax(day):.0f} mm/day)",
             unit="mm/day")
    print("[rain]", stats)


# ─────────────────────────── Temperature ──────────────────────────────────

@pytest.mark.skipif(not (TMAX_FILE.exists() and TMIN_FILE.exists()), reason="temp files missing")
def test_tmax_tmin_year_2020():
    import sys
    sys.path.insert(0, str(CT_ROOT))
    from data.readers.imd_temperature import (read_tmax_year, read_tmin_year, validate_temp,
                                              TEMP_LAT0, TEMP_LAT1, TEMP_LON0, TEMP_LON1)
    tmax = read_tmax_year(TMAX_FILE)
    tmin = read_tmin_year(TMIN_FILE)
    smax = validate_temp(tmax); smin = validate_temp(tmin)
    assert tmax.name == "tmax" and tmin.name == "tmin"
    assert smax["shape"] == (366, 31, 31) == smin["shape"]
    # tmax >= tmin at native resolution over land (Phase-1 showed only 3 violations 1951-2025)
    land = ~np.isnan(tmax.values) & ~np.isnan(tmin.values)
    viol = int((tmax.values[land] < tmin.values[land]).sum())
    assert viol == 0, f"tmax<tmin violations in 2020: {viol}"
    # QC maps
    _save_qc("tmax_2020-05-15.png", tmax.sel(time="2020-05-15", method="nearest").values,
             extent=(TEMP_LON0, TEMP_LON1, TEMP_LAT0, TEMP_LAT1), cmap="turbo",
             title="IMD Tmax — 2020-05-15 (pre-monsoon peak)", unit="°C")
    _save_qc("tmin_2020-01-15.png", tmin.sel(time="2020-01-15", method="nearest").values,
             extent=(TEMP_LON0, TEMP_LON1, TEMP_LAT0, TEMP_LAT1), cmap="turbo",
             title="IMD Tmin — 2020-01-15 (winter minima)", unit="°C")
    print("[tmax]", smax); print("[tmin]", smin)


# ─────────────────────────── INSAT LST ────────────────────────────────────

@pytest.mark.skipif(not INSAT_DIR.exists(), reason="INSAT folder missing")
def test_insat_single_day():
    import sys
    sys.path.insert(0, str(CT_ROOT))
    from data.readers.insat_lst import read_insat_day, validate_insat
    # Pick a known-present day (Phase-1: 2020-05-01 exists)
    candidates = list((REPO_ROOT / "data" / "INSAT" / "2020" / "May").glob(
        "INSAT_LST_IndiaMean_01MAY2020.h5"))
    if not candidates:
        # fall back to first file we can find
        candidates = list((REPO_ROOT / "data" / "INSAT").rglob("*.h5"))
    assert candidates, "no INSAT files available"
    da = read_insat_day(candidates[0])
    stats = validate_insat(da)
    assert da.name == "insat_lst"
    assert stats["nan_pct"] > 40.0                          # ~60% fill expected
    assert stats["min_C"] > -125.0 and stats["max_C"] < 90.0
    # QC map — plot on the native 2-D lat/lon via scatter so we don't accidentally
    # assume a regular grid (the reader intentionally preserves curvilinear coords).
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 5.0), facecolor="#0d1526")
    ax.set_facecolor("#0d1526")
    lat = da.lat2d.values.ravel(); lon = da.lon2d.values.ravel()
    z = da.isel(time=0).values.ravel()
    m = np.isfinite(z)
    sc = ax.scatter(lon[m], lat[m], c=z[m], s=1.0, cmap="turbo", vmin=-5, vmax=45)
    ax.set_xlim(60, 100); ax.set_ylim(0, 40)
    fig.colorbar(sc, ax=ax, label="LST (°C)")
    ax.set_title(f"INSAT LST — {stats['time_first']}", color="#e6edf7")
    ax.tick_params(colors="#8899bb")
    fig.tight_layout(); fig.savefig(QC_DIR / f"insat_{stats['time_first']}.png", dpi=110)
    plt.close(fig)
    print("[insat]", stats)


if __name__ == "__main__":
    # Allow running as a script (skips pytest fixtures).
    test_rainfall_year_2020()
    test_tmax_tmin_year_2020()
    test_insat_single_day()
    print("\nQC maps written to:", QC_DIR)
