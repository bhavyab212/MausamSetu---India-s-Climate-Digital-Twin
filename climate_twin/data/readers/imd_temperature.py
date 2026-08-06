"""
imd_temperature.py — reads IMD gridded daily Tmax / Tmin.

Files (canonical raw): L:/MausamSetu/data/max_temp/Maxtemp_MaxT_<YYYY>.GRD
                      L:/MausamSetu/data/min_temp/Mintemp_MinT_<YYYY>.GRD
Native (Phase-1 verified):
    dtype    float32
    order    C-major  (days, lat=31, lon=31)
    grid     lat 7.5..37.5 (n=31, step 1°), lon 67.5..97.5 (n=31, step 1°)
    sentinel 99.9  (mask values >= 90.0 to NaN)
    unit     °C

We deliberately read the .GRD (canonical) — the sibling .nc still contains 99.9
values unmasked, so it would be a foot-gun.

Returns: xr.DataArray(name='tmax' | 'tmin', dims=('time','lat','lon'), units='°C').
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import calendar
import numpy as np
import xarray as xr

TEMP_NLAT, TEMP_NLON = 31, 31
TEMP_LAT0, TEMP_LAT1 = 7.5, 37.5
TEMP_LON0, TEMP_LON1 = 67.5, 97.5
TEMP_FILL_GE = 90.0                    # any value >= 90 is the 99.9 sentinel
TEMP_PLAUSIBLE = (-45.0, 55.0)         # °C


def _year_days(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def _dates_for(year: int) -> np.ndarray:
    n = _year_days(year)
    return np.array([date(year, 1, 1) + timedelta(days=i) for i in range(n)],
                    dtype="datetime64[D]")


def _read_grd_year(path: Path, year: int) -> np.ndarray:
    """Return (days, 31, 31) float64 array, 99.9 → NaN."""
    raw = np.fromfile(path, dtype=np.float32)
    days = _year_days(year)
    expect = days * TEMP_NLAT * TEMP_NLON
    if raw.size != expect:
        raise ValueError(f"{path.name}: expected {expect} float32, got {raw.size}")
    arr = raw.reshape(days, TEMP_NLAT, TEMP_NLON).astype(np.float64)
    arr[arr >= TEMP_FILL_GE] = np.nan
    return arr


def read_tmax_year(path: str | Path) -> xr.DataArray:
    return _read_temp_year(path, kind="tmax")


def read_tmin_year(path: str | Path) -> xr.DataArray:
    return _read_temp_year(path, kind="tmin")


def _read_temp_year(path: str | Path, kind: str) -> xr.DataArray:
    p = Path(path)
    year = int(p.stem.rsplit("_", 1)[-1])
    arr = _read_grd_year(p, year)
    lat = np.linspace(TEMP_LAT0, TEMP_LAT1, TEMP_NLAT)
    lon = np.linspace(TEMP_LON0, TEMP_LON1, TEMP_NLON)
    da = xr.DataArray(
        arr.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": _dates_for(year), "lat": lat, "lon": lon},
        name=kind,
        attrs={
            "units": "degC",
            "long_name": ("IMD gridded daily maximum temperature" if kind == "tmax"
                          else "IMD gridded daily minimum temperature"),
            "source": p.name,
            "native_resolution_deg": 1.0,
            "sentinel_masked": 99.9,
        },
    )
    return da


def read_temp_range(root: str | Path, years: tuple[int, int], kind: str) -> xr.DataArray:
    """Concatenate several years for kind ∈ {'tmax','tmin'}."""
    root = Path(root)
    prefix = "Maxtemp_MaxT" if kind == "tmax" else "Mintemp_MinT"
    parts = []
    for y in range(years[0], years[1] + 1):
        p = root / f"{prefix}_{y}.GRD"
        if not p.exists():
            continue
        parts.append(_read_temp_year(p, kind))
    if not parts:
        raise FileNotFoundError(f"No {kind} .GRD in {root} for {years}")
    da = xr.concat(parts, dim="time")
    da.name = kind
    return da


def validate_temp(da: xr.DataArray) -> dict:
    """Physical + structural sanity. Asserts no 99.9 sentinel survived."""
    v = da.values
    land = ~np.isnan(v)
    survivors = v[land]
    stats = {
        "kind": da.name,
        "shape": tuple(da.shape),
        "lat_range": (float(da.lat.min()), float(da.lat.max())),
        "lon_range": (float(da.lon.min()), float(da.lon.max())),
        "time_first": str(da.time.values[0])[:10],
        "time_last": str(da.time.values[-1])[:10],
        "n_time": int(da.sizes["time"]),
        "nan_pct": float(100.0 * np.isnan(v).mean()),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
    }
    lo, hi = TEMP_PLAUSIBLE
    stats["out_of_range"] = int(((survivors < lo) | (survivors > hi)).sum()) if survivors.size else 0
    assert stats["out_of_range"] == 0, f"{da.name} out of {TEMP_PLAUSIBLE}: {stats['out_of_range']}"
    assert not np.any(survivors >= TEMP_FILL_GE - 0.01), f"{da.name}: 99.9 sentinel survived"
    return stats
