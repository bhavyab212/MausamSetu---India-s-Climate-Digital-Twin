"""
imd_rainfall_nc.py — reads IMD gridded daily rainfall NetCDFs.

Files: L:/MausamSetu/data/Rainfall/RF25_ind<YYYY>_rfp25.nc
Native: NetCDF-4, variable RAINFALL (mm/day), coords TIME/LATITUDE/LONGITUDE,
        0.25° India grid (lat 6.5–38.5°N, lon 66.5–100.0°E) 129×135,
        dtype mixed float32/float64 across years, NaN already masks non-land.

Returns: xr.DataArray(name='rain', dims=('time','lat','lon'), units='mm/day').
Enforces: canonical dim/coord names, physical range check, no sentinels survive.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

RAIN_PLAUSIBLE = (0.0, 1200.0)   # mm/day sane range for any single day at 0.25°


def read_rainfall_year(path: str | Path) -> xr.DataArray:
    """Open one year's rainfall NetCDF and return a canonical DataArray.

    Canonical form: name='rain', dims=('time','lat','lon'), attrs {units,long_name,source}.
    Tolerant of the naming variations observed across the IMD drop:
        vars  : 'RAINFALL' | 'rf' | first data_var
        coords: TIME/LATITUDE/LONGITUDE  or  time/lat/lon
    """
    ds = xr.open_dataset(path)
    # variable
    for v_candidate in ("RAINFALL", "rf", "rain", "rainfall"):
        if v_candidate in ds.data_vars:
            var = v_candidate
            break
    else:
        var = next(iter(ds.data_vars))
    # coord renames — only rename names that exist
    rename_map = {}
    for old, new in (("TIME", "time"), ("LATITUDE", "lat"), ("LONGITUDE", "lon")):
        if old in ds[var].dims or old in ds[var].coords:
            rename_map[old] = new
    da = ds[var].rename(rename_map) if rename_map else ds[var]
    da = da.astype(np.float32)          # collapse mixed float32/64 → float32
    # Rainfall files store NaN for ocean/outside-India already; no extra sentinel to mask.
    da.name = "rain"
    da.attrs.update({
        "units": "mm/day",
        "long_name": "IMD gridded daily rainfall",
        "source": str(Path(path).name),
    })
    ds.close()
    return da


def read_rainfall_range(root: str | Path, years: tuple[int, int]) -> xr.DataArray:
    """Concatenate several years into a single (time,lat,lon) DataArray."""
    root = Path(root)
    parts = []
    for y in range(years[0], years[1] + 1):
        p = root / f"RF25_ind{y}_rfp25.nc"
        if not p.exists():
            continue
        parts.append(read_rainfall_year(p))
    if not parts:
        raise FileNotFoundError(f"No rainfall files in {root} for {years}")
    da = xr.concat(parts, dim="time")
    da.name = "rain"
    return da


def validate_rainfall(da: xr.DataArray) -> dict:
    """Return sanity stats + raise on gross violations. Never raises on isolated NaN."""
    v = da.values
    land = ~np.isnan(v)
    stats = {
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
    lo, hi = RAIN_PLAUSIBLE
    survivors = v[land]
    if survivors.size and (survivors.min() < lo - 1e-6 or survivors.max() > hi):
        stats["out_of_range"] = int(((survivors < lo - 1e-6) | (survivors > hi)).sum())
    else:
        stats["out_of_range"] = 0
    assert stats["out_of_range"] == 0, f"Rainfall values outside {RAIN_PLAUSIBLE}"
    # Sentinels: nothing near -999 must survive as a valid value
    assert not np.any(survivors <= -998.0), "Sentinel -999 not masked"
    return stats
