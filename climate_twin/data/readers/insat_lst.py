"""
insat_lst.py — reads INSAT daily-averaged Land Surface Temperature HDF5.

Files: L:/MausamSetu/data/INSAT/<YYYY>/<Mon>/INSAT_LST_IndiaMean_<DDMMMYYYY>.h5
Native (Phase-1 verified):
    /LST         float32 (974, 1067)   Kelvin, sentinel -999.0
    /Latitude    float64 (974, 1067)   2-D curvilinear
    /Longitude   float64 (974, 1067)   2-D curvilinear
    daily average already applied (per root attr "Description")

This reader:
  - masks -999 → NaN,
  - converts K → °C (canonical unit),
  - returns an xr.DataArray with 2-D (lat2d, lon2d) COORDINATES so the
    downstream regridder can do proper curvilinear→regular sampling.

Cadence: daily. Coverage (Phase 1): 2020-Jan..Oct partial, 2021-Feb..Sep partial.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import re
import numpy as np
import xarray as xr
import h5py

INSAT_FILL = -999.0
LST_PLAUSIBLE_K = (150.0, 360.0)              # very generous LST range in Kelvin
LST_PLAUSIBLE_C = (LST_PLAUSIBLE_K[0] - 273.15, LST_PLAUSIBLE_K[1] - 273.15)

_MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
        "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def _date_from_name(name: str) -> date:
    m = re.search(r"(\d{2})([A-Z]{3})(\d{4})", name)
    if not m:
        raise ValueError(f"Cannot parse INSAT date from filename: {name}")
    d, mo, y = int(m.group(1)), _MON[m.group(2)], int(m.group(3))
    return date(y, mo, d)


def read_insat_day(path: str | Path) -> xr.DataArray:
    """Return a DataArray (row, col) float32 °C with 2-D lat/lon as auxiliary coords."""
    p = Path(path)
    with h5py.File(p, "r") as h:
        lst = h["LST"][...].astype(np.float32)
        lat2d = h["Latitude"][...].astype(np.float32)
        lon2d = h["Longitude"][...].astype(np.float32)
    # Mask sentinel + non-finite
    lst = np.where(lst <= INSAT_FILL + 0.01, np.nan, lst)
    lst = np.where(np.isfinite(lst), lst, np.nan)
    # Also treat physically impossible K as fill (defensive)
    with np.errstate(invalid="ignore"):
        lst = np.where((lst < LST_PLAUSIBLE_K[0]) | (lst > LST_PLAUSIBLE_K[1]), np.nan, lst)
    # K → °C
    lst_c = (lst - 273.15).astype(np.float32)

    dt = _date_from_name(p.name)
    da = xr.DataArray(
        lst_c,
        dims=("row", "col"),
        coords={
            "row": np.arange(lst_c.shape[0], dtype=np.int32),
            "col": np.arange(lst_c.shape[1], dtype=np.int32),
            "lat2d": (("row", "col"), lat2d),
            "lon2d": (("row", "col"), lon2d),
        },
        name="insat_lst",
        attrs={
            "units": "degC",
            "long_name": "INSAT daily-averaged Land Surface Temperature",
            "source": p.name,
            "date": dt.isoformat(),
            "sentinel_masked": INSAT_FILL,
            "curvilinear": 1,
            "spatial_extent_native": "lat 0..41.6°N, lon 54.7..112.1°E (native, needs cropping)",
        },
    )
    da = da.expand_dims(time=[np.datetime64(dt.isoformat(), "D")])
    return da


def read_insat_range(root: str | Path, start: date | None = None,
                     end: date | None = None) -> xr.DataArray:
    """Load every INSAT day under `root` (within [start,end] if provided).
    Concatenates on 'time'. Non-INSAT files are ignored."""
    root = Path(root)
    files = sorted(root.rglob("INSAT_LST_IndiaMean_*.h5"))
    parts = []
    for f in files:
        try:
            dt = _date_from_name(f.name)
        except ValueError:
            continue
        if start and dt < start:
            continue
        if end and dt > end:
            continue
        parts.append(read_insat_day(f))
    if not parts:
        raise FileNotFoundError(f"No INSAT LST files under {root}")
    # Every daily grid uses the same curvilinear frame → concat by time keeps 2-D coords.
    return xr.concat(parts, dim="time")


def validate_insat(da: xr.DataArray) -> dict:
    v = da.values
    survivors = v[np.isfinite(v)]
    stats = {
        "shape": tuple(da.shape),
        "time_first": str(da.time.values[0])[:10],
        "time_last": str(da.time.values[-1])[:10],
        "n_time": int(da.sizes["time"]),
        "nan_pct": float(100.0 * np.isnan(v).mean()),
        "min_C": float(np.nanmin(v)) if survivors.size else float("nan"),
        "max_C": float(np.nanmax(v)) if survivors.size else float("nan"),
        "mean_C": float(np.nanmean(v)) if survivors.size else float("nan"),
    }
    lo, hi = LST_PLAUSIBLE_C
    oor = int(((survivors < lo) | (survivors > hi)).sum()) if survivors.size else 0
    stats["out_of_range"] = oor
    assert oor == 0, f"INSAT LST °C outside {LST_PLAUSIBLE_C}: {oor}"
    assert not np.any(survivors <= INSAT_FILL + 1e-3), "INSAT sentinel -999 survived"
    return stats
