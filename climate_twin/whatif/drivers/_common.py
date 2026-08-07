"""
whatif.drivers._common — shared helpers for L0 drivers.

Sentinel masking + IST time-axis + attrs-stamping utilities. Every
driver in this subpackage funnels through these so no downstream code
has to remember "did this get masked?".
"""
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..config.constants import (
    IMD_SENTINELS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    N_LAT,
    N_LON,
    TZ,
    UNITS,
)

IST = timezone(timedelta(hours=5, minutes=30))


def mask_sentinels(a: np.ndarray) -> np.ndarray:
    """Set every IMD sentinel occurrence (−999.0, 99.9) to NaN. Idempotent."""
    out = np.asarray(a, dtype=np.float32)
    for s in IMD_SENTINELS:
        out = np.where(np.isclose(out, s, atol=1e-3), np.nan, out)
    return out


def to_ist_index(times) -> pd.DatetimeIndex:
    """Coerce any date/datetime-like sequence to an IST-aware
    ``DatetimeIndex``. If already tz-aware, converted; otherwise
    localised as UTC-midnight then converted to IST (equivalent to
    stamping a naive date as ``00:00 IST``)."""
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(TZ)
    else:
        idx = idx.tz_convert(TZ)
    return idx


def assert_master_grid(da: xr.DataArray) -> None:
    """Assert that a DataArray lives on the canonical 0.25° India grid."""
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"missing lat/lon dims: got {da.dims}")
    if da.sizes["lat"] != N_LAT or da.sizes["lon"] != N_LON:
        raise ValueError(
            f"grid shape {da.sizes} != master ({N_LAT}, {N_LON})"
        )
    lat0, lat1 = float(da.lat[0]), float(da.lat[-1])
    lon0, lon1 = float(da.lon[0]), float(da.lon[-1])
    if abs(lat0 - LAT_MIN) > 1e-2 or abs(lat1 - LAT_MAX) > 1e-2:
        raise ValueError(f"lat extent {lat0}..{lat1} != {LAT_MIN}..{LAT_MAX}")
    if abs(lon0 - LON_MIN) > 1e-2 or abs(lon1 - LON_MAX) > 1e-2:
        raise ValueError(f"lon extent {lon0}..{lon1} != {LON_MIN}..{LON_MAX}")


def stamp_attrs(
    da: xr.DataArray,
    *,
    var: str,
    source: str,
    source_version: str,
    quantile: str = "deterministic",
    extra: dict | None = None,
) -> xr.DataArray:
    """Add the required attrs (units, source, source_version, quantile)
    every driver output must carry."""
    da = da.copy()
    da.attrs["units"] = UNITS.get(var, "")
    da.attrs["source"] = source
    da.attrs["source_version"] = source_version
    da.attrs["quantile"] = quantile
    if extra:
        da.attrs.update(extra)
    return da


def date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")
