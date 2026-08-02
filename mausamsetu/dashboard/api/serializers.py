"""Convert scientific Python values into strict JSON-compatible values."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import xarray as xr

IST = timezone(timedelta(hours=5, minutes=30))
MISSING_SENTINELS: tuple[float, ...] = (-999.0, 99.9)


# ---------------------------------------------------------------------------
# Missing value + JSON helpers
# ---------------------------------------------------------------------------


def mask_missing(values: Any) -> np.ndarray:
    """Return a float array with project missing-value sentinels masked as NaN."""
    if isinstance(values, np.ma.MaskedArray):
        array = values.filled(np.nan).astype(np.float64, copy=False)
    else:
        array = np.asarray(values, dtype=np.float64).copy()
    for sentinel in MISSING_SENTINELS:
        array[np.isclose(array, sentinel, equal_nan=False)] = np.nan
    return array


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy, masked, date, and non-finite values for JSON."""
    if isinstance(value, np.ma.MaskedArray):
        return to_jsonable(value.filled(np.nan))
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def clean_floats(array: np.ndarray) -> list:
    """Return a nested list where NaN/Inf/sentinel values are converted to ``None``."""
    return to_jsonable(mask_missing(array))


# ---------------------------------------------------------------------------
# IST helpers
# ---------------------------------------------------------------------------


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist(value: np.datetime64 | datetime | date | str) -> datetime:
    """Coerce a datacube timestamp / string / date to an IST-aware ``datetime``.

    The datacube stores midnight UTC; the demo treats each timestamp as an
    IST calendar day. We attach the IST offset without shifting the wall clock.
    """
    if isinstance(value, np.datetime64):
        ts = np.datetime64(value, "s").astype("datetime64[s]").astype("int64")
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(IST).replace(tzinfo=IST)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=IST)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=IST)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=IST)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def as_iso_date(value: np.datetime64 | datetime | date | str) -> str:
    return to_ist(value).date().isoformat()


# ---------------------------------------------------------------------------
# Basin mask + completeness
# ---------------------------------------------------------------------------


def basin_mask(datacube: xr.Dataset) -> np.ndarray:
    """Boolean mask of grid cells that belong to the basin (rain_clim finite)."""
    if "rain_clim" not in datacube.data_vars:
        raise KeyError("cauvery.nc is missing rain_clim")
    finite_per_doy = np.isfinite(mask_missing(datacube["rain_clim"].values))
    return finite_per_doy.any(axis=0)


def completeness(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return valid/total basin cells + completeness percentage."""
    if values.shape != mask.shape:
        raise ValueError(f"Shape mismatch: values {values.shape} vs mask {mask.shape}")
    values_clean = mask_missing(values)
    valid = int(np.isfinite(values_clean[mask]).sum())
    total = int(mask.sum())
    pct = (valid / total * 100.0) if total else 0.0
    return {"valid_cells": valid, "basin_cells": total, "completeness_pct": pct}
