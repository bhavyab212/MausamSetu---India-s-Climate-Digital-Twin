"""
whatif.indices.reference — reference period + climatology utilities.

Statistical indices (SPI, SPEI, r95p, return levels, IMD heatwave normals)
need a fixed reference climatology. Two rules govern this file:

    1. ``TRAIN_YEARS = (1971, 2010)`` — 40-year WMO-standard reference
       (WMO-No. 1203, *Guidelines on the Calculation of Climate Normals*).
       This is the ONLY window any ``.fit(...)`` in this layer may see.
    2. ``VALID_YEARS = (2011, 2022)`` — held out for scoring / narrative.
       Fitting on VALID_YEARS is silent leakage and is the same bug that
       the model-side rule bans. :class:`LeakageError` enforces it.

The `climatology(var, doy_window=5)` helper produces a (366, lat, lon)
DOY climatology smoothed with a ±5-day rolling mean (standard practice
to reduce sampling noise). Cached to ``CACHE_DIR/climatology/``.

Nothing in this file changes behaviour based on the DATA. Only on the
years the user requests.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from ..config.paths import CACHE_DIR
from ..drivers.historical import get_historical

TRAIN_YEARS: tuple[int, int] = (1971, 2010)
VALID_YEARS: tuple[int, int] = (2011, 2022)


class LeakageError(RuntimeError):
    """Raised when a fit routine is asked to include years outside TRAIN_YEARS."""


def assert_train_only(years: Iterable[int]) -> None:
    """Refuse any fit whose years touch VALID_YEARS or extend past TRAIN_YEARS."""
    for y in years:
        if not (TRAIN_YEARS[0] <= int(y) <= TRAIN_YEARS[1]):
            raise LeakageError(
                f"fit requested year {y}; only {TRAIN_YEARS} is allowed."
            )


# ---------------------------------------------------------------------------
def _clim_cache_path(var: str, doy_window: int) -> Path:
    key = f"clim|{var}|{TRAIN_YEARS}|{doy_window}"
    d = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / "climatology" / f"{var}_{d}.pkl"


def climatology(var: str, doy_window: int = 5) -> xr.DataArray:
    """Return the ±``doy_window``-day-smoothed DOY climatology for ``var``
    over ``TRAIN_YEARS``. Output dims ``(dayofyear, lat, lon)``,
    366 DOYs (Feb 29 → mean of surrounding days on non-leap years).

    Cached — subsequent calls are ~O(ms)."""
    cache = _clim_cache_path(var, doy_window)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    da = get_historical(var, date(TRAIN_YEARS[0], 1, 1),
                         date(TRAIN_YEARS[1], 12, 31))
    # groupby DOY, take the mean over years, then smooth with a
    # ``2*doy_window+1`` rolling mean along the doy axis.
    grouped = da.groupby("time.dayofyear").mean("time", skipna=True)
    if doy_window > 0:
        grouped = grouped.pad(
            dayofyear=doy_window, mode="wrap"
        ).rolling(dayofyear=2 * doy_window + 1, center=True, min_periods=1
                    ).mean().isel(dayofyear=slice(doy_window, doy_window + 366))
    out = grouped.rename(f"{var}_clim")
    out.attrs["reference_period"] = f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}"
    out.attrs["doy_window"] = int(doy_window)
    out.attrs["source"] = "whatif.indices.reference.climatology"
    out.attrs["units"] = da.attrs.get("units", "")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def zscore(da: xr.DataArray, clim: xr.DataArray,
            sigma: xr.DataArray) -> xr.DataArray:
    """Standardised anomaly = (da − clim) / σ, DOY-aligned.

    Callers are responsible for producing ``clim`` and ``sigma`` on
    matching DOY axes (:func:`climatology` returns clim; you compute σ
    once by grouping over the same DOY axis)."""
    doys = da["time.dayofyear"]
    aligned_clim = clim.sel(dayofyear=doys)
    aligned_sigma = sigma.sel(dayofyear=doys)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (da - aligned_clim) / aligned_sigma
    return _propagate_attrs(out, da, "zscore-v1")


def _propagate_attrs(new: xr.DataArray, src: xr.DataArray, tag: str) -> xr.DataArray:
    """Append `tag` to source_chain and preserve units/quantile."""
    chain = str(src.attrs.get("source_chain", src.attrs.get("source", "")))
    if chain:
        chain = f"{chain} → indices.{tag}"
    else:
        chain = f"indices.{tag}"
    new.attrs["source_chain"] = chain
    new.attrs["quantile"] = src.attrs.get("quantile", "deterministic")
    return new


def valid_fraction(da: xr.DataArray, mask: xr.DataArray | np.ndarray | None = None) -> float:
    """Fraction of finite cells (optionally within a land mask). Used
    to raise a warning when it drops below 0.9 on an all-India window."""
    arr = np.asarray(da.values)
    finite = np.isfinite(arr)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        finite = finite & np.broadcast_to(m, finite.shape)
        denom = np.broadcast_to(m, finite.shape).sum()
    else:
        denom = arr.size
    return float(finite.sum() / max(int(denom), 1))
