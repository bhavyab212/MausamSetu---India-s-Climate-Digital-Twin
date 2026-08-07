"""
whatif.indices.extremes — Rx1day / Rx5day / R95p / return level.

Primary sources:
    * Rx1day / Rx5day / R95p — WMO/ETCCDI standard indices; Karl,
      Nicholls & Ghazi (1999) and Zhang et al. (2011).
    * Return level via GEV — Coles (2001), *An Introduction to
      Statistical Modeling of Extreme Values*, Springer.

Fits (percentile threshold for R95p, GEV parameters per cell for return
levels) live under ``CACHE_DIR/fits/extremes_*_v1.pkl`` and are
addressed by version string. Fits use ``TRAIN_YEARS`` only.
"""
from __future__ import annotations

import hashlib
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from ..config.paths import CACHE_DIR
from ..drivers.historical import get_historical
from .reference import TRAIN_YEARS, _propagate_attrs, assert_train_only


def rx1day(rain: xr.DataArray, freq: str = "YE") -> xr.DataArray:
    """Max 1-day rainfall per period. Same units as input."""
    out = rain.resample(time=freq).max(skipna=False)
    out.name = "rx1day"
    out = _propagate_attrs(out, rain, "rx1day-v1")
    out.attrs["method"] = "etccdi-rx1day"
    return out


def rx5day(rain: xr.DataArray, freq: str = "YE") -> xr.DataArray:
    """Max 5-day running sum per period."""
    roll5 = rain.rolling(time=5, min_periods=5).sum()
    out = roll5.resample(time=freq).max(skipna=False)
    out.name = "rx5day"
    out = _propagate_attrs(out, rain, "rx5day-v1")
    out.attrs["method"] = "etccdi-rx5day"
    return out


def _r95_threshold(rain_train: xr.DataArray) -> xr.DataArray:
    """95th percentile of wet-day (>= 1 mm) rainfall per cell over
    the TRAIN_YEARS window. Cached to disk."""
    key = f"r95thr|{TRAIN_YEARS}"
    d = hashlib.sha256(key.encode()).hexdigest()[:12]
    cache = CACHE_DIR / "fits" / f"r95thr_{d}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    wet = rain_train.where(rain_train >= 1.0)
    thr = wet.quantile(0.95, dim="time", skipna=True)
    thr.name = "r95_threshold"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(thr, f, protocol=pickle.HIGHEST_PROTOCOL)
    return thr


def r95p(rain: xr.DataArray) -> xr.DataArray:
    """Total rainfall from days above the reference-period wet-day 95th
    percentile. Fit uses TRAIN_YEARS only; scoring can use any year."""
    train = get_historical("rain",
                            date(TRAIN_YEARS[0], 1, 1),
                            date(TRAIN_YEARS[1], 12, 31))
    assert_train_only(np.unique(train["time.year"].values).tolist())
    thr = _r95_threshold(train)
    out = rain.where(rain >= thr).sum(dim="time", skipna=True)
    out.name = "r95p"
    out = _propagate_attrs(out, rain, "r95p@train71-10-v1")
    out.attrs["units"] = rain.attrs.get("units", "mm")
    out.attrs["method"] = "etccdi-r95p"
    out.attrs["reference_period"] = f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}"
    return out


def return_level(rain: xr.DataArray, return_period_yr: int = 100,
                  method: str = "gev") -> xr.DataArray:
    """Return level per grid cell via GEV on annual block maxima.
    Method-specific caveats:
        * "gev" fits scipy.stats.genextreme per cell — slow. Cached.
        * At 0.25° cell scale return levels are noisy; recommend zonal
          aggregation for reporting (see Part 3 sectors)."""
    if method != "gev":
        raise NotImplementedError(f"method={method!r} not implemented")
    from scipy.stats import genextreme

    train = get_historical("rain",
                            date(TRAIN_YEARS[0], 1, 1),
                            date(TRAIN_YEARS[1], 12, 31))
    assert_train_only(np.unique(train["time.year"].values).tolist())
    ann_max = rx1day(train, freq="YE")
    key = f"rl_gev|{TRAIN_YEARS}|{return_period_yr}"
    d = hashlib.sha256(key.encode()).hexdigest()[:12]
    cache = CACHE_DIR / "fits" / f"rl_gev_{d}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    def _fit_cell(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size < 10:
            return float("nan")
        try:
            c, loc, scale = genextreme.fit(arr)
            return float(genextreme.isf(1 / return_period_yr, c, loc, scale))
        except Exception:
            return float("nan")

    rl = xr.apply_ufunc(
        _fit_cell, ann_max,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename(f"rl_{return_period_yr}yr")
    rl = _propagate_attrs(rl, rain, f"rl@gev-{return_period_yr}yr-v1")
    rl.attrs["units"] = rain.attrs.get("units", "mm")
    rl.attrs["method"] = "gev-per-cell-scipy"
    rl.attrs["return_period_yr"] = int(return_period_yr)
    rl.attrs["reference_period"] = f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(rl, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rl
