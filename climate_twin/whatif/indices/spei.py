"""
whatif.indices.spei — Standardized Precipitation-Evapotranspiration Index.

Primary source: Vicente-Serrano, Beguería & López-Moreno (2010),
"A multiscalar drought index sensitive to global warming: The
Standardized Precipitation Evapotranspiration Index",
*Journal of Climate* 23(7), 1696-1718.

Recipe:
    1. Compute daily ``D = P − ET0`` on the master grid (rain + ET0
       from :mod:`whatif.indices.et0_hargreaves`).
    2. Monthly-sum D, then rolling-sum over ``accum_months``.
    3. For each cell × calendar-month, fit a **three-parameter
       log-logistic (Fisk)** distribution over TRAIN_YEARS only.
    4. SPEI(x) = Φ⁻¹(F(x))  — normal quantile of the fitted CDF.

Contracts identical to SPI: version string, fit artifact under
``CACHE_DIR/fits/``, leakage guard at ``fit_spei`` time.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config.paths import CACHE_DIR
from ..drivers.historical import get_historical
from .et0_hargreaves import et0_hargreaves
from .reference import TRAIN_YEARS, _propagate_attrs, assert_train_only


@dataclass
class SPEIFit:
    accum_months: int
    version: str
    c: np.ndarray                        # (12, lat, lon) shape
    loc: np.ndarray                      # (12, lat, lon) location
    scale: np.ndarray                    # (12, lat, lon) scale
    train_years: tuple[int, int]


def _fit_cache_path(accum: int) -> Path:
    v = f"spei{accum}-fisk-v1"
    d = hashlib.sha256(f"{v}|{TRAIN_YEARS}".encode()).hexdigest()[:12]
    return CACHE_DIR / "fits" / f"{v}_{d}.pkl"


def _compute_D(rain: xr.DataArray, tmax: xr.DataArray, tmin: xr.DataArray) -> xr.DataArray:
    et0 = et0_hargreaves(tmax, tmin)
    D = rain - et0
    D.name = "P_minus_ET0"
    return D


def _accumulate_monthly(D: xr.DataArray, accum: int) -> xr.DataArray:
    monthly = D.resample(time="ME").sum(skipna=False)
    if accum == 1:
        return monthly
    return monthly.rolling(time=int(accum), min_periods=int(accum)).sum()


def _fit_cell_fisk(x: np.ndarray) -> tuple[float, float, float]:
    from scipy.stats import fisk
    x = x[np.isfinite(x)]
    if x.size < 10:
        return (np.nan, np.nan, np.nan)
    # SPEI 2010: three-parameter log-logistic — shift so all values > 0
    shift = -x.min() + 1e-3 if x.min() < 0 else 0.0
    y = x + shift
    try:
        c, loc, scale = fisk.fit(y, floc=0)
        return (float(c), float(loc - shift), float(scale))
    except Exception:
        return (np.nan, np.nan, np.nan)


def fit_spei(rain: xr.DataArray, tmax: xr.DataArray, tmin: xr.DataArray,
              accum_months: int,
              train_years: tuple[int, int] = TRAIN_YEARS) -> SPEIFit:
    years = np.arange(int(train_years[0]), int(train_years[1]) + 1).tolist()
    assert_train_only(years)

    cache = _fit_cache_path(accum_months)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    def _slice(da):
        return da.sel(time=slice(str(train_years[0]), str(train_years[1])))
    D = _compute_D(_slice(rain), _slice(tmax), _slice(tmin))
    acc = _accumulate_monthly(D, accum_months)

    H, W = acc.sizes["lat"], acc.sizes["lon"]
    c_arr = np.full((12, H, W), np.nan, dtype=np.float32)
    loc_arr = np.full((12, H, W), np.nan, dtype=np.float32)
    sc_arr = np.full((12, H, W), np.nan, dtype=np.float32)
    months = acc["time.month"].values
    values = acc.values
    for m in range(1, 13):
        pick = months == m
        if not pick.any():
            continue
        block = values[pick]
        for i in range(H):
            for j in range(W):
                c, loc, scale = _fit_cell_fisk(block[:, i, j])
                c_arr[m - 1, i, j] = c
                loc_arr[m - 1, i, j] = loc
                sc_arr[m - 1, i, j] = scale
    fit = SPEIFit(
        accum_months=int(accum_months),
        version=f"spei{accum_months}-fisk-v1",
        c=c_arr, loc=loc_arr, scale=sc_arr,
        train_years=tuple(train_years),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fit


def spei(rain: xr.DataArray, tmax: xr.DataArray, tmin: xr.DataArray,
          fit: SPEIFit) -> xr.DataArray:
    from scipy.stats import fisk, norm
    D = _compute_D(rain, tmax, tmin)
    acc = _accumulate_monthly(D, fit.accum_months)
    values = acc.values
    months = acc["time.month"].values
    out = np.full_like(values, np.nan, dtype=np.float32)
    for m in range(1, 13):
        pick = months == m
        if not pick.any():
            continue
        c_m, loc_m, sc_m = fit.c[m - 1], fit.loc[m - 1], fit.scale[m - 1]
        block = values[pick]
        with np.errstate(invalid="ignore"):
            F = fisk.cdf(block, c=c_m, loc=loc_m, scale=sc_m)
        F = np.clip(F, 1e-6, 1 - 1e-6)
        out[pick] = norm.ppf(F).astype(np.float32)
    da = xr.DataArray(out, dims=acc.dims, coords=acc.coords,
                       name=f"spei_{fit.accum_months}")
    da = _propagate_attrs(da, rain, f"spei{fit.accum_months}@{fit.version}")
    da.attrs["units"] = "σ"
    da.attrs["accum_months"] = int(fit.accum_months)
    da.attrs["method"] = "vicente-serrano-2010-fisk"
    da.attrs["fit_version"] = fit.version
    da.attrs["reference_period"] = f"{fit.train_years[0]}-{fit.train_years[1]}"
    return da
