"""
whatif.indices.spi — Standardized Precipitation Index.

Primary sources:
    * McKee, Doesken & Kleist (1993), "The Relationship of Drought
      Frequency and Duration to Time Scales", 8th Conf. on Applied
      Climatology, AMS.
    * WMO SPI User Guide (WMO-No. 1090, 2012).

Recipe:
    1. Accumulate monthly rainfall to ``accum_months`` (typically 1,
       3, 6, 12).
    2. For each grid cell × calendar-month bin, fit a two-parameter
       Gamma to the non-zero accumulations, and record the zero-
       fraction ``q``.
    3. The mixed CDF is  H(x) = q + (1 − q) * G(x)  where G is the
       fitted Gamma CDF (McKee 1993 §3.2 / WMO 1090 eq. 3-5).
    4. SPI(x) = Φ⁻¹(H(x))  — the standard-normal quantile.

Contracts:
    * Fits use TRAIN_YEARS only. Enforced by :func:`assert_train_only`.
    * The fit artifact carries a version string ("spi<N>-gamma-mixed-v1")
      and is saved under ``CACHE_DIR/fits/``.
    * A correctly-fit SPI, evaluated on TRAIN_YEARS, has cross-cell
      mean ≈ 0 and std ≈ 1. The Part-2 test suite enforces this to
      within 0.05.
"""
from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config.paths import CACHE_DIR
from ..drivers.historical import get_historical
from .reference import TRAIN_YEARS, _propagate_attrs, assert_train_only


@dataclass
class SPIFit:
    """Versioned SPI fit artifact.

    Stored as pickle under ``CACHE_DIR/fits/spi<accum>_v<n>.pkl``.
    ``shape``/``scale`` are per (calendar month, lat, lon); ``q_zero``
    is per (calendar month, lat, lon); ``version`` is the string that
    appears in every scenario's provenance."""
    accum_months: int
    version: str
    shape: np.ndarray                   # (12, lat, lon)  gamma shape (k)
    scale: np.ndarray                   # (12, lat, lon)  gamma scale (θ)
    q_zero: np.ndarray                  # (12, lat, lon)  P(x == 0)
    train_years: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accum_months": self.accum_months,
            "version": self.version,
            "shape": self.shape,
            "scale": self.scale,
            "q_zero": self.q_zero,
            "train_years": self.train_years,
        }


def _fit_cache_path(accum: int) -> Path:
    v = "spi{a}-gamma-mixed-v1".format(a=accum)
    d = hashlib.sha256(f"{v}|{TRAIN_YEARS}".encode()).hexdigest()[:12]
    return CACHE_DIR / "fits" / f"{v}_{d}.pkl"


def _accumulate(rain: xr.DataArray, accum_months: int) -> xr.DataArray:
    """Monthly-sum resample then rolling-sum over ``accum_months``."""
    monthly = rain.resample(time="ME").sum(skipna=False)
    if accum_months == 1:
        return monthly
    return monthly.rolling(time=int(accum_months), min_periods=int(accum_months)).sum()


def _fit_cell_gamma(x: np.ndarray) -> tuple[float, float, float]:
    """Return (shape, scale, q_zero) for one cell's accumulations."""
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    zeros = int((x == 0).sum())
    q0 = zeros / n
    x_pos = x[x > 0]
    if x_pos.size < 5:
        return (np.nan, np.nan, q0)
    # MLE via scipy — keep it robust
    from scipy.stats import gamma
    try:
        # Fix loc=0 to keep the model 2-parameter
        shape_, loc_, scale_ = gamma.fit(x_pos, floc=0)
        if scale_ <= 0 or shape_ <= 0 or not np.isfinite(shape_ * scale_):
            return (np.nan, np.nan, q0)
        return (float(shape_), float(scale_), q0)
    except Exception:
        return (np.nan, np.nan, q0)


def fit_spi(rain: xr.DataArray, accum_months: int,
             train_years: tuple[int, int] = TRAIN_YEARS) -> SPIFit:
    """Fit a mixed-Gamma SPI model over ``train_years`` only.

    :raises LeakageError: if ``train_years`` extends outside TRAIN_YEARS.
    """
    # Enforce leakage guard
    years = np.arange(int(train_years[0]), int(train_years[1]) + 1).tolist()
    assert_train_only(years)

    cache = _fit_cache_path(accum_months)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Slice inputs to train years, accumulate
    slc = rain.sel(time=slice(str(train_years[0]), str(train_years[1])))
    acc = _accumulate(slc, accum_months)                 # (time, lat, lon)

    # Group by calendar month (1..12), gather along time, fit per cell
    H, W = acc.sizes["lat"], acc.sizes["lon"]
    shape_arr = np.full((12, H, W), np.nan, dtype=np.float32)
    scale_arr = np.full((12, H, W), np.nan, dtype=np.float32)
    q0_arr = np.full((12, H, W), np.nan, dtype=np.float32)

    months = acc["time.month"].values
    values = acc.values                                  # (T, H, W)
    for m in range(1, 13):
        pick = months == m
        if not pick.any():
            continue
        block = values[pick]                             # (T_m, H, W)
        # apply per-cell fit
        for i in range(H):
            for j in range(W):
                s, sc, q0 = _fit_cell_gamma(block[:, i, j])
                shape_arr[m - 1, i, j] = s
                scale_arr[m - 1, i, j] = sc
                q0_arr[m - 1, i, j] = q0

    fit = SPIFit(
        accum_months=int(accum_months),
        version=f"spi{accum_months}-gamma-mixed-v1",
        shape=shape_arr,
        scale=scale_arr,
        q_zero=q0_arr,
        train_years=tuple(train_years),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fit


def spi(rain: xr.DataArray, fit: SPIFit) -> xr.DataArray:
    """Apply a pre-fit SPI model to any period. Returns a (time, lat, lon)
    unitless σ DataArray. Any year is fine — leakage guard fires only
    at :func:`fit_spi` time."""
    from scipy.stats import gamma, norm

    acc = _accumulate(rain, fit.accum_months)
    months = acc["time.month"].values
    values = acc.values.astype(np.float64)
    out = np.full_like(values, np.nan, dtype=np.float32)

    for m in range(1, 13):
        pick = months == m
        if not pick.any():
            continue
        shape_m = fit.shape[m - 1]
        scale_m = fit.scale[m - 1]
        q0_m = fit.q_zero[m - 1]
        block = values[pick]                             # (T_m, H, W)

        # G(x) for x > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(block > 0,
                          gamma.cdf(block, a=shape_m, scale=scale_m),
                          0.0)
        H = q0_m[None] + (1.0 - q0_m[None]) * g          # mixed CDF
        # clip to open interval to avoid ±inf in norm.ppf
        H = np.clip(H, 1e-6, 1 - 1e-6)
        z = norm.ppf(H)
        out[pick] = z.astype(np.float32)

    da = xr.DataArray(
        out, dims=acc.dims, coords=acc.coords, name=f"spi_{fit.accum_months}"
    )
    da = _propagate_attrs(da, rain, f"spi{fit.accum_months}@{fit.version}")
    da.attrs["units"] = "σ"
    da.attrs["accum_months"] = int(fit.accum_months)
    da.attrs["method"] = "mckee-1993-mixed-gamma"
    da.attrs["fit_version"] = fit.version
    da.attrs["reference_period"] = f"{fit.train_years[0]}-{fit.train_years[1]}"
    return da


def fit_spi_from_cube(accum_months: int) -> SPIFit:
    """Convenience: read TRAIN_YEARS rain from the cube + fit."""
    rain = get_historical(
        "rain",
        date(TRAIN_YEARS[0], 1, 1),
        date(TRAIN_YEARS[1], 12, 31),
    )
    return fit_spi(rain, accum_months=accum_months, train_years=TRAIN_YEARS)
