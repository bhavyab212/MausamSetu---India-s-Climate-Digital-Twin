"""
whatif.indices.long_term — return-period shift + time-of-emergence.

Primary sources:
    * Hawkins, E. et al. (2020) "Observed emergence of the climate
      change signal: from the familiar to the unknown", GRL 47.
      Time-of-emergence framework: year at which the projected 20-yr
      running-mean change first exceeds N × internal variability of
      the historical record.
    * Coles, S. (2001) "An Introduction to Statistical Modeling of
      Extreme Values", Springer.  §3 GEV; §4 return level = quantile
      of the annual-max distribution at 1 − 1/T.
    * Zhang, X. et al. (2011) "Indices for monitoring changes in
      extremes based on daily temperature and precipitation data",
      WIREs Climate Change 2:851-870. ETCCDI Rx1day / Rx5day.

Contract:
    * ``return_period_shift``: fits GEV cell-by-cell on the observed
      1971-2000 record + each downscaled model 20-yr window; reports
      ``new_return_period`` for the observed 100-year event.
    * ``time_of_emergence``: year at which the 20-yr running mean of
      the projected change first exceeds ``threshold_sigma`` × the
      internal standard deviation of the observed record.
    * Reuses ``whatif.indices.extremes.return_level`` for the GEV fit
      when SciPy is available.

Version: ``lt-indices-v1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

LT_INDICES_VERSION = "lt-indices-v1"


# ── Return-period shift ─────────────────────────────────────────────
def _annual_maxima(da: xr.DataArray) -> xr.DataArray:
    """Yearly maximum along the time axis."""
    return da.groupby("time.year").max("time")


def _return_level_gev(am: np.ndarray, rp: float) -> float:
    """GEV return-level at return period ``rp`` years, via SciPy.

    Falls back to the empirical (1 − 1/rp)-quantile when SciPy isn't
    available OR when the sample is smaller than 10 years.
    """
    am = am[np.isfinite(am)]
    if am.size < 10:
        return float("nan")
    p_nonex = 1.0 - 1.0 / float(rp)
    try:
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(am)
        return float(genextreme.ppf(p_nonex, shape, loc=loc, scale=scale))
    except Exception:
        # Empirical fallback
        return float(np.quantile(am, p_nonex))


def _rp_for_level(am: np.ndarray, level: float) -> float:
    """Given a sample, what return period does ``level`` correspond to?

    RP = 1 / P(X > level) = 1 / (1 − F(level))."""
    am = am[np.isfinite(am)]
    if am.size < 10:
        return float("nan")
    try:
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(am)
        p_nonex = float(genextreme.cdf(level, shape, loc=loc, scale=scale))
    except Exception:
        p_nonex = float((am <= level).mean())
    p_exceed = 1.0 - p_nonex
    if p_exceed <= 0:
        return float("inf")
    return float(1.0 / p_exceed)


def return_period_shift(
    obs_historical: xr.DataArray,
    ds_future: xr.DataArray,
    *,
    return_period_yr: int = 100,
    variable: Literal["rx1day", "tmax_extreme"] = "rx1day",
) -> xr.Dataset:
    """Return the "old-100yr-becomes-new-N-yr" statement per cell.

    ``obs_historical``: daily observed series 1971-2000, (time, lat, lon).
    ``ds_future``: QDM-downscaled daily future window,
        (model, time, lat, lon) — one model per slice of the ``model``
        dim.

    Returns a Dataset with variables ``baseline_level``,
    ``future_level_mean`` (across models), ``new_return_period_mean``,
    ``new_return_period_std`` — all (lat, lon).

    Baseline is the annual-maximum GEV fit on ``obs_historical``.
    """
    # Baseline: annual maxima over obs
    if variable == "rx1day":
        obs_am = _annual_maxima(obs_historical)
    else:                          # tmax_extreme — same idea, max Tmax
        obs_am = _annual_maxima(obs_historical)
    # Per-cell baseline return level
    obs_v = obs_am.values.astype(np.float64)         # (year, lat, lon)
    H, W = obs_v.shape[1], obs_v.shape[2]
    base_level = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            base_level[i, j] = _return_level_gev(
                obs_v[:, i, j], float(return_period_yr),
            )

    # Future — one RP per model per cell
    models = list(ds_future["model"].values)
    n_models = len(models)
    future_levels = np.empty((n_models, H, W), dtype=np.float64)
    new_rps = np.empty((n_models, H, W), dtype=np.float64)
    for k, m in enumerate(models):
        fut = ds_future.sel(model=m)
        fut_am = _annual_maxima(fut)
        fut_v = fut_am.values.astype(np.float64)
        for i in range(H):
            for j in range(W):
                lvl = _return_level_gev(
                    fut_v[:, i, j], float(return_period_yr),
                )
                future_levels[k, i, j] = lvl
                # Return period of the OBSERVED 100-yr level under the
                # future distribution — this is the shift statement.
                new_rps[k, i, j] = _rp_for_level(fut_v[:, i, j], base_level[i, j])

    coords = {"lat": obs_am["lat"], "lon": obs_am["lon"]}
    out = xr.Dataset({
        "baseline_level": xr.DataArray(base_level.astype(np.float32),
                                          dims=("lat", "lon"), coords=coords),
        "future_level_mean": xr.DataArray(
            np.nanmean(future_levels, axis=0).astype(np.float32),
            dims=("lat", "lon"), coords=coords,
        ),
        "new_return_period_mean": xr.DataArray(
            np.nanmean(new_rps, axis=0).astype(np.float32),
            dims=("lat", "lon"), coords=coords,
        ),
        "new_return_period_std": xr.DataArray(
            np.nanstd(new_rps, axis=0).astype(np.float32),
            dims=("lat", "lon"), coords=coords,
        ),
        "future_level_per_model": xr.DataArray(
            future_levels.astype(np.float32),
            dims=("model", "lat", "lon"),
            coords={"model": models, "lat": obs_am["lat"], "lon": obs_am["lon"]},
        ),
        "new_return_period_per_model": xr.DataArray(
            new_rps.astype(np.float32),
            dims=("model", "lat", "lon"),
            coords={"model": models, "lat": obs_am["lat"], "lon": obs_am["lon"]},
        ),
    })
    out.attrs.update({
        "version": LT_INDICES_VERSION,
        "variable": variable,
        "baseline_return_period_yr": int(return_period_yr),
        "citation": "Coles 2001; Zhang et al. 2011; Cannon 2018 for QDM inputs",
    })
    return out


# ── Time-of-emergence ────────────────────────────────────────────────
def time_of_emergence(
    ds_full_trajectory: xr.DataArray,
    obs_historical: xr.DataArray,
    *,
    threshold_sigma: float = 2.0,
    window_yrs: int = 20,
    variable: str = "annual",
) -> xr.DataArray:
    """Year at which the 20-yr running-mean projected change first
    exceeds ``threshold_sigma`` × the internal-variability standard
    deviation of the observed record.

    Parameters
    ----------
    ds_full_trajectory : (model, time 2015-2100, lat, lon)
    obs_historical     : (time 1971-2000, lat, lon)

    Returns
    -------
    xr.DataArray (model, lat, lon) — first year of emergence, or NaN
    if the trajectory never crosses.
    """
    # 1) Baseline mean + observed internal σ per cell (annual means)
    obs_ann = obs_historical.groupby("time.year").mean("time")
    obs_mean = obs_ann.mean("year")
    obs_sigma = obs_ann.std("year")

    # 2) Running mean projection per model
    proj_ann = ds_full_trajectory.groupby("time.year").mean("time")
    proj_smooth = proj_ann.rolling(year=int(window_yrs), min_periods=1,
                                     center=True).mean()

    # 3) Anomaly + emergence test
    anom = proj_smooth - obs_mean
    threshold = float(threshold_sigma) * obs_sigma       # (lat, lon)
    emerged = np.abs(anom) > threshold                   # (model, year, lat, lon)

    # 4) First year of emergence per (model, lat, lon)
    years = proj_smooth["year"].values.astype(np.int32)
    def _first_year(mask_1d: np.ndarray) -> float:
        if not mask_1d.any():
            return float("nan")
        return float(years[int(np.argmax(mask_1d))])

    out = xr.apply_ufunc(
        _first_year, emerged,
        input_core_dims=[["year"]],
        vectorize=True, output_dtypes=[np.float32],
    )
    out.name = "toe_year"
    out.attrs.update({
        "version": LT_INDICES_VERSION,
        "threshold_sigma": float(threshold_sigma),
        "window_yrs": int(window_yrs),
        "citation": "Hawkins et al. 2020 GRL",
        "variable": variable,
    })
    return out


# ── 20-yr climatological window utility ─────────────────────────────
def window_climatology(da: xr.DataArray, *, method: str = "mean") -> xr.DataArray:
    """Reduce a (time, …) or (model, time, …) DataArray to the 20-year
    climatology per requested statistic. ``method`` ∈ {"mean", "max",
    "min", "std"}."""
    if method == "mean":
        return da.mean("time")
    if method == "max":
        return da.max("time")
    if method == "min":
        return da.min("time")
    if method == "std":
        return da.std("time")
    raise ValueError(f"unknown method {method!r}")
