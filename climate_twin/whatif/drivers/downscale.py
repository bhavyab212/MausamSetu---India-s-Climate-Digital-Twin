"""
whatif.drivers.downscale — delta downscaling for the Long-Term driver.

Primary sources:
    * Cannon, A.J., Sobie, S.R. & Murdock, T.Q. (2015) "Bias correction
      of GCM precipitation by quantile mapping: how well do methods
      preserve changes in quantiles and extremes?", J. Climate 28:6938-59.
    * Cannon, A.J. (2018) "Multivariate quantile mapping bias correction:
      an N-dimensional probability density function transform for
      climate model simulations of multiple variables", Clim. Dyn.
      50:31-49.  §2.2 defines QDM.
    * Räisänen, J. & Räty, O. (2013) "Analysis of changes in temperature
      and precipitation using two additional bias correction methods",
      Clim. Dyn. 41:1553-68. §4 documents the delta-mean pitfall for
      extremes.

Rule 5 (Part 7): QDM for extremes, delta_mean for means only. Users
who select delta_mean on an extreme variable must tick a caveat.

Version:
    ``downscale-delta-mean-v1``
    ``downscale-qdm-v1``  (parameters: n_quantiles=100, month_binning=on)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

DELTA_MEAN_VERSION = "downscale-delta-mean-v1"
QDM_VERSION = "downscale-qdm-v1"

_PR_FLOOR_MM = 1e-3        # avoid division blow-ups on dry cells


def delta_downscale_mean(
    obs_baseline: xr.DataArray,
    gcm_baseline: xr.DataArray,
    gcm_future: xr.DataArray,
    variable: Literal["pr", "tas"],
) -> xr.DataArray:
    """Additive delta for temperature; multiplicative for precipitation.

    Preserves the observed mean state; adjusts by the model's change.
    Trivially wrong for extremes — use :func:`qdm_downscale` instead.
    """
    if variable == "pr":
        with np.errstate(invalid="ignore", divide="ignore"):
            gcm_base_mean = gcm_baseline.mean("time")
            ratio = (gcm_future.mean("time")
                     / gcm_base_mean.where(gcm_base_mean > _PR_FLOOR_MM, _PR_FLOOR_MM))
        out = obs_baseline * ratio
    elif variable == "tas":
        delta = gcm_future.mean("time") - gcm_baseline.mean("time")
        out = obs_baseline + delta
    else:
        raise ValueError(f"variable must be 'pr' or 'tas', got {variable!r}")

    out.attrs["downscale_method"] = DELTA_MEAN_VERSION
    out.attrs["downscale_variable"] = variable
    out.attrs["caveat"] = (
        "Delta-mean preserves the observed mean state but does not "
        "adjust the tails. Do not use for extremes (Räisänen & Räty 2013)."
    )
    return out


def _by_month_index(times: xr.DataArray) -> np.ndarray:
    return pd.DatetimeIndex(times.values).month.to_numpy()


def _empirical_quantile(v: np.ndarray, x: float) -> float:
    """P(V ≤ x) — empirical CDF used inside QDM."""
    if v.size == 0:
        return 0.5
    return float(np.searchsorted(np.sort(v), x, side="right") / v.size)


def _empirical_value_at_prob(v: np.ndarray, p: float) -> float:
    if v.size == 0:
        return float("nan")
    p = float(np.clip(p, 0.0, 1.0))
    return float(np.quantile(v, p, method="linear"))


def _qdm_cell_month(
    obs_base: np.ndarray, gcm_base: np.ndarray, gcm_fut: np.ndarray,
    variable: str, n_quantiles: int,
) -> np.ndarray:
    """Cell-wise QDM within a fixed month.  Returns the downscaled
    future series (same length as ``gcm_fut``).
    """
    # Empirical CDFs on the pre-existing training samples
    ob = obs_base[np.isfinite(obs_base)]
    gb = gcm_base[np.isfinite(gcm_base)]
    gf = gcm_fut.astype(np.float64)

    if ob.size < 5 or gb.size < 5 or gf.size < 5:
        # Not enough samples — fall back to raw future values
        return gcm_fut.astype(np.float64)

    out = np.empty_like(gf)
    for i, x in enumerate(gf):
        if not np.isfinite(x):
            out[i] = x
            continue
        p = _empirical_quantile(gf, float(x))
        gb_at_p = _empirical_value_at_prob(gb, p)
        ob_at_p = _empirical_value_at_prob(ob, p)
        if variable == "pr":
            with np.errstate(divide="ignore", invalid="ignore"):
                change = (x / max(gb_at_p, _PR_FLOOR_MM))
            out[i] = ob_at_p * change
        else:                                # tas
            change = x - gb_at_p
            out[i] = ob_at_p + change
    return out


def qdm_downscale(
    obs_baseline: xr.DataArray,
    gcm_baseline: xr.DataArray,
    gcm_future: xr.DataArray,
    variable: Literal["pr", "tas"],
    n_quantiles: int = 100,
) -> xr.DataArray:
    """Quantile Delta Mapping — Cannon (2018).

    All three inputs are daily (time, lat, lon). Month-of-year binning
    preserves seasonality. Cell-by-cell, month-by-month:
        1. Find the non-exceedance probability of x_future in the GCM
           future distribution for that month.
        2. Read the corresponding quantile of the GCM baseline.
        3. Compute change (ratio for pr, delta for tas).
        4. Apply the change to the observed baseline quantile.
    """
    if variable not in ("pr", "tas"):
        raise ValueError(f"variable must be 'pr' or 'tas', got {variable!r}")
    if not {"time", "lat", "lon"} <= set(obs_baseline.dims):
        raise ValueError("obs_baseline needs (time, lat, lon)")

    obs_v = obs_baseline.values.astype(np.float64)
    gcm_b_v = gcm_baseline.values.astype(np.float64)
    gcm_f_v = gcm_future.values.astype(np.float64)
    obs_month = _by_month_index(obs_baseline["time"])
    gcm_b_month = _by_month_index(gcm_baseline["time"])
    gcm_f_month = _by_month_index(gcm_future["time"])
    T_f, H, W = gcm_f_v.shape
    out = np.empty_like(gcm_f_v)

    for m in range(1, 13):
        of = obs_month == m
        gb = gcm_b_month == m
        gf = gcm_f_month == m
        for i in range(H):
            for j in range(W):
                out[np.flatnonzero(gf), i, j] = _qdm_cell_month(
                    obs_v[of, i, j],
                    gcm_b_v[gb, i, j],
                    gcm_f_v[gf, i, j],
                    variable=variable,
                    n_quantiles=n_quantiles,
                )

    da = xr.DataArray(
        out.astype(np.float32),
        dims=gcm_future.dims,
        coords=gcm_future.coords,
        name=gcm_future.name,
    )
    da.attrs.update(gcm_future.attrs)
    da.attrs["downscale_method"] = QDM_VERSION
    da.attrs["downscale_variable"] = variable
    da.attrs["qdm_n_quantiles"] = int(n_quantiles)
    da.attrs["qdm_seasonality"] = "monthly-binned"
    return da


def downscale(
    obs_baseline: xr.DataArray,
    gcm_baseline: xr.DataArray,
    gcm_future: xr.DataArray,
    variable: Literal["pr", "tas"],
    *,
    method: Literal["delta_mean", "qdm"] = "qdm",
    n_quantiles: int = 100,
) -> xr.DataArray:
    """Public façade — dispatch to the requested method."""
    if method == "delta_mean":
        return delta_downscale_mean(
            obs_baseline, gcm_baseline, gcm_future, variable,
        )
    if method == "qdm":
        return qdm_downscale(
            obs_baseline, gcm_baseline, gcm_future, variable,
            n_quantiles=n_quantiles,
        )
    raise ValueError(f"unknown method {method!r}")
