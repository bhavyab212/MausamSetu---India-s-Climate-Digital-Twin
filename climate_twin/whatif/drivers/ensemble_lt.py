"""
whatif.drivers.ensemble_lt — Hawkins-Sutton uncertainty decomposition.

Primary source:
    * Hawkins, E. & Sutton, R. (2009) "The potential to narrow
      uncertainty in regional climate projections", BAMS 90:1095-1107.
      §3 formalises the decomposition into scenario / model / internal
      variance.

Contract (Rule 3):
    Every Long-Term chart's confidence band pulls from this decomposition.
    Panels never invent their own error bars.

    Fractions must sum to 1.0 within 1e-6.

Version: ``hawkins-sutton-v1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

ENSEMBLE_LT_VERSION = "hawkins-sutton-v1"


@dataclass
class UncertaintyDecomposition:
    total_var: xr.DataArray
    scenario_var: xr.DataArray
    model_var: xr.DataArray
    internal_var: xr.DataArray
    fractions: xr.Dataset
    citation: str = "Hawkins & Sutton 2009 BAMS 90:1095"
    version: str = ENSEMBLE_LT_VERSION


def _smooth_along_time(da: xr.DataArray, years: int) -> xr.DataArray:
    """Rolling-mean smoothing along the ``time`` dim (in years)."""
    return da.rolling(time=int(years), min_periods=1, center=True).mean()


def decompose_uncertainty(
    ensemble_by_scenario: dict[str, xr.DataArray],
    smoothing_yrs: int = 10,
) -> UncertaintyDecomposition:
    """Decompose the projection uncertainty into scenario / model /
    internal-variability components.

    Input mapping: ``{ssp: da(model, time, lat, lon)}``.

    * **scenario_var**: variance across the ensemble mean of each SSP.
      i.e., variance across SSPs of ``da.mean("model")``.
    * **model_var**: mean across SSPs of the variance across models
      within each SSP. Uses the *smoothed* series per model to isolate
      the systematic model difference from internal variability.
    * **internal_var**: residual = variance around the smoothed series
      averaged over models and SSPs.
    * **total_var** = scenario_var + model_var + internal_var.
    * **fractions**: each divided by total_var; sum to 1.
    """
    if not ensemble_by_scenario:
        raise ValueError("ensemble_by_scenario is empty")

    # 1) Per-scenario ensemble mean
    scenario_means: list[xr.DataArray] = []
    per_ssp_model_var: list[xr.DataArray] = []
    per_ssp_internal_var: list[xr.DataArray] = []
    for ssp, da in ensemble_by_scenario.items():
        # smoothed signal per model → residual is internal
        smooth = _smooth_along_time(da, years=smoothing_yrs)
        internal = (da - smooth).var("time")            # (model, lat, lon)
        per_ssp_internal_var.append(internal.mean("model"))
        # model_var within this SSP (on the smoothed series)
        per_ssp_model_var.append(smooth.var("model"))     # (time, lat, lon)
        scenario_means.append(smooth.mean("model"))       # (time, lat, lon)

    # Stack along a synthetic 'ssp' dim
    scenario_stack = xr.concat(
        scenario_means, dim="ssp",
    ).assign_coords(ssp=list(ensemble_by_scenario))
    model_var_stack = xr.concat(
        per_ssp_model_var, dim="ssp",
    ).assign_coords(ssp=list(ensemble_by_scenario))
    internal_var_stack = xr.concat(
        per_ssp_internal_var, dim="ssp",
    ).assign_coords(ssp=list(ensemble_by_scenario))

    scenario_var = scenario_stack.var("ssp")
    model_var = model_var_stack.mean("ssp")
    internal_var = internal_var_stack.mean("ssp")
    # Broadcast internal_var over time (it has no time dim; broadcast to model_var's time)
    if "time" in model_var.dims and "time" not in internal_var.dims:
        internal_var = internal_var.expand_dims(
            time=model_var["time"], axis=list(model_var.dims).index("time"),
        )

    total_var = scenario_var + model_var + internal_var
    # Avoid div-by-zero
    denom = xr.where(total_var > 1e-12, total_var, 1e-12)

    fractions = xr.Dataset({
        "scenario": scenario_var / denom,
        "model": model_var / denom,
        "internal": internal_var / denom,
    })
    fractions.attrs["citation"] = "Hawkins & Sutton 2009"
    fractions.attrs["smoothing_yrs"] = int(smoothing_yrs)
    fractions.attrs["version"] = ENSEMBLE_LT_VERSION

    return UncertaintyDecomposition(
        total_var=total_var,
        scenario_var=scenario_var,
        model_var=model_var,
        internal_var=internal_var,
        fractions=fractions,
    )
