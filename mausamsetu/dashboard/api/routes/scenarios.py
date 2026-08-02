"""Storyline (scenario) routes."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException

from mausamsetu import config
from mausamsetu.dashboard.api.deps import (
    get_cache,
    get_datacube,
    get_model,
)
from mausamsetu.dashboard.api.forecast_service import cached_forecast
from mausamsetu.dashboard.api.impact_service import scenario_impacts
from mausamsetu.dashboard.api.schemas import (
    ScenarioImpactHeat,
    ScenarioImpactHydrology,
    ScenarioImpactRupee,
    ScenarioImpacts,
    ScenarioPreset,
    ScenarioPresetsResponse,
    ScenarioRunRequest,
    ScenarioRunResponse,
    ScenarioSeriesBlock,
)
from mausamsetu.dashboard.api.serializers import clean_floats, to_ist
from mausamsetu.storyline.scenario import Scenario, apply_scenario

router = APIRouter(tags=["scenarios"])


@router.get("/scenarios/presets", response_model=ScenarioPresetsResponse)
def list_presets() -> ScenarioPresetsResponse:
    presets = [
        ScenarioPreset(
            label=label,
            delta_temp_c=float(values["temp"]),
            delta_rain_pct=float(values["rain_pct"]),
        )
        for label, values in config.IPCC_DELTAS.items()
    ]
    return ScenarioPresetsResponse(presets=presets)


def _basin_mean_series(field: np.ndarray) -> list[float]:
    reshaped = field.reshape(field.shape[0], -1)
    means = np.nanmean(reshaped, axis=1)
    return clean_floats(means)


def _build_forecast_baseline(
    datacube: xr.Dataset,
    model,
    cache: dict,
    start: np.datetime64,
    horizon: int,
) -> xr.Dataset:
    result = cached_forecast(cache, model, datacube, start, horizon)
    return xr.Dataset(
        {
            "rain": (("time", "lat", "lon"), result.quantiles["rain"]["p50"]),
            "tmax": (("time", "lat", "lon"), result.quantiles["tmax"]["p50"]),
            "tmin": (("time", "lat", "lon"), result.quantiles["tmin"]["p50"]),
        },
        coords={
            "time": result.dates,
            "lat": datacube.lat.values,
            "lon": datacube.lon.values,
        },
    )


@router.post("/scenarios/run", response_model=ScenarioRunResponse)
def run_scenario(
    payload: ScenarioRunRequest,
    datacube: xr.Dataset = Depends(get_datacube),
    model=Depends(get_model),
    cache: dict = Depends(get_cache),
) -> ScenarioRunResponse:
    times = datacube.time.values
    start_np = np.datetime64(payload.start_date.isoformat())
    if start_np < times[0] or start_np > times[-1]:
        raise HTTPException(status_code=400, detail=f"start_date outside datacube range")

    if payload.baseline_mode == "historical":
        if payload.days is None:
            raise HTTPException(status_code=400, detail="historical mode requires 'days'")
        end_np = start_np + np.timedelta64(payload.days - 1, "D")
        if end_np > times[-1]:
            raise HTTPException(status_code=400, detail="historical window exceeds datacube")
        baseline = datacube[["rain", "tmax", "tmin"]].sel(
            time=slice(str(start_np)[:10], str(end_np)[:10])
        )
    else:
        horizon = payload.horizon or config.FORECAST_DAYS
        try:
            baseline = _build_forecast_baseline(datacube, model, cache, start_np, horizon)
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    scenario = Scenario(
        delta_temp=float(payload.delta_temp_c),
        delta_rain_pct=float(payload.delta_rain_pct),
        seasonal_months=payload.seasonal_months,
        label=payload.label,
    )
    scenario_ds = apply_scenario(baseline, scenario)
    impacts = scenario_impacts(datacube, baseline, scenario_ds)

    baseline_blocks = {
        var: ScenarioSeriesBlock(
            unit=("mm/day" if var == "rain" else "degC"),
            basin_mean=_basin_mean_series(baseline[var].values),
        )
        for var in ("rain", "tmax", "tmin")
    }
    scenario_blocks = {
        var: ScenarioSeriesBlock(
            unit=("mm/day" if var == "rain" else "degC"),
            basin_mean=_basin_mean_series(scenario_ds[var].values),
        )
        for var in ("rain", "tmax", "tmin")
    }
    delta_blocks = {
        var: ScenarioSeriesBlock(
            unit=("mm/day" if var == "rain" else "degC"),
            basin_mean=_basin_mean_series(scenario_ds[var].values - baseline[var].values),
        )
        for var in ("rain", "tmax", "tmin")
    }

    return ScenarioRunResponse(
        baseline_mode=payload.baseline_mode,
        start_date_ist=to_ist(start_np),
        dates=[np.datetime64(d, "D").astype("O") for d in baseline.time.values],
        label=payload.label,
        delta_temp_c=float(payload.delta_temp_c),
        delta_rain_pct=float(payload.delta_rain_pct),
        baseline=baseline_blocks,
        scenario=scenario_blocks,
        delta=delta_blocks,
        impacts=ScenarioImpacts(
            hydrology=ScenarioImpactHydrology(**impacts["hydrology"]),
            heat=ScenarioImpactHeat(**impacts["heat"]),
            rupee=ScenarioImpactRupee(
                total_crore=impacts["rupee"]["total_crore"],
                rainfall_component_crore=impacts["rupee"]["rainfall_component_crore"],
                heat_component_crore=impacts["rupee"]["heat_component_crore"],
            ),
        ),
    )
