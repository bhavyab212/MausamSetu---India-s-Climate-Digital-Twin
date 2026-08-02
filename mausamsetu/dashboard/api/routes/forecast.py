"""Forecast + assimilation routes."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from mausamsetu import config
from mausamsetu.assimilate.enkf import enkf_update
from mausamsetu.dashboard.api.deps import (
    get_cache,
    get_datacube,
    get_model,
    resolve_date,
)
from mausamsetu.dashboard.api.forecast_service import (
    UNITS,
    cached_forecast,
)
from mausamsetu.dashboard.api.schemas import (
    AssimilationResponse,
    ForecastResponse,
    ForecastVariableBlock,
)
from mausamsetu.dashboard.api.serializers import (
    clean_floats,
    mask_missing,
    to_ist,
)

router = APIRouter(tags=["forecast"])


@router.get("/forecast/{forecast_date}", response_model=ForecastResponse)
def forecast_endpoint(
    forecast_date: date = Path(...),
    horizon: int = Query(config.FORECAST_DAYS, ge=1, le=config.FORECAST_DAYS),
    datacube: xr.Dataset = Depends(get_datacube),
    model=Depends(get_model),
    cache: dict = Depends(get_cache),
) -> ForecastResponse:
    start = resolve_date(datacube, forecast_date)
    try:
        result = cached_forecast(cache, model, datacube, start, horizon)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    blocks = {
        var: ForecastVariableBlock(
            unit=UNITS[var],
            p10=clean_floats(result.quantiles[var]["p10"]),
            p50=clean_floats(result.quantiles[var]["p50"]),
            p90=clean_floats(result.quantiles[var]["p90"]),
        )
        for var in ("rain", "tmax", "tmin")
    }
    return ForecastResponse(
        forecast_start_ist=to_ist(start),
        horizon_days=int(horizon),
        lat=[float(v) for v in result.lat],
        lon=[float(v) for v in result.lon],
        dates=[np.datetime64(d, "D").astype("O") for d in result.dates],
        rain=blocks["rain"],
        tmax=blocks["tmax"],
        tmin=blocks["tmin"],
    )


@router.get("/forecast/assimilation/{assim_date}", response_model=AssimilationResponse)
def assimilation_endpoint(
    assim_date: date = Path(...),
    var: str = Query("rain", pattern="^(rain|tmax|tmin)$"),
    datacube: xr.Dataset = Depends(get_datacube),
    model=Depends(get_model),
    cache: dict = Depends(get_cache),
) -> AssimilationResponse:
    ts = resolve_date(datacube, assim_date)
    try:
        forecast = cached_forecast(cache, model, datacube, ts, horizon=1)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    observed = mask_missing(datacube[var].sel(time=ts).values).astype(np.float32)
    forecast_mean = forecast.quantiles[var]["p50"][0].astype(np.float32)
    # Build a small ensemble around the model mean for the EnKF update:
    rng = np.random.default_rng(0)
    ensemble = forecast_mean[None] + rng.normal(0, 1.0, size=(config.ENKF_ENSEMBLE_SIZE, *forecast_mean.shape)).astype(np.float32)
    obs_error = 5.0 if var == "rain" else 1.5
    obs_input = np.nan_to_num(observed, nan=float(np.nanmean(observed)))
    analysis = enkf_update(ensemble, obs_input, obs_error, inflation=config.ENKF_INFLATION, rng=rng)
    corrected = analysis.mean(axis=0)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(np.nanmean(diff * diff)))

    rmse_before = rmse(forecast_mean, observed)
    rmse_after = rmse(corrected, observed)
    reduction = 0.0 if rmse_before == 0.0 else (rmse_before - rmse_after) / rmse_before * 100.0

    unit = "mm/day" if var == "rain" else "degC"
    return AssimilationResponse(
        date_ist=to_ist(ts),
        variable=var,
        unit=unit,
        lat=[float(v) for v in datacube.lat.values],
        lon=[float(v) for v in datacube.lon.values],
        observed=clean_floats(observed),
        model_mean=clean_floats(forecast_mean),
        corrected=clean_floats(corrected),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        reduction_pct=reduction,
    )
