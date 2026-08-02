"""Validation + baseline routes."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from mausamsetu import config
from mausamsetu.dashboard.api.deps import (
    get_cache,
    get_datacube,
    get_model,
    get_validation_bundle,
    resolve_date,
)
from mausamsetu.dashboard.api.forecast_service import cached_forecast
from mausamsetu.dashboard.api.schemas import (
    MetricBundle,
    ValidationDateMetricsResponse,
    ValidationMetricsResponse,
)
from mausamsetu.dashboard.api.serializers import mask_missing, to_ist
from mausamsetu.dashboard.api.validation_service import ValidationBundle
from mausamsetu.metrics.metrics import bias, csi, far, hss, mae, pod, rmse

router = APIRouter(tags=["validation"])


def _bundle_to_metric(bundle: dict[str, float | None]) -> MetricBundle:
    return MetricBundle.model_validate(bundle)


@router.get("/validation/metrics", response_model=ValidationMetricsResponse)
def full_metrics(
    variable: str = Query("rain", pattern="^rain$"),
    lead_day: int | None = Query(default=None, ge=0, le=config.FORECAST_DAYS - 1),
    bundle: ValidationBundle = Depends(get_validation_bundle),
) -> ValidationMetricsResponse:
    if lead_day is None:
        ours, pers, clim = bundle.ours, bundle.persistence, bundle.climatology
    else:
        block = bundle.per_lead[lead_day]
        ours, pers, clim = block["ours"], block["persistence"], block["climatology"]
    return ValidationMetricsResponse(
        variable=bundle.variable,
        unit=bundle.unit,
        years=bundle.years,
        lead_day=lead_day,
        n_windows=bundle.n_windows,
        ours=_bundle_to_metric(ours),
        persistence=_bundle_to_metric(pers),
        climatology=_bundle_to_metric(clim),
    )


def _clim_series(datacube: xr.Dataset, dates: np.ndarray) -> np.ndarray:
    frames = []
    for day in dates:
        doy = (np.datetime64(day, "D") - np.datetime64(f"{np.datetime64(day, 'Y')}-01-01")).astype(int) + 1
        doy = min(doy, 365)
        frames.append(mask_missing(datacube.rain_clim.sel(dayofyear=doy).values))
    return np.stack(frames, axis=0)


def _metric_dict(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": mae(pred, truth),
        "rmse": rmse(pred, truth),
        "bias": bias(pred, truth),
        "pod@1mm": pod(pred, truth, threshold=1.0),
        "far@1mm": far(pred, truth, threshold=1.0),
        "csi@1mm": csi(pred, truth, threshold=1.0),
        "hss@1mm": hss(pred, truth, threshold=1.0),
        "pod@10mm": pod(pred, truth, threshold=10.0),
        "far@10mm": far(pred, truth, threshold=10.0),
        "csi@10mm": csi(pred, truth, threshold=10.0),
    }


@router.get(
    "/validation/metrics/date/{query_date}",
    response_model=ValidationDateMetricsResponse,
)
def date_metrics(
    query_date: date = Path(...),
    variable: str = Query("rain", pattern="^rain$"),
    datacube: xr.Dataset = Depends(get_datacube),
    model=Depends(get_model),
    cache: dict = Depends(get_cache),
) -> ValidationDateMetricsResponse:
    ts = resolve_date(datacube, query_date)
    times = datacube.time.values
    start_index = int(np.where(times == ts)[0][0])
    end_index = start_index + config.FORECAST_DAYS - 1
    if end_index >= len(times):
        raise HTTPException(status_code=400, detail="Not enough future days to score this date")

    try:
        forecast = cached_forecast(cache, model, datacube, ts, config.FORECAST_DAYS)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    truth = np.stack(
        [mask_missing(datacube.rain.isel(time=start_index + i).values) for i in range(config.FORECAST_DAYS)],
        axis=0,
    )
    ours = forecast.quantiles["rain"]["p50"]
    last_obs = mask_missing(datacube.rain.isel(time=start_index - 1).values)[None, :, :]
    persistence = np.repeat(last_obs, config.FORECAST_DAYS, axis=0)
    clim = _clim_series(datacube, forecast.dates)

    return ValidationDateMetricsResponse(
        variable="rain",
        unit="mm/day",
        date_ist=to_ist(ts),
        n_forecast_days=config.FORECAST_DAYS,
        ours=_bundle_to_metric(_metric_dict(ours, truth)),
        persistence=_bundle_to_metric(_metric_dict(persistence, truth)),
        climatology=_bundle_to_metric(_metric_dict(clim, truth)),
    )
