"""Service health and current-state routes."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, Query

from mausamsetu.dashboard.api.deps import get_datacube, resolve_date
from mausamsetu.dashboard.api.schemas import (
    CompletenessStats,
    CurrentStateKPI,
    CurrentStateResponse,
    HealthResponse,
)
from mausamsetu.dashboard.api.serializers import (
    basin_mask,
    completeness,
    mask_missing,
    now_ist,
    to_ist,
)

router = APIRouter(tags=["state"])


@router.get("/health", response_model=HealthResponse)
def health(datacube: xr.Dataset = Depends(get_datacube)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        datacube_days=int(datacube.sizes["time"]),
        ist_time=now_ist(),
    )


def _basin_mean(field: np.ndarray, mask: np.ndarray) -> float | None:
    values = mask_missing(field)[mask]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


@router.get("/state/current", response_model=CurrentStateResponse)
def current_state(
    day: date | None = Query(default=None, alias="date"),
    datacube: xr.Dataset = Depends(get_datacube),
) -> CurrentStateResponse:
    ts = resolve_date(datacube, day)
    mask = basin_mask(datacube)
    rain = datacube.rain.sel(time=ts).values
    tmax = datacube.tmax.sel(time=ts).values
    tmin = datacube.tmin.sel(time=ts).values

    rain_mean = _basin_mean(rain, mask)
    tmax_mean = _basin_mean(tmax, mask)
    tmin_mean = _basin_mean(tmin, mask)
    stats = completeness(rain, mask)

    kpis = [
        CurrentStateKPI(label="Basin rainfall", value=rain_mean, unit="mm/day"),
        CurrentStateKPI(label="Basin Tmax", value=tmax_mean, unit="degC"),
        CurrentStateKPI(label="Basin Tmin", value=tmin_mean, unit="degC"),
        CurrentStateKPI(label="Data completeness", value=stats["completeness_pct"], unit="%"),
    ]

    return CurrentStateResponse(
        date_ist=to_ist(ts),
        basin_rainfall_mm_per_day=rain_mean,
        basin_tmax_c=tmax_mean,
        basin_tmin_c=tmin_mean,
        completeness=CompletenessStats(**stats),
        kpis=kpis,
    )
