"""Date-scoped derived alerts."""

from __future__ import annotations

from datetime import date

import xarray as xr
from fastapi import APIRouter, Depends, Query

from mausamsetu.dashboard.api.alert_service import derive_alerts
from mausamsetu.dashboard.api.deps import (
    get_datacube,
    get_thresholds,
    resolve_date,
)
from mausamsetu.dashboard.api.schemas import AlertsResponse, ThresholdConfig
from mausamsetu.dashboard.api.serializers import to_ist

router = APIRouter(tags=["alerts"])


@router.get("/alerts", response_model=AlertsResponse)
def list_alerts(
    day: date | None = Query(default=None, alias="date"),
    datacube: xr.Dataset = Depends(get_datacube),
    thresholds: ThresholdConfig = Depends(get_thresholds),
) -> AlertsResponse:
    ts = resolve_date(datacube, day)
    alerts = derive_alerts(datacube, thresholds, ts)
    return AlertsResponse(
        evaluated_date_ist=to_ist(ts),
        alerts=alerts,
        thresholds=thresholds,
    )
