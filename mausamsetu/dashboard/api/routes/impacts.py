"""Climate impact routes."""

from __future__ import annotations

from datetime import date as _date

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Query

from mausamsetu.dashboard.api.deps import get_datacube
from mausamsetu.dashboard.api.impact_service import (
    heat_days,
    hydrology_inflow,
    rupee_risk,
)
from mausamsetu.dashboard.api.schemas import (
    HeatResponse,
    HydrologyResponse,
    RupeeResponse,
)
from mausamsetu.dashboard.api.serializers import to_ist

router = APIRouter(tags=["impacts"])


@router.get("/impacts/hydrology", response_model=HydrologyResponse)
def hydrology_endpoint(
    start_date: _date = Query(...),
    days: int = Query(30, ge=1, le=366),
    datacube: xr.Dataset = Depends(get_datacube),
) -> HydrologyResponse:
    try:
        result = hydrology_inflow(datacube, start_date, days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HydrologyResponse(
        start_date_ist=to_ist(np.datetime64(start_date.isoformat())),
        dates=[_date.fromisoformat(str(d)[:10]) for d in result["dates"]],
        inflow_m3_per_day=result["inflow_m3_per_day"],
        total_m3=result["total_m3"],
    )


@router.get("/impacts/heat", response_model=HeatResponse)
def heat_endpoint(
    start_date: _date = Query(...),
    days: int = Query(30, ge=1, le=366),
    datacube: xr.Dataset = Depends(get_datacube),
) -> HeatResponse:
    try:
        result = heat_days(datacube, start_date, days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HeatResponse(
        start_date_ist=to_ist(np.datetime64(start_date.isoformat())),
        dates=[_date.fromisoformat(str(d)[:10]) for d in result["dates"]],
        heat_stress_days=int(result["heat_stress_days"]),
        extreme_heat_days=int(result["extreme_heat_days"]),
        heat_stress_threshold_c=float(result["heat_stress_threshold_c"]),
        extreme_heat_threshold_c=float(result["extreme_heat_threshold_c"]),
    )


@router.get("/impacts/rupee", response_model=RupeeResponse)
def rupee_endpoint(
    start_date: _date = Query(...),
    days: int = Query(30, ge=1, le=366),
    delta_temp_c: float = Query(0.0, ge=-5.0, le=8.0),
    delta_rain_pct: float = Query(0.0, ge=-80.0, le=80.0),
    datacube: xr.Dataset = Depends(get_datacube),
) -> RupeeResponse:
    try:
        result = rupee_risk(datacube, start_date, days, delta_temp_c, delta_rain_pct)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RupeeResponse(
        start_date_ist=to_ist(np.datetime64(start_date.isoformat())),
        dates=[_date.fromisoformat(str(d)[:10]) for d in result["dates"]],
        total_crore=float(result["total_crore"]),
        rainfall_component_crore=float(result["rainfall_component_crore"]),
        heat_component_crore=float(result["heat_component_crore"]),
    )
