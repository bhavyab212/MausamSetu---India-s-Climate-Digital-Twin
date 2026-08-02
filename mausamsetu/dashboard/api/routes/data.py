"""Datacube discovery and extraction routes."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from mausamsetu.dashboard.api.deps import (
    get_datacube,
    resolve_date,
    resolve_variable,
)
from mausamsetu.dashboard.api.schemas import (
    CompletenessStats,
    DatesResponse,
    GridResponse,
    TimeseriesResponse,
    VariableInfo,
    VariablesResponse,
)
from mausamsetu.dashboard.api.serializers import (
    basin_mask,
    clean_floats,
    completeness,
    mask_missing,
    to_ist,
)

router = APIRouter(tags=["data"])

UNIT_DEFAULT = {
    "rain": "mm/day",
    "tmax": "degC",
    "tmin": "degC",
    "insat_lst": "degC",
    "insat_rain": "mm/day",
    "rain_anom": "mm/day",
    "rain_clim": "mm/day",
    "tmax_anom": "degC",
    "tmax_clim": "degC",
    "tmin_anom": "degC",
    "tmin_clim": "degC",
}


@router.get("/data/dates", response_model=DatesResponse)
def list_dates(datacube: xr.Dataset = Depends(get_datacube)) -> DatesResponse:
    dates = [date.fromisoformat(str(t)[:10]) for t in datacube.time.values]
    return DatesResponse(dates=dates, count=len(dates))


@router.get("/data/variables", response_model=VariablesResponse)
def list_variables(datacube: xr.Dataset = Depends(get_datacube)) -> VariablesResponse:
    variables = []
    for name, da in datacube.data_vars.items():
        variables.append(
            VariableInfo(
                name=str(name),
                unit=str(da.attrs.get("units", UNIT_DEFAULT.get(name, ""))),
                description=str(da.attrs.get("long_name", da.attrs.get("description", ""))),
                dtype=str(da.dtype),
                dims=list(da.dims),
            )
        )
    return VariablesResponse(variables=variables)


@router.get("/data/grid/{grid_date}", response_model=GridResponse)
def grid(
    grid_date: date = Path(...),
    var: str = Query("rain"),
    datacube: xr.Dataset = Depends(get_datacube),
) -> GridResponse:
    variable = resolve_variable(datacube, var)
    ts = resolve_date(datacube, grid_date)
    da = datacube[variable]
    if "time" not in da.dims:
        raise HTTPException(status_code=400, detail=f"variable '{variable}' has no time dimension")
    field = da.sel(time=ts).values
    mask = basin_mask(datacube)
    stats = completeness(field, mask)
    unit = UNIT_DEFAULT.get(variable, str(da.attrs.get("units", "")))
    return GridResponse(
        date_ist=to_ist(ts),
        variable=variable,
        unit=unit,
        lat=[float(v) for v in datacube.lat.values],
        lon=[float(v) for v in datacube.lon.values],
        values=clean_floats(field),
        completeness=CompletenessStats(**stats),
    )


def _nearest_index(coord: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(coord - target)))


@router.get("/data/timeseries", response_model=TimeseriesResponse)
def timeseries(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    days: int = Query(30, ge=1, le=1461),
    end_date: date | None = Query(default=None),
    datacube: xr.Dataset = Depends(get_datacube),
) -> TimeseriesResponse:
    times = datacube.time.values
    end_ts = resolve_date(datacube, end_date)
    end_index = int(np.where(times == end_ts)[0][0])
    start_index = max(0, end_index - days + 1)
    slice_ = datacube.isel(time=slice(start_index, end_index + 1))
    lat_idx = _nearest_index(datacube.lat.values, lat)
    lon_idx = _nearest_index(datacube.lon.values, lon)
    picks = slice_.isel(lat=lat_idx, lon=lon_idx)

    def series(name: str) -> list:
        return clean_floats(mask_missing(picks[name].values))

    dates = [date.fromisoformat(str(t)[:10]) for t in slice_.time.values]
    return TimeseriesResponse(
        lat_deg_north=float(datacube.lat.values[lat_idx]),
        lon_deg_east=float(datacube.lon.values[lon_idx]),
        dates=dates,
        rain_mm_per_day=series("rain"),
        tmax_c=series("tmax"),
        tmin_c=series("tmin"),
        insat_lst_c=series("insat_lst"),
        insat_rain_mm_per_day=series("insat_rain"),
    )
