"""Thin adapters over ``mausamsetu.impacts`` for API routes."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from mausamsetu import config
from mausamsetu.impacts.heat import compare_heat_impact, heat_stress_days, extreme_heat_days
from mausamsetu.impacts.hydrology import basin_daily_inflow, compare_scenario_impact
from mausamsetu.impacts.rupee_risk import total_rupee_risk
from mausamsetu.storyline.scenario import Scenario, apply_scenario


def _select_window(datacube: xr.Dataset, start_date: date, days: int) -> xr.Dataset:
    time = datacube.time.values
    start = np.datetime64(start_date.isoformat())
    end = start + np.timedelta64(days - 1, "D")
    if start < time[0] or end > time[-1]:
        first = str(time[0])[:10]
        last = str(time[-1])[:10]
        raise ValueError(
            f"[{start_date.isoformat()}, +{days}d) exceeds datacube coverage ({first}..{last})"
        )
    return datacube.sel(time=slice(start_date.isoformat(), str(end)[:10]))


def hydrology_inflow(datacube: xr.Dataset, start_date: date, days: int):
    window = _select_window(datacube, start_date, days)
    inflow = basin_daily_inflow(window.rain)
    return {
        "dates": [np.datetime64(t, "D") for t in inflow.time.values],
        "inflow_m3_per_day": inflow.values.astype(float).tolist(),
        "total_m3": float(inflow.sum().item()),
    }


def heat_days(datacube: xr.Dataset, start_date: date, days: int):
    window = _select_window(datacube, start_date, days)
    hot = int(heat_stress_days(window.tmax).sum().item())
    extreme = int(extreme_heat_days(window.tmax).sum().item())
    return {
        "dates": [np.datetime64(t, "D") for t in window.time.values],
        "heat_stress_days": hot,
        "extreme_heat_days": extreme,
        "heat_stress_threshold_c": float(config.HEAT_STRESS_TEMP),
        "extreme_heat_threshold_c": float(config.EXTREME_HEAT_TEMP),
    }


def rupee_risk(
    datacube: xr.Dataset,
    start_date: date,
    days: int,
    delta_temp_c: float,
    delta_rain_pct: float,
):
    window = _select_window(datacube, start_date, days)[["rain", "tmax", "tmin"]]
    scenario = Scenario(
        delta_temp=float(delta_temp_c),
        delta_rain_pct=float(delta_rain_pct),
        label="impacts-rupee",
    )
    scenario_ds = apply_scenario(window, scenario)
    aggregate = total_rupee_risk(window, scenario_ds)
    return {
        "dates": [np.datetime64(t, "D") for t in window.time.values],
        "total_crore": float(aggregate["total_crore"]),
        "rainfall_component_crore": float(aggregate["components"]["rainfall"].sum().item()),
        "heat_component_crore": float(aggregate["components"]["heat"].sum().item()),
    }


def scenario_impacts(
    datacube: xr.Dataset,
    baseline: xr.Dataset,
    scenario: xr.Dataset,
):
    hydro = compare_scenario_impact(baseline.rain, scenario.rain)
    heat = compare_heat_impact(baseline.tmax, scenario.tmax)
    rupee = total_rupee_risk(baseline, scenario)
    return {
        "hydrology": {
            "baseline_total_m3": float(hydro["baseline_total_m3"]),
            "scenario_total_m3": float(hydro["scenario_total_m3"]),
            "delta_m3": float(hydro["delta_m3"]),
            "delta_pct": float(hydro["delta_pct"]),
        },
        "heat": heat,
        "rupee": {
            "total_crore": float(rupee["total_crore"]),
            "rainfall_component_crore": float(rupee["components"]["rainfall"].sum().item()),
            "heat_component_crore": float(rupee["components"]["heat"].sum().item()),
        },
    }
