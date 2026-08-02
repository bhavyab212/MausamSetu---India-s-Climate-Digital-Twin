"""Derive date-scoped alerts from real datacube values + resolved thresholds."""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from mausamsetu.dashboard.api.schemas import AlertItem, ThresholdConfig
from mausamsetu.dashboard.api.serializers import basin_mask, mask_missing


def _basin_mean(field: np.ndarray, mask: np.ndarray) -> float | None:
    values = mask_missing(field)[mask]
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return float(finite.mean())


def _basin_max(field: np.ndarray, mask: np.ndarray) -> float | None:
    values = mask_missing(field)[mask]
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return float(finite.max())


def derive_alerts(
    datacube: xr.Dataset,
    thresholds: ThresholdConfig,
    day: np.datetime64,
) -> list[AlertItem]:
    mask = basin_mask(datacube)
    evaluated_date = date.fromisoformat(str(day)[:10])
    alerts: list[AlertItem] = []

    rain = datacube.rain.sel(time=day).values
    rain_mean = _basin_mean(rain, mask)
    if rain_mean is not None:
        if rain_mean >= thresholds.rain_critical_mm_per_day:
            alerts.append(
                AlertItem(
                    id=f"rain-critical-{evaluated_date.isoformat()}",
                    type="rainfall",
                    severity="critical",
                    title="Basin rainfall at critical level",
                    description=(
                        f"Basin-mean rainfall {rain_mean:.1f} mm/day is at or above the critical threshold."
                    ),
                    observed_value=rain_mean,
                    threshold=thresholds.rain_critical_mm_per_day,
                    unit="mm/day",
                    evaluated_date=evaluated_date,
                    data_source="cauvery.nc:rain (IMD basin mean)",
                    threshold_source=thresholds.sources["rain_critical_mm_per_day"],
                )
            )
        elif rain_mean >= thresholds.rain_warning_mm_per_day:
            alerts.append(
                AlertItem(
                    id=f"rain-warning-{evaluated_date.isoformat()}",
                    type="rainfall",
                    severity="warning",
                    title="Basin rainfall above warning threshold",
                    description=(
                        f"Basin-mean rainfall {rain_mean:.1f} mm/day is at or above the warning threshold."
                    ),
                    observed_value=rain_mean,
                    threshold=thresholds.rain_warning_mm_per_day,
                    unit="mm/day",
                    evaluated_date=evaluated_date,
                    data_source="cauvery.nc:rain (IMD basin mean)",
                    threshold_source=thresholds.sources["rain_warning_mm_per_day"],
                )
            )

    tmax = datacube.tmax.sel(time=day).values
    tmax_max = _basin_max(tmax, mask)
    if tmax_max is not None:
        if tmax_max >= thresholds.heat_critical_c:
            alerts.append(
                AlertItem(
                    id=f"heat-critical-{evaluated_date.isoformat()}",
                    type="extreme_heat",
                    severity="critical",
                    title="Extreme heat detected in basin",
                    description=(
                        f"Peak basin Tmax {tmax_max:.1f} °C exceeds the extreme-heat threshold."
                    ),
                    observed_value=tmax_max,
                    threshold=thresholds.heat_critical_c,
                    unit="degC",
                    evaluated_date=evaluated_date,
                    data_source="cauvery.nc:tmax (IMD basin max)",
                    threshold_source=thresholds.sources["heat_critical_c"],
                )
            )
        elif tmax_max >= thresholds.heat_warning_c:
            alerts.append(
                AlertItem(
                    id=f"heat-warning-{evaluated_date.isoformat()}",
                    type="heat",
                    severity="warning",
                    title="Heat stress detected in basin",
                    description=(
                        f"Peak basin Tmax {tmax_max:.1f} °C exceeds the heat-stress warning."
                    ),
                    observed_value=tmax_max,
                    threshold=thresholds.heat_warning_c,
                    unit="degC",
                    evaluated_date=evaluated_date,
                    data_source="cauvery.nc:tmax (IMD basin max)",
                    threshold_source=thresholds.sources["heat_warning_c"],
                )
            )

    return alerts
