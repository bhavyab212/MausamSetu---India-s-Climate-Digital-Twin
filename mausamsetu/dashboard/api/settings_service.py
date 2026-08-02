"""Resolve alert thresholds from environment overrides + train-only defaults."""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from mausamsetu import config
from mausamsetu.dashboard.api.schemas import ThresholdConfig
from mausamsetu.dashboard.api.serializers import mask_missing

TRAIN_YEARS = tuple(config.TRAIN_YEARS)

_ENV_KEYS = {
    "rain_warning_mm_per_day": "MAUSAMSETU_ALERT_RAIN_WARNING_MM",
    "rain_critical_mm_per_day": "MAUSAMSETU_ALERT_RAIN_CRITICAL_MM",
    "heat_warning_c": "MAUSAMSETU_ALERT_HEAT_WARNING_C",
    "heat_critical_c": "MAUSAMSETU_ALERT_HEAT_CRITICAL_C",
}


def _train_basin_mean_rain(datacube: xr.Dataset) -> np.ndarray:
    years = datacube.time.dt.year.values
    train_mask = np.isin(years, TRAIN_YEARS)
    if not train_mask.any():
        raise RuntimeError("Datacube does not contain any TRAIN_YEARS days")
    rain = mask_missing(datacube.rain.sel(time=train_mask).values)
    daily_mean = np.nanmean(rain.reshape(rain.shape[0], -1), axis=1)
    return daily_mean[np.isfinite(daily_mean)]


def resolve_thresholds(datacube: xr.Dataset) -> ThresholdConfig:
    """Compute defaults from the training years, then apply env overrides."""
    basin_mean = _train_basin_mean_rain(datacube)
    default_rain_warn = float(np.percentile(basin_mean, 95))
    default_rain_crit = float(np.percentile(basin_mean, 99))
    default_heat_warn = float(config.HEAT_STRESS_TEMP)
    default_heat_crit = float(config.EXTREME_HEAT_TEMP)

    defaults = {
        "rain_warning_mm_per_day": default_rain_warn,
        "rain_critical_mm_per_day": default_rain_crit,
        "heat_warning_c": default_heat_warn,
        "heat_critical_c": default_heat_crit,
    }
    sources: dict[str, str] = {}
    values: dict[str, float] = {}
    for key, default_value in defaults.items():
        env_key = _ENV_KEYS[key]
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                values[key] = float(env_value)
                sources[key] = f"env:{env_key}"
                continue
            except ValueError:
                pass
        values[key] = default_value
        sources[key] = (
            "train_p95_2020_2021" if key == "rain_warning_mm_per_day"
            else "train_p99_2020_2021" if key == "rain_critical_mm_per_day"
            else "config.HEAT_STRESS_TEMP" if key == "heat_warning_c"
            else "config.EXTREME_HEAT_TEMP"
        )
    return ThresholdConfig(sources=sources, **values)
