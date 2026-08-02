"""Real ConvLSTM forecast adapter: normalized anomalies → physical units.

The Python core stores normalization statistics in the ``cauvery.nc`` global
attribute ``norm_stats_json`` and holds day-of-year climatology in the
``rain_clim``/``tmax_clim``/``tmin_clim`` variables. Model outputs live in
normalized anomaly space, so this service:

1. builds the six-day input window from the datacube;
2. runs 20 MC-Dropout samples on the trained model;
3. undoes z-scoring using train-only statistics;
4. adds the day-of-year climatology to move from anomalies to physical units;
5. applies physics clamps (``rain ≥ 0``, ``tmin ≤ tmax``);
6. caches per ``(start_date, horizon)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr

from mausamsetu import config
from mausamsetu.dashboard.api.serializers import mask_missing
from mausamsetu.model.predict import predict_with_uncertainty
from mausamsetu.preprocess.dataset import INPUT_VARS, OUTPUT_VARS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mausamsetu.model.forecaster import MausamSetuForecaster


PHYSICS_CLAMPS = {
    "rain": (0.0, 500.0),
    "tmax": (-5.0, 55.0),
    "tmin": (-15.0, 45.0),
}
UNITS = {"rain": "mm/day", "tmax": "degC", "tmin": "degC"}


@dataclass(frozen=True)
class ForecastResult:
    """Physical-unit forecast bundle."""

    start_date: np.datetime64
    dates: np.ndarray                     # (T_out,) datetime64[D]
    lat: np.ndarray
    lon: np.ndarray
    quantiles: dict[str, dict[str, np.ndarray]]  # var -> {'p10','p50','p90'} -> (T,H,W)


def _norm_stats(datacube: xr.Dataset) -> dict:
    stats = datacube.attrs.get("norm_stats_json")
    if not stats:
        raise RuntimeError("cauvery.nc lacks norm_stats_json attribute")
    return json.loads(stats)


def _build_input_window(datacube: xr.Dataset, start_date: np.datetime64) -> torch.Tensor:
    """Stack the six days ending the day before ``start_date`` into (1, T, C, H, W)."""
    times = datacube.time.values
    end_index = int(np.where(times == start_date - np.timedelta64(1, "D"))[0][0])
    window_start = end_index - config.INPUT_DAYS + 1
    if window_start < 0:
        raise ValueError(
            f"Not enough history before {start_date} for a {config.INPUT_DAYS}-day input"
        )
    slice_ = datacube.isel(time=slice(window_start, end_index + 1))
    norm_stats = _norm_stats(datacube)
    channels = []
    for var in INPUT_VARS:
        raw = mask_missing(slice_[var].values)
        stats = norm_stats.get(var, {"mean": 0.0, "std": 1.0})
        z = (raw - stats["mean"]) / (stats["std"] + 1e-8)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        channels.append(z.astype(np.float32))
    stacked = np.stack(channels, axis=1)  # (T, C, H, W)
    return torch.from_numpy(stacked).unsqueeze(0)  # (1, T, C, H, W)


def _forecast_dates(start_date: np.datetime64, horizon: int) -> np.ndarray:
    start_day = np.datetime64(start_date, "D")
    return np.array([start_day + np.timedelta64(i, "D") for i in range(horizon)])


def _clim_frame(datacube: xr.Dataset, day: np.datetime64, var: str) -> np.ndarray:
    doy = (np.datetime64(day, "D") - np.datetime64(f"{np.datetime64(day, 'Y')}-01-01")).astype(int) + 1
    doy = min(doy, 365)
    clim_name = {"rain": "rain_clim", "tmax": "tmax_clim", "tmin": "tmin_clim"}[var]
    return mask_missing(datacube[clim_name].sel(dayofyear=doy).values)


def _denormalize(sample: np.ndarray, datacube: xr.Dataset, dates: np.ndarray) -> dict[str, np.ndarray]:
    """Convert (T, 3, H, W) normalized anomalies to per-variable (T, H, W) physical arrays."""
    stats = _norm_stats(datacube)
    means = np.array([stats[v]["mean"] for v in OUTPUT_VARS], dtype=np.float32)
    stds = np.array([stats[v]["std"] for v in OUTPUT_VARS], dtype=np.float32)
    anom = sample * stds[None, :, None, None] + means[None, :, None, None]
    per_var: dict[str, np.ndarray] = {}
    for c, output_var in enumerate(OUTPUT_VARS):
        var = output_var.replace("_anom", "")
        physical = np.empty_like(anom[:, c])
        for t, day in enumerate(dates):
            clim = _clim_frame(datacube, day, var)
            physical[t] = anom[t, c] + clim
        low, high = PHYSICS_CLAMPS[var]
        physical = np.clip(physical, low, high)
        per_var[var] = physical
    if "rain" in per_var:
        per_var["rain"] = np.maximum(per_var["rain"], 0.0)
    if "tmax" in per_var and "tmin" in per_var:
        per_var["tmin"] = np.minimum(per_var["tmin"], per_var["tmax"] - 0.5)
    return per_var


def _quantiles(samples: list[dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    keys = samples[0].keys()
    stacked = {var: np.stack([s[var] for s in samples], axis=0) for var in keys}
    out: dict[str, dict[str, np.ndarray]] = {}
    for var, values in stacked.items():
        out[var] = {
            "p10": np.quantile(values, 0.10, axis=0),
            "p50": np.quantile(values, 0.50, axis=0),
            "p90": np.quantile(values, 0.90, axis=0),
        }
    return out


def run_forecast(
    model: "MausamSetuForecaster",
    datacube: xr.Dataset,
    start_date: np.datetime64,
    horizon: int = config.FORECAST_DAYS,
) -> ForecastResult:
    """Run the trained forecaster and return per-variable p10/p50/p90 in physical units."""
    if horizon < 1 or horizon > config.FORECAST_DAYS:
        raise ValueError(f"horizon must be in [1, {config.FORECAST_DAYS}]")
    window = _build_input_window(datacube, start_date)
    result = predict_with_uncertainty(model, window, n_samples=config.MC_SAMPLES, device="cpu")
    samples = result["samples"].numpy()   # (N, 1, T, 3, H, W)
    samples = samples[:, 0, :horizon]     # (N, T, 3, H, W)
    dates = _forecast_dates(start_date, horizon)
    per_sample = [_denormalize(sample, datacube, dates) for sample in samples]
    quantiles = _quantiles(per_sample)
    return ForecastResult(
        start_date=start_date,
        dates=dates,
        lat=datacube.lat.values.astype(float),
        lon=datacube.lon.values.astype(float),
        quantiles=quantiles,
    )


def cached_forecast(
    cache: dict,
    model: "MausamSetuForecaster",
    datacube: xr.Dataset,
    start_date: np.datetime64,
    horizon: int,
) -> ForecastResult:
    key = ("forecast", str(start_date)[:10], int(horizon))
    if key not in cache:
        cache[key] = run_forecast(model, datacube, start_date, horizon)
    return cache[key]
