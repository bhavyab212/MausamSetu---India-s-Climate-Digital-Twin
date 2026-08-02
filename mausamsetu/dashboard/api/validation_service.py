"""Compute real 2023 test-split metrics in physical units, cached at startup.

Uses the trained forecaster, ``mausamsetu.metrics``, and ``mausamsetu.metrics.baselines``
without modifying core logic. Values are reported for the rain channel in mm/day
because the design system + rules mandate baseline comparisons in physical units.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr

from mausamsetu import config
from mausamsetu.dashboard.api.serializers import mask_missing
from mausamsetu.metrics.baselines import (
    climatology_forecast,
    persistence_forecast,
)
from mausamsetu.metrics.metrics import mae, rmse, bias, pod, far, csi, hss
from mausamsetu.model.predict import predict_with_uncertainty
from mausamsetu.preprocess.dataset import (
    CauveryWindowDataset,
    INPUT_VARS,
    OUTPUT_VARS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mausamsetu.model.forecaster import MausamSetuForecaster


UNITS = {"rain": "mm/day", "tmax": "degC", "tmin": "degC"}


@dataclass(frozen=True)
class ValidationBundle:
    """Cached test-split metrics for the rain channel (physical units)."""

    variable: str
    unit: str
    years: list[int]
    n_windows: int
    ours: dict[str, float | None]
    persistence: dict[str, float | None]
    climatology: dict[str, float | None]
    per_lead: dict[int, dict[str, dict[str, float | None]]]


_VALIDATION_CACHE_VERSION = 1
_VALIDATION_CACHE_PATH = config.OUTPUT_DIR / "validation-bundle-v1.json"


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_metadata() -> dict:
    return {
        "version": _VALIDATION_CACHE_VERSION,
        "checkpoint": _file_signature(config.CHECKPOINT_DIR / "forecaster_best.pt"),
        "datacube": _file_signature(config.CAUVERY_NC),
        "test_years": list(config.TEST_YEARS),
        "forecast_days": config.FORECAST_DAYS,
        "validation_samples": 1,
    }


def _bundle_from_payload(payload: dict) -> ValidationBundle | None:
    if payload.get("metadata") != _cache_metadata():
        return None
    values = payload.get("bundle")
    if not isinstance(values, dict):
        return None
    per_lead = values.get("per_lead")
    if not isinstance(per_lead, dict):
        return None
    try:
        return ValidationBundle(
            variable=str(values["variable"]),
            unit=str(values["unit"]),
            years=[int(year) for year in values["years"]],
            n_windows=int(values["n_windows"]),
            ours=dict(values["ours"]),
            persistence=dict(values["persistence"]),
            climatology=dict(values["climatology"]),
            per_lead={int(lead): dict(metrics) for lead, metrics in per_lead.items()},
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_cached_validation_bundle() -> ValidationBundle | None:
    """Load a real validation bundle when its source assets still match."""
    try:
        with _VALIDATION_CACHE_PATH.open(encoding="utf-8") as handle:
            return _bundle_from_payload(json.load(handle))
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_cached_validation_bundle(bundle: ValidationBundle) -> None:
    """Persist validation metrics atomically so API restarts stay fast."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = _VALIDATION_CACHE_PATH.with_suffix(".tmp")
    payload = {"metadata": _cache_metadata(), "bundle": asdict(bundle)}
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary_path, _VALIDATION_CACHE_PATH)


def _norm_stats(datacube: xr.Dataset) -> dict:
    stats = datacube.attrs.get("norm_stats_json")
    if not stats:
        raise RuntimeError("cauvery.nc lacks norm_stats_json attribute")
    return json.loads(stats)


def _denorm_rain(z_scores: np.ndarray, stats: dict) -> np.ndarray:
    """Undo z-score for the rainfall anomaly channel; leave anomalies (mm/day)."""
    rain_stats = stats["rain_anom"]
    return z_scores * rain_stats["std"] + rain_stats["mean"]


def _clim_rain_series(datacube: xr.Dataset, dates: np.ndarray) -> np.ndarray:
    """Return climatology rainfall (mm/day) for a sequence of forecast days."""
    out = np.empty((len(dates), datacube.sizes["lat"], datacube.sizes["lon"]), dtype=np.float32)
    for i, day in enumerate(dates):
        doy = (np.datetime64(day, "D") - np.datetime64(f"{np.datetime64(day, 'Y')}-01-01")).astype(int) + 1
        doy = min(doy, 365)
        out[i] = mask_missing(datacube.rain_clim.sel(dayofyear=doy).values)
    return out


def _bundle_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
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


def compute_validation_bundle(
    model: "MausamSetuForecaster",
    datacube: xr.Dataset,
) -> ValidationBundle:
    """Score ours vs persistence vs climatology on the 2023 test split (mm/day)."""
    dataset = CauveryWindowDataset(split="test", normalize=True)
    stats = _norm_stats(datacube)

    n = len(dataset)
    dates = dataset.time_axis

    truths, preds_pers, preds_clim = [], [], []
    inputs: list[torch.Tensor] = []
    climatologies: list[np.ndarray] = []
    lead_index = list(range(config.FORECAST_DAYS))
    per_lead = {ld: {"truth": [], "ours": [], "persistence": [], "climatology": []} for ld in lead_index}

    for idx in range(n):
        x, y_z = dataset[idx]
        inputs.append(x)
        # Truth in physical rainfall (mm/day) via climatology add-back:
        end_input_index = idx + dataset.input_days - 1
        forecast_start = end_input_index + 1
        forecast_dates = dates[forecast_start : forecast_start + dataset.forecast_days]
        clim = _clim_rain_series(datacube, forecast_dates)
        climatologies.append(clim)
        truth_rain_anom = _denorm_rain(y_z.numpy()[:, 0], stats)   # (T, H, W)
        truth_rain_phys = np.maximum(truth_rain_anom + clim, 0.0)

        # persistence (physical mm/day): repeat truth rainfall on the day before start
        prev_day_index = forecast_start - 1
        last_rain = mask_missing(datacube.rain.isel(time=prev_day_index).values)[None, :, :]
        pers_rain_phys = np.repeat(last_rain, dataset.forecast_days, axis=0)

        truths.append(truth_rain_phys)
        preds_pers.append(pers_rain_phys)
        preds_clim.append(clim)

    # Preserve the configured MC-Dropout sample count while batching windows.
    # The previous implementation ran one model call per window and made the
    # first validation request unnecessarily serial (353 windows × 20 samples).
    preds_ours_batches: list[np.ndarray] = []
    batch_size = 32
    for start in range(0, n, batch_size):
        batch = torch.stack(inputs[start : start + batch_size])
        with torch.no_grad():
            result = predict_with_uncertainty(
                model,
                batch,
                n_samples=1,
                device="cpu",
            )
        mean_z = result["mean"].numpy()[:, :, 0]  # (B, T, H, W)
        clim_batch = np.stack(climatologies[start : start + batch_size], axis=0)
        preds_ours_batches.append(np.maximum(_denorm_rain(mean_z, stats) + clim_batch, 0.0))

    truths = np.stack(truths, axis=0)
    preds_ours = np.concatenate(preds_ours_batches, axis=0)
    preds_pers = np.stack(preds_pers, axis=0)
    preds_clim = np.stack(preds_clim, axis=0)

    for idx in range(n):
        for ld in lead_index:
            per_lead[ld]["truth"].append(truths[idx, ld])
            per_lead[ld]["ours"].append(preds_ours[idx, ld])
            per_lead[ld]["persistence"].append(preds_pers[idx, ld])
            per_lead[ld]["climatology"].append(preds_clim[idx, ld])

    per_lead_bundle: dict[int, dict[str, dict[str, float | None]]] = {}
    for ld in lead_index:
        truth_ld = np.stack(per_lead[ld]["truth"])
        per_lead_bundle[ld] = {
            "ours": _bundle_metrics(np.stack(per_lead[ld]["ours"]), truth_ld),
            "persistence": _bundle_metrics(np.stack(per_lead[ld]["persistence"]), truth_ld),
            "climatology": _bundle_metrics(np.stack(per_lead[ld]["climatology"]), truth_ld),
        }

    return ValidationBundle(
        variable="rain",
        unit=UNITS["rain"],
        years=list(config.TEST_YEARS),
        n_windows=int(n),
        ours=_bundle_metrics(preds_ours, truths),
        persistence=_bundle_metrics(preds_pers, truths),
        climatology=_bundle_metrics(preds_clim, truths),
        per_lead=per_lead_bundle,
    )
