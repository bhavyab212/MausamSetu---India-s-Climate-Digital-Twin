"""Baseline models: persistence and climatology.

Both operate on normalized [0,1] data to match the model's output space.
"""

from __future__ import annotations

import numpy as np


def persistence_prediction(last_observed: np.ndarray) -> np.ndarray:
    """Predict = last observed value (repeat forward).

    last_observed: (lat, lon, channels) — the final frame of the input sequence.
    Returns same shape — the prediction for the next time step.
    """
    return last_observed.copy()


def climatology_prediction(
    historical_data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Predict = mean of all historical data at each cell.

    historical_data: (time, lat, lon, channels)
    mask: (lat, lon)
    Returns: (lat, lon, channels) — per-cell historical mean.
    """
    # Mean over time axis, ignoring structure
    clim = np.mean(historical_data, axis=0)  # (lat, lon, channels)
    # Apply mask
    mask_expanded = mask[:, :, np.newaxis]
    return clim * mask_expanded


def compute_baseline_metrics(
    truth: np.ndarray,
    last_observed: np.ndarray,
    historical_data: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute persistence and climatology metrics for comparison.

    truth: (lat, lon, channels) — ground truth for the prediction step
    last_observed: (lat, lon, channels) — input to persistence
    historical_data: (time, lat, lon, channels) — input to climatology
    mask: (lat, lon) — land mask
    """
    from .metrics import compute_all_metrics

    pers = persistence_prediction(last_observed)
    clim = climatology_prediction(historical_data, mask)

    results = {}
    # Compute per channel, average across channels for summary
    n_channels = truth.shape[-1]

    for name, pred in [("persistence", pers), ("climatology", clim)]:
        channel_metrics = []
        for c in range(n_channels):
            m = compute_all_metrics(pred[:, :, c], truth[:, :, c], mask)
            channel_metrics.append(m)
        # Average across channels
        avg = {}
        for key in channel_metrics[0]:
            vals = [cm[key] for cm in channel_metrics if not np.isnan(cm[key])]
            avg[key] = float(np.mean(vals)) if vals else float("nan")
        results[name] = avg

    return results
