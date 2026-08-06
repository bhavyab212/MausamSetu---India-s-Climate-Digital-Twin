"""
train.eval.baselines — per-zone-per-season persistence + climatology.

The plan's rule 8: "Every claim carries a baseline (per-zone persistence
AND climatology) and a confidence interval." This module produces those
two baselines and evaluates them on a validation window with the same
metric machinery a real model uses, so any Phase-2+ model's numbers can
be reported side-by-side.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import (
    per_zone_metrics, INSUFFICIENT,
)


def persistence_prediction(truth: np.ndarray, fallback_first_day: np.ndarray | None = None) -> np.ndarray:
    """``pred[t] = truth[t-1]``. First row filled from ``fallback_first_day``
    (typically the day right before the val window) so we don't cheat."""
    T = truth.shape[0]
    pred = np.empty_like(truth)
    pred[1:] = truth[:-1]
    if fallback_first_day is None:
        pred[0] = truth[0]     # degenerate: only used when caller doesn't have a fallback
    else:
        pred[0] = fallback_first_day
    return pred


def climatology_prediction(
    doys: np.ndarray,            # (T,) day-of-year 1..366
    per_cell_clim: np.ndarray,   # (366, H, W) precomputed from train years only
) -> np.ndarray:
    """``pred[t] = per_cell_clim[doy(t) - 1]``."""
    return per_cell_clim[doys - 1]


def per_zone_baseline_skill(
    truth: np.ndarray,           # (T, H, W)
    persistence: np.ndarray,     # (T, H, W)
    climatology: np.ndarray,     # (T, H, W)
    times_months: np.ndarray,
    zones, mask,
    thresholds_absolute: dict[str, float],
    thresholds_zone_percentile: dict[str, dict[str, float]],
    min_cells: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run the full metric cross-product on persistence + climatology.

    Returns::
        {"persistence": <per_zone_metrics-shape dict>,
         "climatology": <same>}
    """
    pers = per_zone_metrics(
        persistence, truth, times_months, zones, mask,
        thresholds_absolute, thresholds_zone_percentile, min_cells,
    )
    clim = per_zone_metrics(
        climatology, truth, times_months, zones, mask,
        thresholds_absolute, thresholds_zone_percentile, min_cells,
    )
    return {"persistence": pers, "climatology": clim}
