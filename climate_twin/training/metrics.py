"""Metrics for walk-forward training evaluation.

All metrics operate on numpy arrays and handle the India land mask.
Every metric is computed alongside persistence + climatology baselines.
"""

from __future__ import annotations

import numpy as np


def masked_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over valid (mask=1) cells."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    diff = pred[valid] - truth[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def masked_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """MAE over valid cells."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - truth[valid])))


def masked_bias(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean bias (pred - truth) over valid cells."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    return float(np.mean(pred[valid] - truth[valid]))


def masked_pearson(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation over valid cells."""
    valid = mask > 0.5
    if valid.sum() < 3:
        return float("nan")
    p, t = pred[valid], truth[valid]
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def pod(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, threshold: float = 0.01) -> float:
    """Probability of Detection (hit rate) for rain occurrence.

    threshold is in normalized [0,1] space matching the model output.
    """
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    p_occ = pred[valid] > threshold
    t_occ = truth[valid] > threshold
    hits = np.sum(p_occ & t_occ)
    misses = np.sum(~p_occ & t_occ)
    return float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")


def far(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, threshold: float = 0.01) -> float:
    """False Alarm Ratio."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    p_occ = pred[valid] > threshold
    t_occ = truth[valid] > threshold
    hits = np.sum(p_occ & t_occ)
    false_alarms = np.sum(p_occ & ~t_occ)
    return float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else float("nan")


def csi(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, threshold: float = 0.01) -> float:
    """Critical Success Index."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    p_occ = pred[valid] > threshold
    t_occ = truth[valid] > threshold
    hits = np.sum(p_occ & t_occ)
    misses = np.sum(~p_occ & t_occ)
    false_alarms = np.sum(p_occ & ~t_occ)
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else float("nan")


def ensemble_calibration(p10: np.ndarray, p90: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Empirical coverage of p10-p90 interval (target ~0.80)."""
    valid = mask > 0.5
    if not valid.any():
        return float("nan")
    inside = (truth[valid] >= p10[valid]) & (truth[valid] <= p90[valid])
    return float(np.mean(inside))


def compute_all_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Compute the full metric suite for one prediction frame."""
    return {
        "rmse": masked_rmse(pred, truth, mask),
        "mae": masked_mae(pred, truth, mask),
        "bias": masked_bias(pred, truth, mask),
        "pearson_r": masked_pearson(pred, truth, mask),
        "pod": pod(pred, truth, mask, threshold),
        "far": far(pred, truth, mask, threshold),
        "csi": csi(pred, truth, mask, threshold),
    }
