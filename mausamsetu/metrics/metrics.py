"""
metrics.py
===========
Meteorological verification metrics for MausamSetu.

METRICS IMPLEMENTED (in the language meteorologists actually use)
------------------------------------------------------------------
Deterministic (per-pixel error):
  - MAE   : Mean Absolute Error
  - RMSE  : Root Mean Squared Error
  - Bias  : Mean signed error (pred − true)
  - ACC   : Anomaly Correlation Coefficient

Precipitation "event" metrics (from the contingency table):
  Hit         = pred rain AND obs rain
  Miss        = no pred rain BUT obs rain
  False Alarm = pred rain BUT no obs rain
  Correct Neg = no pred rain AND no obs rain

  - POD (Probability of Detection) = Hits / (Hits + Misses)         [0-1, higher better]
  - FAR (False Alarm Ratio)         = FA / (Hits + FA)               [0-1, lower  better]
  - CSI (Critical Success Index)    = Hits / (Hits + Misses + FA)    [0-1, higher better]
  - HSS (Heidke Skill Score)        — accounts for random chance

Ensemble/probabilistic:
  - CRPS (Continuous Ranked Probability Score)  — how good is your DISTRIBUTION
  - Reliability                                  — is your uncertainty calibrated?

USAGE
-----
    from mausamsetu.metrics.metrics import compute_all
    metrics = compute_all(y_pred, y_true, ensemble=None)
    # metrics is a dict of scalars

    from mausamsetu.metrics.metrics import contingency
    hits, misses, fa, cn = contingency(y_pred, y_true, threshold=1.0)
"""
from __future__ import annotations
import numpy as np


# ============================================================================
# BASIC DETERMINISTIC METRICS
# ============================================================================
def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error. Lower = better."""
    return float(np.nanmean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error. Lower = better."""
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))


def bias(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean bias (pred - true). 0 = unbiased."""
    return float(np.nanmean(pred - true))


def acc(pred_anom: np.ndarray, true_anom: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient.

    Both inputs must be ANOMALIES (deviation from climatology).
    Range: -1 to 1. Higher = better spatial pattern agreement.

    ACC = <pred_anom · true_anom> / sqrt(<pred_anom²> · <true_anom²>)
    """
    p = pred_anom.flatten()
    t = true_anom.flatten()
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    if len(p) == 0:
        return float("nan")
    num = np.mean(p * t)
    den = np.sqrt(np.mean(p ** 2) * np.mean(t ** 2)) + 1e-12
    return float(num / den)


# ============================================================================
# CONTINGENCY TABLE (for precipitation events)
# ============================================================================
def contingency(pred: np.ndarray, true: np.ndarray, threshold: float = 1.0) -> tuple:
    """
    Build the 2x2 contingency table for a binary "rain event" definition.

    An "event" = value > threshold.

    Returns
    -------
    hits, misses, false_alarms, correct_negatives : int
    """
    pred_event = pred > threshold
    true_event = true > threshold

    hits = int(np.sum(pred_event & true_event))
    misses = int(np.sum(~pred_event & true_event))
    false_alarms = int(np.sum(pred_event & ~true_event))
    correct_negatives = int(np.sum(~pred_event & ~true_event))
    return hits, misses, false_alarms, correct_negatives


# ============================================================================
# METEOROLOGY EVENT METRICS
# ============================================================================
def pod(pred: np.ndarray, true: np.ndarray, threshold: float = 1.0) -> float:
    """Probability of Detection. Higher = better (max 1.0)."""
    H, M, _, _ = contingency(pred, true, threshold)
    return H / (H + M + 1e-12)


def far(pred: np.ndarray, true: np.ndarray, threshold: float = 1.0) -> float:
    """False Alarm Ratio. Lower = better (min 0.0)."""
    H, _, FA, _ = contingency(pred, true, threshold)
    return FA / (H + FA + 1e-12)


def csi(pred: np.ndarray, true: np.ndarray, threshold: float = 1.0) -> float:
    """Critical Success Index (Threat Score). Higher = better (max 1.0)."""
    H, M, FA, _ = contingency(pred, true, threshold)
    return H / (H + M + FA + 1e-12)


def hss(pred: np.ndarray, true: np.ndarray, threshold: float = 1.0) -> float:
    """
    Heidke Skill Score. Corrects for the score expected by chance.
    Range: -1 to 1. 0 = no skill vs random, 1 = perfect.
    """
    H, M, FA, CN = contingency(pred, true, threshold)
    total = H + M + FA + CN
    if total == 0:
        return float("nan")
    exp_corr = ((H + M) * (H + FA) + (CN + M) * (CN + FA)) / total
    obs_corr = H + CN
    denom = total - exp_corr
    return (obs_corr - exp_corr) / (denom + 1e-12)


# ============================================================================
# PROBABILISTIC METRIC — CRPS
# ============================================================================
def crps_ensemble(ensemble: np.ndarray, true: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score for an ensemble forecast.

    ensemble: shape (N, ...) — N members
    true:     shape (...)

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Lower = better. 0 = perfect (all members = truth).
    """
    N = ensemble.shape[0]
    # Term 1: mean absolute error of members vs truth
    abs_err = np.abs(ensemble - true[None, ...])
    term1 = np.nanmean(abs_err)

    # Term 2: mean pairwise distance between ensemble members
    # For efficiency: |X - X'| pairs
    diff = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    term2 = 0.5 * np.nanmean(diff)

    return float(term1 - term2)


def spread_skill_ratio(ensemble: np.ndarray, true: np.ndarray) -> float:
    """
    Ratio of ensemble spread to RMSE of ensemble mean.

    Well-calibrated ensembles have ratio close to 1.0:
      < 1 → ensemble is UNDER-dispersive (over-confident)
      > 1 → ensemble is OVER-dispersive (under-confident)
    """
    mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0).mean()
    err = np.sqrt(np.nanmean((mean - true) ** 2))
    return float(spread / (err + 1e-12))


# ============================================================================
# COMPUTE ALL — one call for the validation table
# ============================================================================
def compute_all(
    pred: np.ndarray,
    true: np.ndarray,
    pred_anom: np.ndarray | None = None,
    true_anom: np.ndarray | None = None,
    ensemble: np.ndarray | None = None,
    rain_threshold: float = 1.0,
) -> dict:
    """
    Compute a comprehensive metrics dictionary.

    pred, true : arrays of predictions and observations (same shape)
    pred_anom, true_anom : optional, for ACC (arrays as anomalies)
    ensemble : optional ensemble (N, ...) for CRPS
    """
    results = {
        "MAE":  mae(pred, true),
        "RMSE": rmse(pred, true),
        "Bias": bias(pred, true),
        "POD@1mm":  pod(pred, true, threshold=1.0),
        "FAR@1mm":  far(pred, true, threshold=1.0),
        "CSI@1mm":  csi(pred, true, threshold=1.0),
        "HSS@1mm":  hss(pred, true, threshold=1.0),
        "POD@10mm": pod(pred, true, threshold=10.0),
        "FAR@10mm": far(pred, true, threshold=10.0),
        "CSI@10mm": csi(pred, true, threshold=10.0),
    }
    if pred_anom is not None and true_anom is not None:
        results["ACC"] = acc(pred_anom, true_anom)
    if ensemble is not None:
        results["CRPS"] = crps_ensemble(ensemble, true)
        results["SpreadSkill"] = spread_skill_ratio(ensemble, true)
    return results


# ============================================================================
# CLI SANITY CHECK
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_data = np.clip(rng.exponential(5.0, size=(100, 19, 17)), 0, 200)
    # Perfect prediction
    pred_perfect = true_data.copy()
    print("Perfect prediction:")
    for k, v in compute_all(pred_perfect, true_data).items():
        print(f"  {k}: {v:.4f}")

    # Noisy prediction
    pred_noisy = true_data + rng.normal(0, 3, size=true_data.shape)
    print("\nNoisy prediction (Gaussian σ=3):")
    for k, v in compute_all(pred_noisy, true_data).items():
        print(f"  {k}: {v:.4f}")

    # Ensemble CRPS
    ensemble = np.stack([true_data + rng.normal(0, 2, true_data.shape) for _ in range(10)])
    print("\n10-member ensemble:")
    print(f"  CRPS = {crps_ensemble(ensemble, true_data):.4f}")
    print(f"  Spread-skill = {spread_skill_ratio(ensemble, true_data):.4f}")
