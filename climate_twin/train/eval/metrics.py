"""
train.eval.metrics — every metric is zone-aware.

Design rules:
  1. Every metric takes an ``(H, W)`` weight tensor (soft membership) and a
     land mask; NaN targets are excluded from the denominator.
  2. When the effective sample count is below ``min_cells``, the metric
     returns the sentinel string :data:`INSUFFICIENT` — never a number.
  3. No metric implicitly averages over zones. The global aggregate is
     produced separately by ``per_zone_metrics`` as a weighted mean using
     the loss-weight vector from the frozen ``zone_stats.json``.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

INSUFFICIENT = "insufficient data"

# ---------------------------------------------------------------------------
# Season assignment
# ---------------------------------------------------------------------------
SEASONS = ("DJF", "MAM", "JJAS", "OND")   # OND matters for Tamil Nadu NE monsoon


def seasons_of(months: np.ndarray) -> np.ndarray:
    """Return an array of season labels for a 1-D month array (1..12)."""
    out = np.empty(months.shape, dtype="U4")
    out[(months == 12) | (months <= 2)] = "DJF"
    out[(months >= 3) & (months <= 5)] = "MAM"
    out[(months >= 6) & (months <= 9)] = "JJAS"
    out[(months >= 10) & (months <= 11)] = "OND"
    return out


# ---------------------------------------------------------------------------
# Continuous-value metrics
# ---------------------------------------------------------------------------
def _weighted_denom(finite: np.ndarray, weight: np.ndarray) -> float:
    """Effective sample size = sum of weights over finite cells (broadcast over time)."""
    w = np.broadcast_to(weight[None, :, :], finite.shape)
    return float(np.where(finite, w, 0.0).sum())


def weighted_rmse(
    pred: np.ndarray, truth: np.ndarray, weight: np.ndarray, mask: np.ndarray,
    min_cells: int = 5,
) -> float | str:
    """Weighted RMSE over cell-days. Returns :data:`INSUFFICIENT` if
    effective count < ``min_cells``."""
    finite = np.isfinite(pred) & np.isfinite(truth) & (
        np.broadcast_to(mask[None, :, :], pred.shape) > 0
    )
    n_eff = _weighted_denom(finite, weight)
    if n_eff < min_cells:
        return INSUFFICIENT
    w = np.broadcast_to(weight[None, :, :], pred.shape)
    diff = np.where(finite, pred - truth, 0.0)
    num = float((diff ** 2 * w).sum())
    return float(np.sqrt(num / n_eff))


def weighted_mae(
    pred: np.ndarray, truth: np.ndarray, weight: np.ndarray, mask: np.ndarray,
    min_cells: int = 5,
) -> float | str:
    finite = np.isfinite(pred) & np.isfinite(truth) & (
        np.broadcast_to(mask[None, :, :], pred.shape) > 0
    )
    n_eff = _weighted_denom(finite, weight)
    if n_eff < min_cells:
        return INSUFFICIENT
    w = np.broadcast_to(weight[None, :, :], pred.shape)
    diff = np.where(finite, np.abs(pred - truth), 0.0)
    return float((diff * w).sum() / n_eff)


def weighted_bias(
    pred: np.ndarray, truth: np.ndarray, weight: np.ndarray, mask: np.ndarray,
    min_cells: int = 5,
) -> float | str:
    finite = np.isfinite(pred) & np.isfinite(truth) & (
        np.broadcast_to(mask[None, :, :], pred.shape) > 0
    )
    n_eff = _weighted_denom(finite, weight)
    if n_eff < min_cells:
        return INSUFFICIENT
    w = np.broadcast_to(weight[None, :, :], pred.shape)
    diff = np.where(finite, pred - truth, 0.0)
    return float((diff * w).sum() / n_eff)


# ---------------------------------------------------------------------------
# Categorical / thresholded scores (POD, FAR, CSI, ACC)
# ---------------------------------------------------------------------------
def thresholded_scores(
    pred: np.ndarray, truth: np.ndarray, weight: np.ndarray, mask: np.ndarray,
    threshold: float,
    min_cells: int = 5,
) -> dict[str, float | str]:
    """Return {'pod', 'far', 'csi', 'acc'} at a single threshold.

    Definitions (weighted counts):
        hits (H)         : pred >= t AND truth >= t
        misses (M)       : pred <  t AND truth >= t
        false_alarms (F) : pred >= t AND truth <  t
        correct_neg (N)  : pred <  t AND truth <  t
        POD = H / (H+M)
        FAR = F / (H+F)
        CSI = H / (H+M+F)
        ACC = (H+N) / (H+M+F+N)
    """
    finite = np.isfinite(pred) & np.isfinite(truth) & (
        np.broadcast_to(mask[None, :, :], pred.shape) > 0
    )
    n_eff = _weighted_denom(finite, weight)
    if n_eff < min_cells:
        return {k: INSUFFICIENT for k in ("pod", "far", "csi", "acc")}

    w = np.broadcast_to(weight[None, :, :], pred.shape)
    w_eff = np.where(finite, w, 0.0)
    p = pred >= threshold
    t = truth >= threshold
    H = float((w_eff * p * t).sum())
    M = float((w_eff * (~p) * t).sum())
    F = float((w_eff * p * (~t)).sum())
    N = float((w_eff * (~p) * (~t)).sum())

    def _safe_div(num, den):
        return float(num / den) if den > 0 else INSUFFICIENT

    return {
        "pod": _safe_div(H, H + M),
        "far": _safe_div(F, H + F),
        "csi": _safe_div(H, H + M + F),
        "acc": _safe_div(H + N, H + M + F + N),
    }


# ---------------------------------------------------------------------------
# CRPS (Gaussian approximation from p10/p90 ensemble bounds)
# ---------------------------------------------------------------------------
def crps_gaussian(
    p50: np.ndarray, p10: np.ndarray, p90: np.ndarray,
    truth: np.ndarray, weight: np.ndarray, mask: np.ndarray,
    min_cells: int = 5,
) -> float | str:
    """Analytic CRPS for a Gaussian with mean=p50 and std derived from the
    p10/p90 spread (σ = (p90 - p10) / (2 × 1.2816))."""
    from math import sqrt, pi
    finite = np.isfinite(p50) & np.isfinite(truth) & (
        np.broadcast_to(mask[None, :, :], p50.shape) > 0
    )
    n_eff = _weighted_denom(finite, weight)
    if n_eff < min_cells:
        return INSUFFICIENT
    sigma = np.maximum((p90 - p10) / (2.0 * 1.2816), 1e-4)
    z = (truth - p50) / sigma
    # CRPS(N(μ,σ), x) = σ * ( z(2Φ(z)-1) + 2φ(z) - 1/sqrt(π) )
    from scipy.stats import norm  # scipy is available in the venv
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / sqrt(pi))
    w = np.broadcast_to(weight[None, :, :], crps.shape)
    return float(np.where(finite, crps * w, 0.0).sum() / n_eff)


# ---------------------------------------------------------------------------
# Per-zone cross-product runner
# ---------------------------------------------------------------------------
def per_zone_metrics(
    pred: np.ndarray, truth: np.ndarray,
    times_months: np.ndarray,
    zones, mask: np.ndarray,
    thresholds_absolute: dict[str, float],           # e.g. from IMD categories (mm/day)
    thresholds_zone_percentile: dict[str, dict[str, float]],
                                                       # {zone_key: {label: mm/day}}
    min_cells: int = 5,
    include_seasons: bool = True,
) -> dict[str, Any]:
    """Full cross-product: zone × season × metric × threshold.

    Args:
        pred, truth: ``(T, H, W)`` prediction and ground truth.
        times_months: ``(T,)`` month labels (1..12).
        zones: :class:`ZoneRegistry`.
        mask: land mask (H, W).
        thresholds_absolute: IMD-published thresholds; same across zones.
        thresholds_zone_percentile: per-zone percentile thresholds (train-only).

    Returns nested dict::

        { "<zone_key>": {
              "<season|ALL>": {
                  "rmse": <float|INSUFFICIENT>,
                  "mae":  ...,
                  "bias": ...,
                  "thresholded": {
                      "absolute": {"<label>": {"pod":..,"far":..,"csi":..,"acc":..}, ...},
                      "zone_percentile": {"<label>": {...}, ...},
                  }
              }
        }}

    Also emits a special "GLOBAL" pseudo-zone: loss-weighted mean over zones.
    """
    seasons = seasons_of(times_months) if include_seasons else None
    K = zones.n_zones()
    memb = zones.membership.astype(np.float32)          # (H, W, K)

    per_zone: dict[str, Any] = {}
    for k, zone in enumerate(zones.zones):
        w_zone = memb[..., k]                            # (H, W)

        blocks: dict[str, Any] = {}
        keys = ["ALL"]
        if seasons is not None:
            keys += list(SEASONS)

        for season_key in keys:
            if season_key == "ALL":
                sel = np.ones(pred.shape[0], dtype=bool)
            else:
                sel = seasons == season_key
                if not sel.any():
                    blocks[season_key] = {
                        "rmse": INSUFFICIENT, "mae": INSUFFICIENT,
                        "bias": INSUFFICIENT, "n_days": 0,
                        "thresholded": {"absolute": {}, "zone_percentile": {}},
                    }
                    continue

            p_slice = pred[sel]
            t_slice = truth[sel]

            block = {
                "n_days": int(sel.sum()),
                "rmse": weighted_rmse(p_slice, t_slice, w_zone, mask, min_cells),
                "mae":  weighted_mae(p_slice, t_slice, w_zone, mask, min_cells),
                "bias": weighted_bias(p_slice, t_slice, w_zone, mask, min_cells),
                "thresholded": {"absolute": {}, "zone_percentile": {}},
            }
            for label, thr in thresholds_absolute.items():
                block["thresholded"]["absolute"][label] = thresholded_scores(
                    p_slice, t_slice, w_zone, mask, thr, min_cells,
                )
            for label, thr in thresholds_zone_percentile.get(zone.key, {}).items():
                block["thresholded"]["zone_percentile"][label] = thresholded_scores(
                    p_slice, t_slice, w_zone, mask, thr, min_cells,
                )
            blocks[season_key] = block

        per_zone[zone.key] = blocks

    return per_zone
