"""
train.eval.validate — the tiered validation engine.

Design (from plan §3):
  TIER 1 (every 50 batches, cheap)      — global loss + global RMSE.
  TIER 2 (every epoch, medium)           — per-zone RMSE / MAE / skill-vs-baseline.
  TIER 3 (every 5 epochs, expensive)     — 9 zones × 4 seasons × N metrics × K thresholds.
  TIER 4 (end of round, very expensive)  — bootstrap CIs + paired Wilcoxon vs baselines.

Every tier is a pure function that takes prediction/truth arrays and returns
a JSON-serialisable dict. The trainer decides when to call each — that
lives in Phase 4/loop; this module just provides the callable primitives.

Small-sample honesty is enforced by ``metrics.INSUFFICIENT``: if a
(zone × season × threshold) cell has fewer than ``min_cells`` valid
observations, the returned value is the sentinel string, NOT a number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .metrics import (
    weighted_rmse, weighted_mae, INSUFFICIENT, per_zone_metrics,
)
from .significance import bootstrap_ci, paired_wilcoxon


ValidationTier = Literal["tier1", "tier2", "tier3", "tier4"]


# ---------------------------------------------------------------------------
# Tier 1 — global loss + global RMSE (cheap, feeds the live curve)
# ---------------------------------------------------------------------------
def run_tier1(
    pred: np.ndarray, truth: np.ndarray,
    weight: np.ndarray, mask: np.ndarray,
    min_cells: int = 5,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rmse = weighted_rmse(pred, truth, weight, mask, min_cells)
    dt = time.perf_counter() - t0
    return {"tier": "tier1", "rmse": rmse, "runtime_s": round(dt, 4)}


# ---------------------------------------------------------------------------
# Tier 2 — per-zone RMSE + MAE + skill-vs-baseline (medium)
# ---------------------------------------------------------------------------
def run_tier2(
    pred: np.ndarray, truth: np.ndarray,
    zones, mask: np.ndarray,
    baseline_rmses: dict[str, float] | None = None,
    min_cells: int = 5,
) -> dict[str, Any]:
    """Per-zone RMSE + MAE + skill (relative to best baseline if supplied).

    Skill = 1 - (model_rmse / baseline_rmse). >0 means better than baseline.
    """
    t0 = time.perf_counter()
    K = zones.n_zones()
    memb = zones.membership.astype(np.float32)
    per_zone: dict[str, dict[str, Any]] = {}
    for k, zone in enumerate(zones.zones):
        w = memb[..., k]
        rmse = weighted_rmse(pred, truth, w, mask, min_cells)
        mae = weighted_mae(pred, truth, w, mask, min_cells)
        block = {"rmse": rmse, "mae": mae}
        if baseline_rmses and zone.key in baseline_rmses:
            b = baseline_rmses[zone.key]
            if isinstance(rmse, float) and isinstance(b, float) and b > 0:
                block["skill_vs_baseline"] = 1.0 - rmse / b
                block["baseline_rmse"] = b
            else:
                block["skill_vs_baseline"] = INSUFFICIENT
        per_zone[zone.key] = block
    dt = time.perf_counter() - t0
    return {"tier": "tier2", "per_zone": per_zone, "runtime_s": round(dt, 4)}


# ---------------------------------------------------------------------------
# Tier 3 — full cross-product (zone × season × metric × threshold)
# ---------------------------------------------------------------------------
def run_tier3(
    pred: np.ndarray, truth: np.ndarray,
    times_months: np.ndarray,
    zones, mask: np.ndarray,
    thresholds_absolute: dict[str, float],
    thresholds_zone_percentile: dict[str, dict[str, float]],
    min_cells: int = 5,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = per_zone_metrics(
        pred, truth, times_months, zones, mask,
        thresholds_absolute, thresholds_zone_percentile, min_cells,
    )
    dt = time.perf_counter() - t0
    return {"tier": "tier3", "per_zone_season": result, "runtime_s": round(dt, 4)}


# ---------------------------------------------------------------------------
# Tier 4 — bootstrap CIs + paired Wilcoxon vs baselines
# ---------------------------------------------------------------------------
def _per_cell_day_errors(
    pred: np.ndarray, truth: np.ndarray,
    weight: np.ndarray, mask: np.ndarray,
    common_finite: np.ndarray | None = None,
) -> np.ndarray:
    """Return a flat 1-D array of soft-weighted signed errors on land cells.

    ``common_finite`` (optional) is a shared boolean array that intersects
    finite-masks across multiple predictions being compared — required for
    paired significance tests (all comparators must have the same length)."""
    w = np.broadcast_to(weight[None, :, :], pred.shape)
    if common_finite is None:
        finite = np.isfinite(pred) & np.isfinite(truth) & (
            np.broadcast_to(mask[None, :, :], pred.shape) > 0
        )
    else:
        finite = common_finite
    diff = (pred - truth) * w
    return diff[finite & (w > 0.05)]


def run_tier4(
    pred: np.ndarray, truth: np.ndarray,
    baseline_pers: np.ndarray, baseline_clim: np.ndarray,
    zones, mask: np.ndarray,
    bootstrap_samples: int = 200,
    min_pairs: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """For each zone, produce:
        rmse with 95% bootstrap CI,
        paired Wilcoxon vs persistence,
        paired Wilcoxon vs climatology.
    """
    t0 = time.perf_counter()
    K = zones.n_zones()
    memb = zones.membership.astype(np.float32)

    per_zone: dict[str, Any] = {}
    for k, zone in enumerate(zones.zones):
        w = memb[..., k]
        # Shared finite mask so paired samples have equal length
        common_finite = (
            np.isfinite(pred) & np.isfinite(baseline_pers) & np.isfinite(baseline_clim)
            & np.isfinite(truth) & (np.broadcast_to(mask[None, :, :], pred.shape) > 0)
        )
        err_ours = _per_cell_day_errors(pred, truth, w, mask, common_finite)
        err_pers = _per_cell_day_errors(baseline_pers, truth, w, mask, common_finite)
        err_clim = _per_cell_day_errors(baseline_clim, truth, w, mask, common_finite)

        block: dict[str, Any] = {"n_samples": int(err_ours.size)}
        if err_ours.size < min_pairs:
            block["rmse_ci"] = INSUFFICIENT
            block["vs_persistence"] = INSUFFICIENT
            block["vs_climatology"] = INSUFFICIENT
        else:
            # RMSE as sqrt(mean(err²)) — bootstrap the squared errors
            mean, lo, hi = bootstrap_ci(
                err_ours ** 2, stat=np.mean, n_boot=bootstrap_samples, seed=seed,
            )
            block["rmse_ci"] = {
                "point": float(np.sqrt(mean)) if np.isfinite(mean) else INSUFFICIENT,
                "ci95_lower": float(np.sqrt(max(lo, 0.0))) if np.isfinite(lo) else INSUFFICIENT,
                "ci95_upper": float(np.sqrt(max(hi, 0.0))) if np.isfinite(hi) else INSUFFICIENT,
            }
            block["vs_persistence"] = paired_wilcoxon(err_ours, err_pers)
            block["vs_climatology"] = paired_wilcoxon(err_ours, err_clim)

        per_zone[zone.key] = block

    dt = time.perf_counter() - t0
    return {"tier": "tier4", "per_zone": per_zone, "runtime_s": round(dt, 4)}


# ---------------------------------------------------------------------------
# Per-zone early-stopping state
# ---------------------------------------------------------------------------
@dataclass
class ZoneConvergence:
    """Track per-zone best metric + patience counter."""
    best: float = float("inf")
    patience_counter: int = 0
    best_epoch: int = 0
    converged: bool = False


def per_zone_convergence_state(
    zones,
    tier2_history: list[dict[str, Any]],
    monitor_key: str = "rmse",
    patience: int = 8,
    min_delta: float = 1e-4,
    policy: str = "worst_zone",
) -> dict[str, Any]:
    """Given a list of Tier-2 reports (one per epoch), determine whether the
    zone-aware early-stopping policy triggers.

    Returns ``{"should_stop": bool, "reason": str, "per_zone": {key: ZoneConvergence-as-dict}}``.
    """
    per_zone: dict[str, ZoneConvergence] = {z.key: ZoneConvergence() for z in zones.zones}
    if not tier2_history:
        return {"should_stop": False, "reason": "no history", "per_zone": {}}

    for epoch_i, report in enumerate(tier2_history):
        pz = report["per_zone"]
        for k, zc in per_zone.items():
            v = pz.get(k, {}).get(monitor_key, None)
            if not isinstance(v, float) or not np.isfinite(v):
                continue
            if v < zc.best - min_delta:
                zc.best = float(v)
                zc.best_epoch = epoch_i
                zc.patience_counter = 0
            else:
                zc.patience_counter += 1
            if zc.patience_counter >= patience:
                zc.converged = True

    if policy == "worst_zone":
        # Stop when every zone has converged (worst-of ensemble stopped improving)
        should_stop = all(zc.converged for zc in per_zone.values())
        reason = ("all zones converged (worst_zone policy)" if should_stop
                  else "waiting for slowest zone")
    elif policy == "weighted":
        # Stop when the (loss-weight mean over zones) hasn't improved for patience
        weights = np.array([z.loss_weight if hasattr(z, "loss_weight") else 1.0
                            for z in zones.zones])
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        weighted_hist = []
        for report in tier2_history:
            v = 0.0
            for zi, zone in enumerate(zones.zones):
                m = report["per_zone"].get(zone.key, {}).get(monitor_key, None)
                if isinstance(m, float) and np.isfinite(m):
                    v += weights[zi] * m
            weighted_hist.append(v)
        best_ep = int(np.argmin(weighted_hist))
        should_stop = len(weighted_hist) - best_ep - 1 >= patience
        reason = (f"weighted policy: best epoch was {best_ep}, "
                  f"patience {len(weighted_hist) - best_ep - 1}")
    else:  # global
        # Just the global-ish "any-zone still improving?"
        should_stop = any(zc.converged for zc in per_zone.values())
        reason = "global policy: at least one zone stopped improving"

    return {
        "should_stop": bool(should_stop),
        "reason": reason,
        "per_zone": {k: vars(v) for k, v in per_zone.items()},
    }
