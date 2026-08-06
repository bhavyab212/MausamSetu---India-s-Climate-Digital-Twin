"""train.eval — zone-aware tiered validation."""
from .metrics import (
    INSUFFICIENT,
    weighted_rmse,
    weighted_mae,
    weighted_bias,
    thresholded_scores,
    crps_gaussian,
    per_zone_metrics,
    seasons_of,
)
from .baselines import (
    persistence_prediction,
    climatology_prediction,
    per_zone_baseline_skill,
)
from .significance import bootstrap_ci, paired_wilcoxon
from .validate import (
    ValidationTier,
    run_tier1, run_tier2, run_tier3, run_tier4,
    per_zone_convergence_state,
)
from .heatmap import render_tier3_heatmap

__all__ = [
    "INSUFFICIENT",
    "weighted_rmse", "weighted_mae", "weighted_bias",
    "thresholded_scores", "crps_gaussian",
    "per_zone_metrics", "seasons_of",
    "persistence_prediction", "climatology_prediction", "per_zone_baseline_skill",
    "bootstrap_ci", "paired_wilcoxon",
    "ValidationTier",
    "run_tier1", "run_tier2", "run_tier3", "run_tier4",
    "per_zone_convergence_state",
    "render_tier3_heatmap",
]
