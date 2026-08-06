"""
climate_twin.training — Phase 0 shim package.

The previous walk-forward, region-aware training system has been archived at
``climate_twin/_archive/training_pre_zones/`` as part of the transition to a
zone-aware training rebuild (Phase 0-5 of the training rebuild plan).

This shim exists ONLY so ``import climate_twin.app_v2`` keeps working while
the new ``climate_twin/train/`` package is being built. Every function still
imports cleanly, but any actual call raises ``TrainingRebuildInProgress``
with an explanatory message. The Streamlit UI's Training tab detects the
shim and shows a banner instead of the training controls.

Do NOT restore the archived training area. It was designed for a single
global normalisation and a single global RMSE — provably incorrect on
India's ~70× rainfall gradient. See ``_archive/training_pre_zones/README.md``
for the full rationale.
"""
from __future__ import annotations


REBUILD_BANNER_TEXT = (
    "⚙️ **Training area rebuild in progress**\n\n"
    "The walk-forward, region-aware training system has been archived. It is "
    "being replaced with a **zone-aware** system that respects India's ~70× "
    "rainfall gradient (Mawsynram ~11,000 mm/yr vs Thar <150 mm/yr) — one "
    "global normalisation and one global RMSE are statistically invalid on "
    "data that heteroscedastic.\n\n"
    "The new system will provide 9-zone stratified sampling, FiLM zone "
    "conditioning, per-zone loss weights, per-zone validation, and honest "
    "per-zone confidence intervals.\n\n"
    "Every other tab (Maps, Daily Explorer, Forecasts, Animation, RL, "
    "Analytics) continues to work. Only this tab is offline while the "
    "rebuild is in progress.\n\n"
    "See `climate_twin/_phase0/PHASE0_REPORT.md` for the current status."
)


class TrainingRebuildInProgress(RuntimeError):
    """Raised by any archived training symbol reached via the shim."""

    def __init__(self, symbol: str = "training"):
        super().__init__(
            f"climate_twin.training.{symbol} is archived while the "
            f"zone-aware training system is being built. "
            f"See _archive/training_pre_zones/README.md and "
            f"_phase0/PHASE0_REPORT.md."
        )


__all__ = ["REBUILD_BANNER_TEXT", "TrainingRebuildInProgress"]
