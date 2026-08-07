"""whatif.ui.copy.banners — permanent banners and status strings.

Never edit these by concatenation in panel code; the wording is
carefully scoped and reviewed.
"""
from __future__ import annotations

PERTURBATION_CAVEAT = (
    "**Method 1 (delta perturbation) is physically inconsistent by "
    "construction.** A 20 % rainfall reduction with unchanged humidity, "
    "wind, and radiation is not a physically realisable world "
    "(Räisänen & Räty 2013). Use only for exploratory sensitivity. "
    "Prefer historical analogs (Method 2) for reported scenarios."
)

CAVEAT_ACKNOWLEDGE_CHECKBOX = (
    "I acknowledge the physical-inconsistency caveat and understand "
    "this run should not be exported as a report."
)

LONG_TERM_SCENARIO_BANNER = (
    "**Long Term is a scenario tab, not a forecast tab.** Values shown "
    "here are conditional trajectories under a stated pathway, not "
    "predictions of what will happen."
)

NO_STRONG_ANALOG_HEADLINE = (
    "No strong or fair historical analog exists for this target in the "
    "training record."
)

BACKTEST_FAILED_TAG = "Rule failed — does not beat climatology"

EXPORT_BLOCKED_DIRTY = (
    "Export blocked — the code tree is dirty (uncommitted changes). "
    "Commit or stash, then rerun to export."
)

EXPORT_BLOCKED_NO_CAVEAT = (
    "Export blocked — this scenario used the delta-perturbation flavour "
    "without acknowledging the physical-inconsistency caveat. Set the "
    "caveat checkbox and rerun, or switch to Method 2 (analogs)."
)

RESOLUTION_CEILING_NOTE = (
    "Snapped to the nearest 0.25° grid cell — the What-If engine "
    "refuses sub-district readouts."
)

DEV_ONLY_HEADER = (
    "🛠️ Developer diagnostic view — hidden by default. "
    "Access via `?dev=1` query parameter."
)
