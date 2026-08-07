"""
whatif.ui.copy.long_term — Long Term tab strings.

Rule 1 (Part 7 vocab lint): none of ``predict``, ``forecast``,
``will``, or ``is going to`` may appear here. The module-level test
:mod:`whatif.tests.test_ui` enforces the ban.

Allowed verbs: ``under``, ``if the world follows``, ``conditional on``,
``projected under scenario``, ``would``. Every conditional-trajectory
template uses one of them explicitly.
"""
from __future__ import annotations

SSP_PLACEHOLDER = (
    "Under SSP1-2.6 / SSP2-4.5 / SSP5-8.5 the tab renders conditional "
    "trajectories at 2030 / 2050 / 2075 horizons, downscaled from "
    "NEX-GDDP-CMIP6."
)

CONDITIONAL_VERB_TEMPLATES = (
    "If the world follows {ssp}, {sector} sees {outcome} in {year}.",
    "Under the {ssp} pathway, the conditional trajectory for "
    "{sector} at {year} would be {outcome}.",
    "Conditional on {ssp} through {year}, {sector} is on a trajectory "
    "of {outcome}.",
)

HORIZON_LABELS = ("2030", "2050", "2075")

# Displayed labels use the SSP short-forms from the registry
SSP_CHOICES = ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5")

DEFAULT_SSP_DISPLAY = ("ssp126", "ssp245", "ssp585")

BASELINE_LABEL = "Baseline: 1971-2000 (IMD)"

WINDOW_NOTE = (
    "All Long-Term values are 20-year climatological windows centred "
    "on the target year (2030 = 2021-2040, 2050 = 2041-2060, "
    "2075 = 2066-2085) to average out internal variability."
)

CONFIDENCE_NOTE = (
    "'Confidence' in Long Term reports inter-model agreement and "
    "emergence significance. It is not a claim of skill against "
    "out-of-sample climate."
)

DELTA_MEAN_EXTREMES_CAVEAT = (
    "Delta-mean downscaling preserves the observed mean state but "
    "does NOT capture tail shifts. For extremes (Rx1day, Rx5day, "
    "Tmax extreme) use QDM (Cannon 2018). Tick the checkbox to "
    "proceed anyway."
)

SMALL_ENSEMBLE_CAVEAT = (
    "Fewer than five GCMs in the ensemble collapses the model-"
    "uncertainty estimate. Tick the checkbox to proceed anyway."
)

RECOMMENDATION_TEMPLATE = (
    "**Under {ssp_label}, the best-NPV adaptation for {region} "
    "{crop} at the {horizon} horizon would be _{option}_: "
    "NPV {npv_08}/ha at 8% discount "
    "(range {npv_12} at 12% to {npv_07} at 7%), BCR {bcr}. "
    "This assumes the world follows {ssp_label} — it is a "
    "conditional trajectory, not a claim about what happens.**"
)

DISABLED_HINT = "Available Part 7 onward."
