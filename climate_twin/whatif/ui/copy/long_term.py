"""
whatif.ui.copy.long_term — Long Term tab strings.

Rule 9: never say "prediction" or "forecast" here. The module-level
test :mod:`whatif.tests.test_ui` enforces the ban by scanning this
module's exported strings.
"""
from __future__ import annotations

SSP_PLACEHOLDER = (
    "SSP scenario wiring arrives in Part 7 (NEX-GDDP-CMIP6). "
    "Under SSP2-4.5 / SSP3-7.0 / SSP5-8.5 the tab will show "
    "conditional trajectories at 2030 / 2050 / 2075 horizons."
)

CONDITIONAL_VERB_TEMPLATES = (
    "If the world follows {ssp}, {sector} sees {outcome} in {year}.",
    "Under the {ssp} pathway, the conditional trajectory for "
    "{sector} at {year} is {outcome}.",
    "Assuming {ssp} through {year}, {sector} is on a trajectory of "
    "{outcome}.",
)

HORIZON_LABELS = ("2030", "2050", "2075")

SSP_CHOICES = ("SSP2-4.5", "SSP3-7.0", "SSP5-8.5")

DISABLED_HINT = "Enabled in Part 7."
