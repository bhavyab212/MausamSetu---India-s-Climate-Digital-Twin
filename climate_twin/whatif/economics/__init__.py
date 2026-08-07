"""
whatif.economics — L4, monetise the physical sector outputs.

Public entry points:
    * :func:`load_prices`, :class:`PriceSet` — YAML-backed price registry.
    * :func:`value_agriculture`, :class:`EconomicOutcome` — physical
      yield → ₹/ha with quantile-integrity guard.
    * :func:`build_payoff_matrix`, :class:`Decision`, :class:`ClimateState`,
      :class:`PayoffMatrix` — decision × state grid.
    * :func:`expected_value`, :func:`regret_matrix`, :func:`minimax_regret`,
      :func:`var_cvar`, :func:`stochastic_dominance`, :func:`recommend`.
    * :func:`value_curve`, :class:`CostLossSetup`, :class:`ValueCurve`,
      ``SHIPPED_SETUPS`` — Murphy 1977 cost–loss + verification bundle.
    * :func:`walk_forward_backtest`, :class:`BacktestResult`,
      :func:`synthesise_history`, :class:`HistoricalFrequencyRule`,
      :class:`OnsetAnomalyRule` — evaluation harness.
    * :func:`tornado`, :class:`TornadoResult`, :class:`Knob`,
      ``DEFAULT_KNOBS`` — OFAT sensitivity.

Rules enforced (see Part 4 Golden rules):
    1. No bare rupees — every ₹ output is a q10/q50/q90 dict.
    2. Prices are a lever — MSP season is mandatory; live mandi feed
       is off by default.
    3. Every ₹ output ships with its climatology counterpart.
    4. Sensitivity analysis is mandatory (see :func:`tornado`).
    5. Resolution ceiling: district. Grid-cell inputs raise.
    6. Backtest publishes negative results honestly.
    7. Determinism: same YAML ⇒ same rupees.
"""
from __future__ import annotations

from .backtest import (
    BACKTEST_VERSION,
    BacktestResult,
    DecisionRule,
    HistoricalFrequencyRule,
    OnsetAnomalyRule,
    get_last_backtest,
    synthesise_history,
    walk_forward_backtest,
)
from .cost_loss import (
    COST_LOSS_VERSION,
    SHIPPED_SETUPS,
    CostLossSetup,
    ValueCurve,
    brier_score,
    brier_skill_score,
    reliability_diagram,
    roc_auc,
    sharpness_histogram,
    value_curve,
)
from .decision import (
    expected_value,
    expected_value_with_uncertainty,
    minimax_regret,
    recommend,
    regret_matrix,
    stochastic_dominance,
    var_cvar,
    worst_case,
)
from .payoff import ClimateState, Decision, PayoffMatrix, build_payoff_matrix
from .prices import (
    PriceSet,
    list_priced_crops,
    load_agmarknet_series,
    load_prices,
    registry_sha256 as prices_registry_sha256,
    registry_version as prices_registry_version,
)
from .sensitivity import DEFAULT_KNOBS, Knob, TornadoResult, tornado
from .valuation import (
    VALUATION_VERSION,
    EconomicOutcome,
    SchemaError,
    value_agriculture,
)

__all__ = [
    # prices
    "PriceSet", "load_prices", "list_priced_crops", "load_agmarknet_series",
    "prices_registry_version", "prices_registry_sha256",
    # valuation
    "EconomicOutcome", "SchemaError", "value_agriculture", "VALUATION_VERSION",
    # payoff
    "Decision", "ClimateState", "PayoffMatrix", "build_payoff_matrix",
    # decision
    "expected_value", "expected_value_with_uncertainty",
    "regret_matrix", "minimax_regret", "worst_case", "var_cvar",
    "stochastic_dominance", "recommend",
    # cost–loss
    "CostLossSetup", "ValueCurve", "value_curve", "SHIPPED_SETUPS",
    "brier_score", "brier_skill_score", "reliability_diagram",
    "roc_auc", "sharpness_histogram", "COST_LOSS_VERSION",
    # backtest
    "DecisionRule", "HistoricalFrequencyRule", "OnsetAnomalyRule",
    "BacktestResult", "walk_forward_backtest", "get_last_backtest",
    "synthesise_history", "BACKTEST_VERSION",
    # sensitivity
    "Knob", "DEFAULT_KNOBS", "TornadoResult", "tornado",
]
