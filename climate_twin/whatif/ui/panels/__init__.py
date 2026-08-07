"""whatif.ui.panels — Streamlit panel functions.

Each panel is a pure function ``(state, result) -> None`` that reads
from a ``WhatIfState`` and a ``ScenarioResult`` (both provided by the
page orchestrator), and emits Streamlit widgets. Panels never perform
arithmetic and never talk to the driver / sector / economics layers
directly; everything comes pre-computed via the engine call.
"""
from .analogs import render_analog_narrative
from .backtest import render_backtest_card
from .context import render_context_strip
from .levers import render_levers
# Part 7: the live Long-Term panel replaces the Part-6 shell.
# The shell is kept alongside for reference / rollback but no longer
# wired into pages/30_What_If.py.
from .long_term import render_long_term_tab
from .payoff import render_payoff_matrix
from .provenance import render_provenance_drawer
from .recommendation import render_recommendation
from .save_load_export import render_save_load_export_bar
from .sensitivity import render_sensitivity_tornado

__all__ = [
    "render_context_strip", "render_levers",
    "render_payoff_matrix", "render_recommendation",
    "render_analog_narrative", "render_sensitivity_tornado",
    "render_backtest_card", "render_provenance_drawer",
    "render_save_load_export_bar", "render_long_term_tab",
]
