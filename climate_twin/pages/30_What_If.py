"""
pages/30_What_If.py — Scenario Engine (Part 6 UI).

Thin orchestrator. Reads WhatIfState, calls the engine once through the
cached adapter, delegates each panel to a pure ``(state, result) -> None``
function under :mod:`whatif.ui.panels`.

The diagnostic self-check expander from Parts 1–5 is preserved as a
developer-only view accessible via ``?dev=1`` in the URL. The shipped
page is the layout described in the Part 6 spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve `climate_twin.*` package imports when Streamlit invokes this
# page directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="What If — Scenario Engine · MausamSetu",
    page_icon="❓",
    layout="wide",
)

# ── Dev diagnostic switch ────────────────────────────────────────────
# `?dev=1` in the URL brings back the Parts-1–5 self-check expander.
# Everything else in this file is the Part-6 shipped UI.
_dev_flag = False
try:
    _dev_flag = st.query_params.get("dev") == "1"
except Exception:
    # Older Streamlit APIs stashed query params under experimental_
    try:
        _dev_flag = st.experimental_get_query_params().get("dev", ["0"])[0] == "1"
    except Exception:
        _dev_flag = False


from climate_twin.whatif.ui.copy import banners as _banners
from climate_twin.whatif.ui.copy import context as _ctxcopy
from climate_twin.whatif.ui.engine_adapter import cached_run_for_state
from climate_twin.whatif.ui.panels import (
    render_analog_narrative,
    render_backtest_card,
    render_context_strip,
    render_levers,
    render_long_term_tab,
    render_payoff_matrix,
    render_provenance_drawer,
    render_recommendation,
    render_save_load_export_bar,
    render_sensitivity_tornado,
)
from climate_twin.whatif.ui.state import get_state, update_state

# ── Header ──
st.markdown(
    f"<h2 style='margin-bottom:0;'>What If — Scenario Engine · "
    f"<span style='font-size:0.6em;color:#aaa;'>{_ctxcopy.BRAND_HTML}</span></h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Scenario-driven climate reasoning. Historical analogs (Method 2) "
    "by default; delta perturbation (Method 1) is gated behind a "
    "physical-inconsistency caveat."
)

# ── State + tabs ──
state = get_state()
tabs = st.tabs(["Short Term", "Long Term"])

with tabs[0]:
    update_state(horizon="short_term")

    # Context strip runs before the engine call so a screenshot is
    # always dated / labelled even if the engine hasn't run yet.
    render_context_strip(state, None)
    st.markdown("---")

    left, right = st.columns([1, 3])

    with left:
        run_clicked = render_levers(state)

    # Auto-run when levers change (cached — identical levers never re-run).
    # Also run on explicit button press.
    should_run = run_clicked or state.last_run_id != state.cache_key()
    result: dict | None = None
    if should_run:
        try:
            result = cached_run_for_state(state)
            update_state(last_run_id=state.cache_key())
        except Exception as e:
            st.error(f"Scenario run failed: {type(e).__name__}: {e}")
            result = None

    with right:
        if not result:
            st.info(
                "Set levers on the left and click **Run scenario** to "
                "populate the payoff matrix, recommendation, analogs, "
                "and sensitivity tornado."
            )
        else:
            render_payoff_matrix(state, result)
            render_recommendation(state, result)
            render_analog_narrative(state, result)
            render_sensitivity_tornado(state, result)
            render_backtest_card(state, result)

    # Provenance + save/load span both columns
    st.markdown("---")
    render_provenance_drawer(state, result or {})
    render_save_load_export_bar(state, result or {})


with tabs[1]:
    update_state(horizon="long_term")
    render_context_strip(state, None)
    st.markdown("---")
    render_long_term_tab(state)


# ── Developer diagnostic view — hidden by default ────────────────────
if _dev_flag:
    st.divider()
    st.info(_banners.DEV_ONLY_HEADER)
    with st.expander("Engine self-check (Parts 1–5 diagnostic)", expanded=True):
        st.caption(
            "This is the Parts-1-through-5 self-check UI, preserved as a "
            "developer aid. Access it via the `?dev=1` query flag; the "
            "shipped page shows only the Part-6 UI above."
        )
        try:
            from climate_twin.whatif.ui._legacy_diagnostic import (
                _render_agriculture_diagnostic,
                _render_analogs_diagnostic,
                _render_decisions_diagnostic,
                _render_dry_spell_map,
                _render_et0_map,
                _render_l0_smoketest,
                _render_registry_table,
                _render_spi3_map,
            )
        except Exception as e:
            st.warning(f"Legacy diagnostic import failed — {type(e).__name__}: {e}")
        else:
            for name, fn in [
                ("L0 driver", _render_l0_smoketest),
                ("ET0 map", _render_et0_map),
                ("Dry-spell map", _render_dry_spell_map),
                ("SPI-3", _render_spi3_map),
                ("INDEX_REGISTRY", _render_registry_table),
                ("Agriculture", _render_agriculture_diagnostic),
                ("Decisions", _render_decisions_diagnostic),
                ("Analogs", _render_analogs_diagnostic),
            ]:
                st.markdown(f"#### {name}")
                try:
                    fn()
                except Exception as e:
                    st.error(f"❌ {name} failed — {type(e).__name__}: {e}")
                st.markdown("---")
