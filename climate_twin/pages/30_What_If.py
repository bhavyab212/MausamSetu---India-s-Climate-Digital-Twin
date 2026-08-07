"""
pages/30_What_If.py — Scenario Engine entry point.

Streamlit's multipage router auto-registers this file as a top-level
page because it sits at ``pages/*.py`` (no subfolder). The numeric
prefix ``30_`` orders it AFTER Home (0) / Explorer (10) / What-If (30)
and BEFORE the training + validation pages that will migrate here in
subsequent parts.

Part 0 delivers ONLY the two-tab shell:
    * Short Term
    * Long Term
Both bodies are placeholders. The Part-1+ builds fill them in.

Do NOT bind sidebar controls, drivers, or heavy imports here yet —
they belong under ``climate_twin/whatif/`` and its Part-1+ subpackages.
"""
from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="What If — Scenario Engine · MausamSetu",
    page_icon="❓",
    layout="wide",
)

st.title("What If — Scenario Engine")
st.caption(
    "Scenario-driven climate reasoning: perturb history, resample analog "
    "years, or project SSP futures, then propagate through indices, "
    "biophysical models, sectors, and economics."
)

_tab_short, _tab_long = st.tabs(["Short Term", "Long Term"])

with _tab_short:
    st.info("Coming online in Part 6.")

with _tab_long:
    st.info("Coming online in Part 6.")
