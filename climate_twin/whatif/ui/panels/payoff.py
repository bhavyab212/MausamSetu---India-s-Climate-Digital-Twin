"""whatif.ui.panels.payoff — the payoff matrix heatmap."""
from __future__ import annotations

import streamlit as st

from ...report.payoff_render import payoff_heatmap


def render_payoff_matrix(state, result: dict) -> None:
    st.subheader("Payoff matrix — net revenue ₹/ha (q50)")
    pm = result.get("payoff_matrix")
    if pm is None:
        st.warning("Payoff matrix unavailable — run a scenario first.")
        return
    st.plotly_chart(payoff_heatmap(pm), use_container_width=True)
    st.caption(
        "Rows: decisions. Columns: climate states with weights. "
        "Colour: diverging around the climatology baseline "
        "(blue = above baseline, red = below). Dashed borders mark "
        "cells with high regret against the column-best decision."
    )
