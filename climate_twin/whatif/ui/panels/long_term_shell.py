"""whatif.ui.panels.long_term_shell — Long Term tab skeleton.

Part 6 ships only the layout. Part 7 wires the NEX-GDDP-CMIP6 backend.
The permanent scenario-not-forecast banner (Rule 9) is set here and
must never be removed.
"""
from __future__ import annotations

import streamlit as st

from ..copy import banners, long_term


def render_long_term_tab(state) -> None:
    st.warning(banners.LONG_TERM_SCENARIO_BANNER)

    left, right = st.columns([1, 3])

    with left:
        st.subheader("Levers")
        st.selectbox(
            "SSP scenario", list(long_term.SSP_CHOICES),
            index=0, key="lt_ssp", disabled=True,
            help=long_term.DISABLED_HINT,
        )
        st.radio(
            "Horizon", list(long_term.HORIZON_LABELS),
            index=0, key="lt_horizon", disabled=True,
            horizontal=True,
        )
        st.info(long_term.SSP_PLACEHOLDER)

    with right:
        st.subheader("Conditional trajectory (Part 7)")
        st.markdown(
            f"<div style='padding:24px;border-radius:8px;"
            f"background:repeating-linear-gradient(45deg,"
            f"rgba(255,180,0,0.05) 0 10px,rgba(255,180,0,0.1) 10px 20px);"
            f"border:1px dashed rgba(255,180,0,0.35);"
            f"text-align:center;'>"
            f"<b>Payoff matrix under SSP scenario</b><br>"
            f"<span style='color:#aaa;'>{long_term.SSP_PLACEHOLDER}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div style='padding:16px;border-radius:8px;"
            f"background:repeating-linear-gradient(45deg,"
            f"rgba(58,166,245,0.05) 0 10px,rgba(58,166,245,0.1) 10px 20px);"
            f"border:1px dashed rgba(58,166,245,0.35);text-align:center;'>"
            f"<b>Recommendation (conditional)</b><br>"
            f"<span style='color:#aaa;'>"
            f"Verbs pinned to \"if the world follows SSP…\" — no "
            f"\"predict\" / \"forecast\" copy allowed here."
            f"</span></div>",
            unsafe_allow_html=True,
        )
