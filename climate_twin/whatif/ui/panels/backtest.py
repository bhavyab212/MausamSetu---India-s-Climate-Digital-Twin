"""whatif.ui.panels.backtest — value-of-forecast card."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ..copy import banners


def render_backtest_card(state, result: dict) -> None:
    st.subheader("Backtest — value of forecast")
    bt = result.get("backtest")
    if bt is None:
        st.info(
            "No backtest for this rule/setup yet. See the Provenance "
            "drawer for the CLI command to generate one."
        )
        return

    v = float(getattr(bt, "V_forecast", 0.0))
    bs = float(getattr(bt, "brier", float("nan")))
    bss = float(getattr(bt, "brier_skill", float("nan")))
    setup_id = getattr(bt, "setup_id", "?")
    if v <= 0:
        st.error(
            f"{banners.BACKTEST_FAILED_TAG} · "
            f"V(forecast) = {v:+.2f} · BS = {bs:.3f} · BSS = {bss:+.2f} · "
            f"rule: {setup_id}"
        )
    else:
        st.success(
            f"V(forecast) = {v:+.2f} · BS = {bs:.3f} · BSS = {bss:+.2f} · "
            f"rule: {setup_id}"
        )

    rel = getattr(bt, "reliability", None)
    if rel is not None and not rel.empty:
        rel_clean = rel.dropna(subset=["mean_forecast", "observed_freq"])
        if not rel_clean.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#888", dash="dot"),
                name="perfect calibration",
            ))
            fig.add_trace(go.Scatter(
                x=rel_clean["mean_forecast"], y=rel_clean["observed_freq"],
                mode="markers+lines",
                marker=dict(size=8, color="#3b7ad9"),
                name="observed",
            ))
            fig.update_layout(
                title=dict(text="Reliability diagram", x=0.02, font=dict(size=13)),
                xaxis=dict(title="mean forecast prob", range=[0, 1]),
                yaxis=dict(title="observed frequency", range=[0, 1],
                              scaleanchor="x"),
                height=300, margin=dict(l=8, r=8, t=32, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-year audit table", expanded=False):
        per_year = getattr(bt, "per_year", None)
        if per_year is not None and not per_year.empty:
            st.dataframe(per_year, use_container_width=True, hide_index=True)
