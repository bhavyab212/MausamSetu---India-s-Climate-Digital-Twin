"""whatif.ui.panels.sensitivity — tornado plot + dominant-driver sentence."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ...report.payoff_render import format_inr


def render_sensitivity_tornado(state, result: dict) -> None:
    st.subheader("Sensitivity — what drives net revenue?")
    tor = result.get("tornado")
    if tor is None:
        st.info(
            "Tornado not computed for this run. It is regenerated on "
            "every scenario run — trigger one via the Run button."
        )
        return

    df = tor.rows.head(8).copy()
    labels = df["label"].tolist()
    ups = df["net_p50_up"].tolist()
    dns = df["net_p50_dn"].tolist()
    base = float(tor.net_p50_base)

    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(go.Bar(
            y=[label], x=[dns[i] - base], name="down",
            orientation="h", base=[base],
            marker_color="#e76f51", showlegend=(i == 0),
        ))
        fig.add_trace(go.Bar(
            y=[label], x=[ups[i] - base], name="up",
            orientation="h", base=[base],
            marker_color="#2a9d8f", showlegend=(i == 0),
        ))
    fig.add_vline(
        x=base, line=dict(color="#888", width=1, dash="dot"),
        annotation_text=f"base = {format_inr(base)}",
        annotation_position="top",
    )
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="net revenue ₹/ha (q50)"),
        yaxis=dict(autorange="reversed"),
        height=320, margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(tor.sentence())
