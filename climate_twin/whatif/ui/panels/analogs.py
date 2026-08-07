"""whatif.ui.panels.analogs — historical-analog narrative panel."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..copy import analogs as _copy
from ..copy import banners as _b


_TIER_COLOR = {
    "strong": "#2a9d8f",
    "fair":   "#e9c46a",
    "poor":   "#e76f51",
}


def _card_html(m: dict) -> str:
    tier = m.get("quality", "poor")
    word = _copy.QUALITY_WORDS.get(tier, tier.title())
    color = _TIER_COLOR.get(tier, "#888")
    return (
        f"<div style='padding:10px 12px;border-radius:8px;"
        f"background:rgba(255,255,255,0.03);"
        f"border:1px solid rgba(255,255,255,0.12);height:100%;'>"
        f"<div style='font-size:1.8em;font-weight:600;'>{m['year']}</div>"
        f"<div style='margin-top:2px;'>d² = {m['distance']:.2f} · "
        f"<span style='background:{color};color:#111;padding:1px 6px;"
        f"border-radius:4px;font-size:0.85em;font-weight:600;'>{word}</span></div>"
        f"<div style='margin-top:6px;font-size:0.88em;color:#bbb;'>{m.get('summary', '')}</div>"
        f"</div>"
    )


def render_analog_narrative(state, result: dict) -> None:
    st.subheader("Historical analogs")
    matches = result.get("analog_matches") or []
    if not matches:
        st.info(
            "The active scenario uses the perturbation flavour or hasn't "
            "resolved analogs yet — switch to Method 2 in the levers."
        )
        return

    strong_or_fair = [m for m in matches if m.get("quality") in ("strong", "fair")]
    if not strong_or_fair:
        st.warning(f"⚠️ {_b.NO_STRONG_ANALOG_HEADLINE}")
        st.caption(_copy.NO_STRONG_ANALOG_BODY)
        top3 = matches[:3]
        rows = ", ".join(
            f"{m['year']} (d²={m['distance']:.2f}, poor)" for m in top3
        )
        st.markdown(f"Closest matches: **{rows}**.")
        return

    st.markdown(_copy.HEADLINE_TEMPLATE.format(n=len(matches[:5])))
    top5 = matches[:5]
    cols = st.columns(len(top5))
    for c, m in zip(cols, top5):
        with c:
            st.markdown(_card_html(m), unsafe_allow_html=True)

    # Outcome strip plot
    st.markdown("---")
    st.markdown("**Outcome distribution — observed Ya per analog year**")
    ya = [m.get("ya_observed", float("nan")) for m in matches]
    labels = [str(m["year"]) for m in matches]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=ya, marker_color="#3b7ad9", name="observed Ya",
    ))
    q10 = result.get("analog_outcome_q10")
    q50 = result.get("analog_outcome_q50")
    q90 = result.get("analog_outcome_q90")
    baseline = result.get("analog_outcome_baseline")
    if q10 is not None and q90 is not None:
        fig.add_hrect(y0=float(q10), y1=float(q90),
                      fillcolor="rgba(58, 166, 245, 0.14)",
                      line_width=0, annotation_text="weighted q10–q90",
                      annotation_position="top left")
    if q50 is not None:
        fig.add_hline(y=float(q50), line=dict(color="#3b7ad9", width=2,
                                                dash="dash"),
                       annotation_text=f"q50 = {float(q50):.2f} t/ha",
                       annotation_position="top right")
    if baseline is not None:
        fig.add_hline(y=float(baseline),
                       line=dict(color="#888", width=1, dash="dot"),
                       annotation_text=f"climatology = {float(baseline):.2f} t/ha",
                       annotation_position="bottom right")
    fig.update_layout(
        xaxis=dict(title="analog year"),
        yaxis=dict(title="Ya (t/ha)"),
        height=280, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
