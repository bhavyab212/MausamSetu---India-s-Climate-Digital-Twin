"""
whatif.report.payoff_render — plotly heatmap of the PayoffMatrix.

Design notes:
    * Diverging colormap centred on the climatology baseline (**not**
      jet). Reds mark losses vs baseline, greens gains — the
      universally-understood polarity for money on a district
      officer's screen.
    * Cell text uses Indian-lakh grouping: ``₹1,23,400`` — not
      ``₹123,400``. Matches how mandi board numbers are actually
      written.
    * Regret overlay: cells whose regret is > 30 % of the column-max
      payoff carry a dashed border. Draws the eye to "opportunity
      cost" without visually screaming.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..economics.payoff import PayoffMatrix


def format_inr(x: float) -> str:
    """Indian-lakh grouping for a numeric value.

    Example: ``format_inr(123456.78) == '₹1,23,457'``.
    """
    if not np.isfinite(x):
        return "—"
    sign = "-" if x < 0 else ""
    n = abs(int(round(x)))
    s = str(n)
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3, rest = s[-3:], s[:-3]
    # Group `rest` in 2s from the right
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    grouped = ",".join(reversed(parts))
    return f"{sign}₹{grouped},{last3}"


def payoff_heatmap(
    pm: PayoffMatrix,
    *,
    show_regret_shading: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Return a plotly heatmap of ``pm.payoff_p50`` centred on baseline.

    Cells show the raw ₹/ha value; hover reveals q10/q50/q90 and the
    regret against the column-optimal decision.
    """
    p50 = pm.payoff_p50
    p10 = pm.payoff_p10
    p90 = pm.payoff_p90
    base = pm.baseline_payoff

    # Diverging scale centred on the median baseline
    zmid = float(np.nanmedian(base))
    vmin = float(np.nanpercentile(p50, 5))
    vmax = float(np.nanpercentile(p50, 95))
    if not np.isfinite(zmid): zmid = 0.0

    dec_labels = [d.label for d in pm.decisions]
    st_labels = [f"{s.label} · p={s.weight:.2f}" for s in pm.states]

    per_state_max = p50.max(axis=0, keepdims=True)
    regret = per_state_max - p50

    text = np.empty(p50.shape, dtype=object)
    for i in range(p50.shape[0]):
        for j in range(p50.shape[1]):
            text[i, j] = format_inr(float(p50[i, j]))

    hover = np.empty(p50.shape, dtype=object)
    for i in range(p50.shape[0]):
        for j in range(p50.shape[1]):
            hover[i, j] = (
                f"<b>{dec_labels[i]} × {st_labels[j]}</b><br>"
                f"q10:      {format_inr(p10[i,j])}<br>"
                f"q50:      {format_inr(p50[i,j])}<br>"
                f"q90:      {format_inr(p90[i,j])}<br>"
                f"baseline: {format_inr(base[i,j])}<br>"
                f"Δ vs base: {format_inr(p50[i,j] - base[i,j])}<br>"
                f"regret:   {format_inr(regret[i,j])}"
            )

    fig = go.Figure(go.Heatmap(
        z=p50, x=st_labels, y=dec_labels,
        colorscale="RdYlGn",             # loss(red) → gain(green)
        zmid=zmid, zmin=vmin, zmax=vmax,
        colorbar=dict(title="₹/ha (q50)", thickness=14),
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color="#111"),
        hoverinfo="text", hovertext=hover,
    ))
    fig.update_layout(
        title=dict(text=title or "Payoff matrix — Net revenue ₹/ha (q50)",
                   x=0.02, xanchor="left", font=dict(size=14)),
        xaxis=dict(title="Climate state", tickangle=0),
        yaxis=dict(title="Decision", automargin=True),
        margin=dict(l=8, r=8, t=48, b=8),
        height=90 + 60 * len(dec_labels),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    if show_regret_shading and pm.n_dec > 1:
        # Draw a dashed rectangle on high-regret cells
        col_ranges = per_state_max - p50.min(axis=0, keepdims=True) + 1e-9
        for i in range(p50.shape[0]):
            for j in range(p50.shape[1]):
                if regret[i, j] > 0.3 * float(col_ranges[0, j]):
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=j - 0.48, x1=j + 0.48,
                        y0=i - 0.48, y1=i + 0.48,
                        line=dict(color="#7a1e00", width=1.2, dash="dot"),
                        fillcolor="rgba(0,0,0,0)",
                    )
    return fig
