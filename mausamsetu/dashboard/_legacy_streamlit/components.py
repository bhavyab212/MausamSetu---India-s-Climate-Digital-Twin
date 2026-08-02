"""
components.py
==============
Reusable UI widgets for the Command Center dashboard.

Each function renders one panel (tile / chart / gauge) with a consistent
look-and-feel matching the mockup:
  - Rounded borders with subtle orange border
  - Panel header (small caps title + optional ⓘ tooltip)
  - Big value + small sparkline + status badge
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================================
# COLORS
# ============================================================================
PANEL_BG      = "#0E1522"
PANEL_BORDER  = "#1E2A3E"
ORANGE        = "#F4A34A"
CYAN          = "#22D3EE"
GREEN         = "#22C55E"
RED           = "#EF4444"
YELLOW        = "#F59E0B"
BLUE          = "#3B82F6"
PURPLE        = "#A855F7"
TEXT_MAIN     = "#E5E9F0"
TEXT_DIM      = "#9CA3AF"


# ============================================================================
# GLOBAL CSS INJECTION
# ============================================================================
def inject_css():
    """Inject once at the top of the dashboard. Sets fonts, panel style, spacing."""
    st.markdown(f"""
    <style>
      /* --- HIDE STREAMLIT CHROME --- */
      #MainMenu, footer, header {{ visibility: hidden; }}
      .stApp {{ background: linear-gradient(180deg, #050912 0%, #030812 100%); }}
      .block-container {{ padding-top: 1rem; padding-bottom: 0.5rem; max-width: 100% !important; }}
      section[data-testid="stSidebar"] {{ display: none; }}
      div[data-testid="stToolbar"] {{ display: none; }}

      /* --- PANEL STYLE --- */
      .panel {{
        background: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        padding: 14px 16px 12px;
        margin-bottom: 10px;
        box-shadow: 0 0 20px rgba(244, 163, 74, 0.03);
      }}
      .panel:hover {{ border-color: rgba(244, 163, 74, 0.35); }}

      .p-title {{
        color: {TEXT_DIM};
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.6px;
        text-transform: none;
        margin: 0 0 6px 0;
        display: flex; align-items: center; gap: 4px;
      }}

      .p-value {{
        color: {TEXT_MAIN};
        font-size: 36px;
        font-weight: 300;
        line-height: 1;
        letter-spacing: -1px;
        display: inline-block;
      }}
      .p-value-sm {{ font-size: 26px; }}
      .p-unit {{ color: {TEXT_DIM}; font-size: 12px; margin-left: 6px; }}

      .badge {{
        display: inline-block;
        padding: 2px 10px;
        font-size: 10px;
        font-weight: 600;
        border-radius: 4px;
        letter-spacing: 0.5px;
        margin-left: 6px;
        vertical-align: 4px;
      }}
      .badge-green  {{ background: rgba(34,197,94,0.18);  color: {GREEN};  }}
      .badge-orange {{ background: rgba(244,163,74,0.18); color: {ORANGE}; }}
      .badge-red    {{ background: rgba(239,68,68,0.18);   color: {RED};    }}
      .badge-cyan   {{ background: rgba(34,211,238,0.18);  color: {CYAN};   }}
      .badge-blue   {{ background: rgba(59,130,246,0.18);  color: {BLUE};   }}

      /* --- TOP BAR --- */
      .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 12px 20px;
        background: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        margin-bottom: 10px;
      }}
      .brand {{
        display: flex; align-items: center; gap: 12px;
      }}
      .brand-logo {{
        width: 40px; height: 40px;
        background: radial-gradient(circle at 30% 30%, {ORANGE}, #a05a20);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 20px;
      }}
      .brand-name {{ color: {TEXT_MAIN}; font-size: 20px; font-weight: 600; }}
      .brand-sub  {{ color: {TEXT_DIM}; font-size: 10px; letter-spacing: 1px; }}

      .title-block {{ flex: 1; margin-left: 24px; }}
      .title-main  {{ color: {TEXT_MAIN}; font-size: 22px; font-weight: 400; margin: 0; }}
      .title-sub   {{ color: {TEXT_DIM};  font-size: 12px; margin: 2px 0 0; }}
      .title-tag   {{ color: {ORANGE};    font-size: 11px; letter-spacing: 1px; margin-top: 2px; }}

      .nav {{ display: flex; gap: 8px; }}
      .nav-pill {{
        padding: 6px 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid transparent;
        border-radius: 20px;
        color: {TEXT_DIM};
        font-size: 11px;
        letter-spacing: 0.5px;
        text-align: center;
        cursor: pointer;
      }}
      .nav-pill.active {{
        border-color: {ORANGE};
        color: {ORANGE};
        background: rgba(244,163,74,0.08);
      }}

      .clock {{
        text-align: right;
        color: {TEXT_MAIN};
        font-size: 20px;
        font-weight: 300;
      }}
      .clock-date {{ font-size: 11px; color: {TEXT_DIM}; margin-top: 2px; }}

      /* --- FOOTER STRIP --- */
      .footer-strip {{
        display: flex; align-items: center; justify-content: space-between;
        background: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        padding: 12px 20px;
        margin-top: 10px;
      }}
      .stage {{
        display: flex; align-items: center; gap: 8px;
      }}
      .stage-icon {{
        width: 32px; height: 32px;
        border-radius: 50%;
        border: 1px solid {PANEL_BORDER};
        display: flex; align-items: center; justify-content: center;
        color: {TEXT_DIM};
        font-size: 14px;
      }}
      .stage.active .stage-icon {{ border-color: {ORANGE}; color: {ORANGE}; }}
      .stage-label {{ color: {TEXT_DIM}; font-size: 11px; }}
      .stage-time  {{ color: {TEXT_MAIN}; font-size: 12px; }}
      .stage.active .stage-time {{ color: {ORANGE}; }}

      /* --- STATUS LIGHT ROW --- */
      .status-row {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 11px; }}
      .status-row .name {{ color: {TEXT_DIM}; }}

      /* --- Streamlit-provided element overrides --- */
      div[data-testid="stMarkdownContainer"] p {{ margin: 0; }}
      div.stPlotlyChart {{ margin-top: -8px; }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# TOP BAR
# ============================================================================
def render_topbar(active_tab: str = "Overview", clock: str = "", datestr: str = ""):
    tabs = ["Overview", "Map View", "Forecast", "Scenarios", "Alerts", "Reports", "Settings"]
    icons = {"Overview":"◈","Map View":"◇","Forecast":"◊","Scenarios":"◐","Alerts":"◔","Reports":"▤","Settings":"⚙"}
    pills_html = "".join(
        f'<div class="nav-pill{" active" if t == active_tab else ""}">{icons.get(t,"·")}<br><span style="font-size:9px">{t}</span></div>'
        for t in tabs
    )
    st.markdown(f"""
    <div class="topbar">
      <div class="brand">
        <div class="brand-logo">☔</div>
        <div>
          <div class="brand-name">MausamSetu</div>
          <div class="brand-sub">मौसम सेतु</div>
        </div>
      </div>
      <div class="title-block">
        <div class="title-main">Command Center</div>
        <div class="title-sub">AI-Powered Digital Twin of India's Climate</div>
        <div class="title-tag">Cauvery River Basin  •  Monsoon Monitoring  •  Real-time Insights</div>
      </div>
      <div class="nav">{pills_html}</div>
      <div class="clock">
        <div>{clock}</div>
        <div class="clock-date">{datestr}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# METRIC TILE (big number + optional sparkline + badge)
# ============================================================================
def metric_tile(
    title: str,
    value: str,
    unit: str = "",
    badge: str | None = None,
    badge_color: str = "green",
    sparkline: Sequence[float] | None = None,
    sparkline_color: str = CYAN,
    height: int = 40,
    tooltip: str | None = None,
):
    tip = f' <span title="{tooltip}" style="color:{TEXT_DIM};font-size:10px">ⓘ</span>' if tooltip else ""
    badge_html = f'<span class="badge badge-{badge_color}">{badge}</span>' if badge else ""

    st.markdown(f"""
    <div class="panel">
      <div class="p-title">{title}{tip}<span style="flex:1"></span>
        <span style="color:{TEXT_DIM};font-size:10px">{unit}</span>
      </div>
      <div>
        <span class="p-value">{value}</span>
        {badge_html}
      </div>
    """, unsafe_allow_html=True)

    if sparkline is not None and len(sparkline) > 1:
        # Lightweight inline SVG sparkline (replaces a full Plotly figure per tile).
        # Visually identical: filled area + colored line, no axes, transparent bg.
        arr = np.asarray(sparkline, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        n = len(arr)
        vmin, vmax = float(arr.min()), float(arr.max())
        vrange = (vmax - vmin) or 1.0
        vb_w, vb_h, pad = 100.0, float(height), 2.0
        xs = [i / (n - 1) * vb_w for i in range(n)]
        ys = [vb_h - pad - (v - vmin) / vrange * (vb_h - 2 * pad) for v in arr]
        line_pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        fill_pts = f"0,{vb_h:.2f} " + line_pts + f" {vb_w:.2f},{vb_h:.2f}"
        st.markdown(
            f'<svg viewBox="0 0 {vb_w:.0f} {vb_h:.0f}" preserveAspectRatio="none" '
            f'style="width:100%;height:{height}px;display:block">'
            f'<polygon points="{fill_pts}" fill="rgba(34,211,238,0.08)" stroke="none"/>'
            f'<polyline points="{line_pts}" fill="none" stroke="{sparkline_color}" '
            f'stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
            f'</svg>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# ASSIMILATION CORRECTION CHART (3-line)
# ============================================================================
def assimilation_chart(
    x_labels: Sequence[str],
    observed: Sequence[float],
    model: Sequence[float],
    corrected: Sequence[float],
    now_index: int = None,
    unit: str = "mm/day",
    height: int = 200,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=observed, mode="lines+markers",
                             name="Observed", line=dict(color=ORANGE, width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x_labels, y=model, mode="lines",
                             name="Model", line=dict(color=RED, dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=x_labels, y=corrected, mode="lines+markers",
                             name="Corrected (AI)", line=dict(color=CYAN, width=2),
                             marker=dict(size=4)))
    if now_index is not None and 0 <= now_index < len(x_labels):
        # Filter out None / NaN so max() works
        clean = [v for v in observed if v is not None and not (isinstance(v, float) and np.isnan(v))]
        y_max = max(clean) if clean else 0
        fig.add_vline(x=x_labels[now_index], line=dict(color=TEXT_DIM, dash="dot", width=1))
        fig.add_annotation(x=x_labels[now_index], y=y_max,
                           text="Now", showarrow=False, yshift=10, font=dict(color=TEXT_DIM, size=9))
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=10, t=10, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=9)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=9), title=unit, title_font=dict(size=9)),
        legend=dict(orientation="h", yanchor="top", y=1.15, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )
    return fig


# ============================================================================
# 3D SURFACE (rainfall surface)
# ============================================================================
def rainfall_surface_3d(
    z: np.ndarray,          # 2D array (lat, lon)
    lat: np.ndarray,
    lon: np.ndarray,
    height: int = 240,
):
    fig = go.Figure(data=[go.Surface(
        x=lon, y=lat, z=z,
        colorscale=[
            [0.00, "#1E3A5F"],   # deep blue
            [0.20, "#3B82F6"],
            [0.45, "#22D3EE"],
            [0.65, "#84CC16"],
            [0.85, "#FCD34D"],
            [1.00, "#F87171"],   # red for heavy
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="mm/day", font=dict(color=TEXT_DIM, size=10)),
            tickfont=dict(color=TEXT_DIM, size=9),
            len=0.9, thickness=10, x=1.05,
        ),
        lighting=dict(ambient=0.4, diffuse=0.8),
        contours_z=dict(show=True, usecolormap=True, project_z=True),
    )])
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title=dict(text="°E", font=dict(color=TEXT_DIM, size=9)),
                       tickfont=dict(color=TEXT_DIM, size=8),
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=dict(text="°N", font=dict(color=TEXT_DIM, size=9)),
                       tickfont=dict(color=TEXT_DIM, size=8),
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title="", tickfont=dict(color=TEXT_DIM, size=8),
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgba(255,255,255,0.05)"),
            camera=dict(eye=dict(x=1.5, y=-1.2, z=0.9)),
            aspectratio=dict(x=1, y=1, z=0.4),
        ),
    )
    return fig


# ============================================================================
# HORIZONTAL BAR (data sources widget)
# ============================================================================
def source_bars(sources: dict[str, tuple[float, str]]):
    """
    sources: {name: (percent, color)}
    Renders as horizontal 4-up bars.
    """
    cols = st.columns(len(sources))
    for (name, (pct, color)), col in zip(sources.items(), cols):
        with col:
            st.markdown(f"""
            <div style="text-align:left;">
              <div style="color:{TEXT_DIM};font-size:10px;letter-spacing:0.5px;">{name}</div>
              <div style="background:rgba(255,255,255,0.05);height:6px;border-radius:3px;margin-top:6px;">
                <div style="width:{pct}%;background:{color};height:6px;border-radius:3px;"></div>
              </div>
              <div style="color:{color};font-size:11px;margin-top:4px;">{pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# GAUGE (used for risk assessment)
# ============================================================================
def risk_gauge(value: float, label: str = "Risk level", height: int = 180):
    color = GREEN if value < 15 else YELLOW if value < 35 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"color": TEXT_MAIN, "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_DIM, "tickfont": {"color": TEXT_DIM, "size": 9}},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15],  "color": "rgba(34,197,94,0.15)"},
                {"range": [15, 35], "color": "rgba(245,158,11,0.15)"},
                {"range": [35, 100], "color": "rgba(239,68,68,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM),
    )
    return fig


# ============================================================================
# DONUT (assimilation progress)
# ============================================================================
def progress_donut(percent: float, label: str = "Progress", height: int = 180):
    fig = go.Figure(data=[go.Pie(
        values=[percent, 100 - percent],
        hole=0.75,
        marker=dict(colors=[GREEN, "rgba(255,255,255,0.05)"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
    )])
    fig.add_annotation(text=f"{percent:.0f}%", showarrow=False,
                       font=dict(size=28, color=TEXT_MAIN, family="monospace"),
                       y=0.55, x=0.5)
    fig.add_annotation(text=label, showarrow=False,
                       font=dict(size=10, color=TEXT_DIM),
                       y=0.35, x=0.5)
    fig.update_layout(
        height=height, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================================
# STATUS LIGHTS (data quality)
# ============================================================================
def status_lights(items: list[tuple[str, str, str]]):
    """items = [(name, status_text, color_key), ...]"""
    rows = ""
    for name, status, color in items:
        rows += f"""
        <div class="status-row">
          <span class="name">{name}</span>
          <span class="badge badge-{color}">{status}</span>
        </div>"""
    st.markdown(rows, unsafe_allow_html=True)


# ============================================================================
# FOOTER PIPELINE STRIP
# ============================================================================
def render_footer(stages: list[dict], next_cycle: str, flood_risk: str, flood_color: str):
    stages_html = ""
    for s in stages:
        active = " active" if s.get("active") else ""
        stages_html += f"""
        <div class="stage{active}">
          <div class="stage-icon">{s.get('icon','●')}</div>
          <div>
            <div class="stage-label">{s['name']}</div>
            <div class="stage-time">{s['time']}</div>
            {f"<div style='color:{ORANGE};font-size:9px'>ACTIVE</div>" if s.get('active') else ""}
          </div>
        </div>"""
    st.markdown(f"""
    <div class="footer-strip">
      <div>
        <div class="stage-label">Next Cycle</div>
        <div class="stage-time" style="font-size:22px;color:{ORANGE}">{next_cycle}</div>
      </div>
      {stages_html}
      <div style="text-align:right">
        <div class="stage-label">Flood Risk</div>
        <div style="color:{flood_color};font-size:20px;font-weight:300">〰 {flood_risk}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
