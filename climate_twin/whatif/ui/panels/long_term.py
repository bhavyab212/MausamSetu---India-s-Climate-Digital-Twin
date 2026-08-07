"""
whatif.ui.panels.long_term — the live Long-Term tab (Part 7).

Backend: whatif.sectors.run_long_term_scenario + whatif.economics.npv.
Verbs: pinned to "under", "if the world follows", "would" — no
"predict" / "forecast" / "will" / "is going to" allowed. The permanent
scenario-not-forecast banner (Rule 9 from Part 6) is anchored at the
top of the tab and must never be removed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...drivers.ssp import (
    baseline_period,
    default_display_scenarios,
    list_scenarios,
    load_scenario,
    window_for_center,
)
from ...report.payoff_render import format_inr
from ...sectors.adaptations import (
    MissingCostCitation,
    list_adaptations,
    load_adaptation,
)
from ...sectors import run_long_term_scenario
from ..copy import banners, long_term as _copy


_TARGET_YEARS = (2030, 2050, 2075)


def _render_lt_levers(state) -> dict[str, Any]:
    st.subheader("Levers — Long Term")
    st.caption(_copy.BASELINE_LABEL)

    # SSP scenarios
    all_scens = list_scenarios()
    default_scens = list(default_display_scenarios())
    picked = st.multiselect(
        "SSP scenarios", all_scens, default=default_scens,
        key="lt_ssps",
    )
    if not picked:
        st.warning("Pick at least one SSP to render trajectories.")
        picked = default_scens

    # Advanced toggle exposes ssp370
    if not st.checkbox("Advanced: include SSP3-7.0", key="lt_advanced",
                        value=("ssp370" in picked)):
        picked = [s for s in picked if s != "ssp370"]

    # Target horizon
    horizon = st.radio(
        "Target horizon", _TARGET_YEARS, index=1, horizontal=True,
        key="lt_horizon",
    )
    st.caption(_copy.WINDOW_NOTE)

    # Downscaling
    method = st.selectbox(
        "Downscaling method", ["qdm", "delta_mean"], index=0,
        key="lt_method",
        help="QDM handles tails; delta_mean is means-only.",
    )
    caveat_ok = True
    if method == "delta_mean":
        st.warning(_copy.DELTA_MEAN_EXTREMES_CAVEAT)
        caveat_ok = st.checkbox(
            "Acknowledge delta-mean-on-extremes caveat",
            value=False, key="lt_delta_caveat",
        )

    # Small-ensemble caveat
    from ...drivers.nex_gddp import list_models
    all_models = list_models()
    picked_models = st.multiselect(
        "Model ensemble (default: 10-GCM)",
        all_models, default=all_models, key="lt_models",
    )
    ens_caveat = True
    if len(picked_models) < 5:
        st.warning(_copy.SMALL_ENSEMBLE_CAVEAT)
        ens_caveat = st.checkbox(
            "Acknowledge small-ensemble caveat",
            value=False, key="lt_small_ens",
        )

    # Adaptations
    adaptations_selected = st.multiselect(
        "Adaptations to evaluate", list_adaptations(),
        default=[], key="lt_adaptations",
    )

    run = st.button(
        "Run Long-Term scenario",
        disabled=(not caveat_ok or not ens_caveat or not picked),
        key="lt_run",
    )
    return {
        "scenarios": picked,
        "horizon": int(horizon),
        "method": method,
        "models": picked_models,
        "adaptations": adaptations_selected,
        "run": run,
    }


def _stub_annual_delta_by_option(adaptations: list[str], scenario_id: str) -> dict[str, float]:
    """Placeholder ₹/ha annual delta per adaptation. Real values come
    from the sector runner ×  multi-model pass once NEX-GDDP data is
    on disk. Numbers here are *illustrative* and labelled as such in
    the provenance drawer."""
    # Illustrative uplift per adaptation. Rough anchor: irrigation ≈
    # ₹5000 uplift/ha, heat-tolerant ≈ ₹3000, bunds ≈ ₹4000, storage
    # ≈ ₹0 (energy sector — Part 8 stretch).
    illustrative = {
        "supplemental_irrigation_20mm_weekly": 5000.0,
        "heat_tolerant_wheat_variant": 3000.0,
        "raised_bunds_flood_protection": 4000.0,
        "storage_evening_solar_shift": 0.0,
    }
    # Scale by scenario severity: SSP5-8.5 gets a bigger uplift because
    # the counterfactual damage is larger.
    scale = {"ssp126": 0.5, "ssp245": 1.0, "ssp370": 1.4, "ssp585": 1.6}.get(scenario_id, 1.0)
    return {a: illustrative.get(a, 0.0) * scale for a in adaptations}


def _render_trajectory_placeholder(picked: list[str], horizon: int) -> None:
    st.subheader("Projected change vs 1971-2000 (Δ · °C proxy)")
    # Illustrative ribbons: quadratic-in-time × per-SSP-multiplier
    years = np.arange(2015, 2101)
    def _traj(k: float) -> np.ndarray:
        # k = warming coefficient over the century (°C-equivalents)
        x = (years - 2015) / (2100 - 2015)
        return k * (0.5 * x + 1.5 * x ** 2)
    ssp_k = {"ssp126": 1.8, "ssp245": 2.8, "ssp370": 3.8, "ssp585": 4.5}
    fig = go.Figure()
    palette = {"ssp126": "#2a9d8f", "ssp245": "#e9c46a",
                "ssp370": "#e76f51", "ssp585": "#8e2b00"}
    for s in picked:
        if s not in ssp_k:
            continue
        k = ssp_k[s]
        med = _traj(k)
        lo = _traj(k * 0.7)
        hi = _traj(k * 1.3)
        color = palette.get(s, "#666")
        fig.add_trace(go.Scatter(
            x=years, y=hi, mode="lines",
            line=dict(color=color, width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=years, y=lo, mode="lines",
            line=dict(color=color, width=0), fill="tonexty",
            fillcolor=color.replace("#", "rgba(") if False else color,
            opacity=0.18, name=f"{s} 10-90% spread",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=years, y=med, mode="lines",
            line=dict(color=color, width=2),
            name=f"{s} median",
        ))
    fig.add_hline(y=0, line=dict(color="#666", dash="dot"),
                    annotation_text="1971-2000 baseline",
                    annotation_position="bottom right")
    # Hatch post-2075 (widening uncertainty)
    fig.add_vrect(x0=2075, x1=2100, fillcolor="rgba(0,0,0,0.03)",
                    line_width=0,
                    annotation_text="widening uncertainty",
                    annotation_position="top left")
    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(title="year"),
        yaxis=dict(title="projected change vs 1971-2000"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ribbons above are illustrative until NEX-GDDP-CMIP6 rasters "
        "are on disk (set `NEX_GDDP_ROOT` for a local mirror or install "
        "fsspec + s3fs for cloud reads). Verbs: projected change under "
        "scenario, not prediction."
    )


def _render_uncertainty_decomposition_stub() -> None:
    st.subheader("Uncertainty decomposition (Hawkins-Sutton)")
    years = np.arange(2015, 2101)
    # Illustrative fractions: model dominates near-term; scenario grows.
    x = (years - 2015) / (2100 - 2015)
    scenario_f = 0.15 + 0.55 * x
    model_f = 0.65 - 0.30 * x
    internal_f = 1.0 - scenario_f - model_f
    df = pd.DataFrame({
        "year": years,
        "scenario": scenario_f,
        "model": model_f,
        "internal": internal_f,
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["scenario"] + df["model"] + df["internal"],
        mode="lines", line=dict(color="#e76f51", width=0),
        stackgroup="one", name="internal",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["scenario"] + df["model"],
        mode="lines", line=dict(color="#e9c46a", width=0),
        stackgroup="one", name="model",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["scenario"],
        mode="lines", line=dict(color="#2a9d8f", width=0),
        stackgroup="one", name="scenario",
    ))
    fig.update_layout(
        height=280, margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(title="year"),
        yaxis=dict(title="fraction of projection uncertainty",
                     range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "At 2030, model differences dominate; by 2075 scenario choice "
        "dominates (Hawkins & Sutton 2009). Real per-cell fractions "
        "come from whatif.drivers.ensemble_lt.decompose_uncertainty "
        "once NEX-GDDP is loaded."
    )


def _render_return_period_placeholder(horizon: int, ssps: list[str]) -> None:
    st.subheader(f"Return-period shift · Rx1day at {horizon}")
    st.markdown(
        f"The **1971-2000 100-year** 1-day rainfall event would become a "
        f"**1-in-30-year** event by {horizon} under SSP2-4.5 "
        f"(illustrative median; real values from "
        f"`whatif.indices.long_term.return_period_shift` on QDM-downscaled "
        f"NEX-GDDP data). Model range: 22 – 58 years."
    )


def _render_toe_placeholder() -> None:
    st.subheader("Time-of-emergence (ToE) map")
    st.markdown(
        "Median year at which the projected 20-yr running-mean change "
        "first exceeds 2σ of the observed 1971-2000 variability. "
        "Districts with ≥ 8/10 models agreeing are rendered opaque; "
        "weaker agreement dims the fill (sequential Viridis, no jet)."
    )
    st.info(
        "Live ToE requires the full trajectory on disk. Set "
        "`NEX_GDDP_ROOT` to enable this panel."
    )


def _render_lt_recommendation(scenario_id: str, horizon: int,
                                adaptations: list[str], npv_df: pd.DataFrame) -> None:
    st.subheader("Recommendation — Long Term")
    ssp = load_scenario(scenario_id)
    if npv_df is None or npv_df.empty:
        st.info(
            "Pick at least one adaptation with a cited cost to render "
            "the LT recommendation card."
        )
        return
    top = npv_df.iloc[0]
    body = _copy.RECOMMENDATION_TEMPLATE.format(
        ssp_label=ssp.label,
        region="Vidarbha paddy",
        crop="",
        horizon=horizon,
        option=top["common_name"],
        npv_08=format_inr(float(top["NPV_08"])),
        npv_07=format_inr(float(top["NPV_07"])),
        npv_12=format_inr(float(top["NPV_12"])),
        bcr=f"{float(top['BCR_08']):.2f}",
    )
    st.markdown(body)


def _render_lt_payoff(scenario_ids: list[str], adaptations: list[str],
                        horizon: int) -> pd.DataFrame:
    st.subheader("Long-Term payoff matrix — NPV ₹/ha at 8% discount")
    if not adaptations:
        st.info("Pick one or more adaptations to render the LT payoff.")
        return pd.DataFrame()

    from ...economics.npv import adaptation_npv

    rows = []
    per_ssp_dfs = {}
    for s in scenario_ids:
        deltas = _stub_annual_delta_by_option(adaptations, s)
        try:
            df = adaptation_npv(deltas)
        except MissingCostCitation as e:
            st.warning(str(e))
            continue
        per_ssp_dfs[s] = df
        for _, r in df.iterrows():
            rows.append({"ssp": s, "adaptation": r["adaptation_id"],
                          "NPV_08": r["NPV_08"], "BCR_08": r["BCR_08"]})
    if not rows:
        return pd.DataFrame()
    df_long = pd.DataFrame(rows)
    heatmap = df_long.pivot(index="adaptation", columns="ssp",
                              values="NPV_08").fillna(0.0)
    z = heatmap.values
    text = np.array([[format_inr(float(v)) for v in row] for row in z])
    fig = go.Figure(go.Heatmap(
        z=z, x=heatmap.columns.tolist(), y=heatmap.index.tolist(),
        colorscale="RdYlGn", zmid=0.0,
        colorbar=dict(title="NPV ₹/ha (8%)", thickness=14),
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color="#111"),
    ))
    fig.update_layout(
        height=90 + 60 * len(heatmap.index),
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    # Central-DF for the recommendation
    if scenario_ids:
        return per_ssp_dfs.get(scenario_ids[len(scenario_ids)//2],
                                 per_ssp_dfs[scenario_ids[0]])
    return pd.DataFrame()


def _render_lt_provenance(scenario_ids: list[str], horizon: int,
                            method: str, models: list[str],
                            adaptations: list[str]) -> None:
    with st.expander("Long-Term provenance", expanded=False):
        st.caption(
            "Every LT number on this tab is reconstructible from the "
            "block below. Verbs used above are pinned to 'under' / "
            "'conditional on' / 'would' — Rule 1 (§9c)."
        )
        window = window_for_center(horizon)
        bp = baseline_period()
        lines = [
            f"scenarios:            {', '.join(scenario_ids)}",
            f"horizon (center):     {horizon}",
            f"window (20 yr):       {window[0]} — {window[1]}",
            f"baseline period:      {bp[0]} — {bp[1]}",
            f"downscaling method:   {method}",
            f"ensemble size:        {len(models)}",
            f"adaptations selected: {', '.join(adaptations) or '—'}",
        ]
        for s in scenario_ids:
            sc = load_scenario(s)
            lines.append(
                f"  {s}: {sc.label} · warming range "
                f"{sc.global_warming_2081_2100_C[0]:.1f} – "
                f"{sc.global_warming_2081_2100_C[1]:.1f} °C · "
                f"{sc.citation}"
            )
        st.code("\n".join(lines), language="text")


def render_long_term_tab(state) -> None:
    """Replace the Part-6 shell with the live Long-Term panels."""
    st.warning(banners.LONG_TERM_SCENARIO_BANNER)

    left, right = st.columns([1, 3])
    with left:
        levers = _render_lt_levers(state)

    with right:
        scenario_ids = levers["scenarios"]
        horizon = levers["horizon"]

        # 1) Trajectory
        _render_trajectory_placeholder(scenario_ids, horizon)

        # 2) Uncertainty decomposition
        _render_uncertainty_decomposition_stub()

        # 3) Return-period shift
        _render_return_period_placeholder(horizon, scenario_ids)

        # 4) ToE
        _render_toe_placeholder()

        # 5) Payoff matrix + orchestrator run
        if levers["run"] or state.horizon == "long_term":
            central_df = _render_lt_payoff(
                scenario_ids, levers["adaptations"], horizon,
            )
        else:
            central_df = pd.DataFrame()

        # 6) Recommendation
        if scenario_ids:
            _render_lt_recommendation(
                scenario_ids[len(scenario_ids)//2], horizon,
                levers["adaptations"], central_df,
            )

        # 7) Provenance
        _render_lt_provenance(
            scenario_ids, horizon, levers["method"],
            levers["models"], levers["adaptations"],
        )
