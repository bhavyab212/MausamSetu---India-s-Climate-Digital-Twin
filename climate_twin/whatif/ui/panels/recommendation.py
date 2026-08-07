"""whatif.ui.panels.recommendation — the "one thing an officer screenshots"."""
from __future__ import annotations

import numpy as np
import streamlit as st

from ...report.payoff_render import format_inr
from ..copy.recommendation import render_recommendation as _render_copy


def _pick_tier(result: dict) -> tuple[str, int, float]:
    """Return (tier, analog_year, distance)."""
    matches = result.get("analog_matches") or []
    if not matches:
        return "poor", 0, float("nan")
    top = matches[0]
    year = int(top.get("year", 0))
    dist = float(top.get("distance", float("nan")))
    tier = str(top.get("quality", "poor"))
    return tier, year, dist


def render_recommendation(state, result: dict) -> None:
    st.subheader("Recommendation")
    rec = result.get("recommendation")
    pm = result.get("payoff_matrix")
    if rec is None or pm is None:
        st.info("No recommendation yet — run a scenario.")
        return

    # Second-best decision for the "beats X" line
    ev_arr = np.asarray(
        result.get("expected_value") or [rec["EV_q50"]], dtype=float,
    )
    argmax = int(rec.get("argmax_EV", 0))
    beats_label = "climatology"
    beats_amount = rec.get("delta_vs_baseline_EV", 0.0)
    if ev_arr.size >= 2:
        order = np.argsort(ev_arr)[::-1]
        second_i = int(order[1]) if order[0] == argmax else int(order[0])
        beats_label = pm.decisions[second_i].label
        beats_amount = float(ev_arr[argmax] - ev_arr[second_i])

    delta_pct_val = 0.0
    baseline_ev = rec.get("baseline_EV", 0.0)
    if abs(baseline_ev) > 1e-9:
        delta_pct_val = (rec["delta_vs_baseline_EV"] / abs(baseline_ev)) * 100.0
    delta_pct = f"{delta_pct_val:+.0f}%"

    downside_pct = 0.0
    if beats_amount and rec.get("worst_case_q50"):
        wc = rec["worst_case_q50"]
        base_wc = wc - beats_amount
        if abs(base_wc) > 1e-9:
            downside_pct = (beats_amount / max(abs(base_wc), 1e-9)) * 100.0
    downside_reduction = f"{downside_pct:+.0f}%"

    tier, analog_year, analog_dist = _pick_tier(result)
    confidence_word = {"strong": "Strong", "fair": "Fair"}.get(tier, "Low")

    body = _render_copy(
        tier=tier,
        argmax_label=rec.get("argmax_EV_label", "?"),
        ev_p50=format_inr(rec["EV_q50"]),
        ev_p10=format_inr(rec["EV_q10"]),
        ev_p90=format_inr(rec["EV_q90"]),
        worst_case=format_inr(rec.get("worst_case_q50", float("nan"))),
        delta_vs_baseline=format_inr(rec.get("delta_vs_baseline_EV", 0.0)),
        delta_pct=delta_pct,
        beats_label=beats_label,
        beats_amount=format_inr(beats_amount),
        downside_reduction=downside_reduction,
        confidence_word=confidence_word,
        analog_year=analog_year or "—",
        analog_dist=f"{analog_dist:.2f}" if np.isfinite(analog_dist) else "—",
        analog_tier=tier,
    )
    st.markdown(body)

    if rec.get("delta_vs_baseline_EV", 0.0) > 0:
        st.success(
            f"Beats climatology by {format_inr(rec['delta_vs_baseline_EV'])} / ha "
            "in expectation."
        )
    else:
        st.warning(
            f"Does not beat climatology "
            f"({format_inr(rec['delta_vs_baseline_EV'])} / ha in expectation)."
        )
