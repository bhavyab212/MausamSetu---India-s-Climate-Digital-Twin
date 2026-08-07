"""
whatif.tests.test_ui — Part 6 UI smoke + guard tests.

Fast, hermetic. AppTest driving pages/30_What_If.py is slow (loads all
of the whatif package + the diagnostic legacy module); we keep the
AppTest usage minimal and rely on direct-import assertions for the
copy library, colormap guard, and reproducibility rules.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from climate_twin.whatif.economics import (
    ClimateState,
    Decision,
    EconomicOutcome,
    PayoffMatrix,
)
from climate_twin.whatif.report.payoff_render import format_inr, payoff_heatmap
from climate_twin.whatif.ui.copy import (
    analogs as copy_analogs,
    banners as copy_banners,
    context as copy_context,
    long_term as copy_long_term,
    recommendation as copy_recommendation,
)
from climate_twin.whatif.ui.state import (
    WhatIfState,
    _SESSION_KEY,
    get_state,
    update_state,
)


# ─── STEP 14.1 — Copy ban: long_term must not say predict / forecast ─
def test_long_term_copy_bans_forbidden_verbs():
    """Part 7 Rule 1 (§9c): none of ``predict`` / ``forecast`` / ``will``
    / ``is going to`` may appear in long_term copy. The regex uses word
    boundaries so ``William`` / ``willing`` wouldn't false-positive
    (there aren't any today anyway)."""
    import re
    banned_patterns = [
        r"\bpredict",           # covers predict, predicts, prediction
        r"\bforecast",          # covers forecast, forecasts, forecasting
        r"\bwill\b",            # bare will as a verb
        r"\bis going to\b",     # future-tense construction
    ]
    for name in dir(copy_long_term):
        if name.startswith("_"):
            continue
        obj = getattr(copy_long_term, name)
        if isinstance(obj, str):
            values = [obj]
        elif isinstance(obj, tuple):
            values = [s for s in obj if isinstance(s, str)]
        else:
            continue
        for i, s in enumerate(values):
            for pat in banned_patterns:
                assert not re.search(pat, s, flags=re.IGNORECASE), (
                    f"long_term copy {name}[{i}] contains banned "
                    f"pattern {pat!r}: {s!r}"
                )


def test_context_copy_has_devanagari_branding():
    """Devanagari branding stays consistent (Rule 15)."""
    assert "मौसम सेतु" in copy_context.BRAND_HTML


def test_banners_have_required_permanent_strings():
    """The permanent banners named in the spec must exist and be non-empty."""
    for name in (
        "PERTURBATION_CAVEAT",
        "CAVEAT_ACKNOWLEDGE_CHECKBOX",
        "LONG_TERM_SCENARIO_BANNER",
        "NO_STRONG_ANALOG_HEADLINE",
        "BACKTEST_FAILED_TAG",
        "EXPORT_BLOCKED_DIRTY",
        "EXPORT_BLOCKED_NO_CAVEAT",
        "RESOLUTION_CEILING_NOTE",
    ):
        assert hasattr(copy_banners, name), f"missing banner {name}"
        assert getattr(copy_banners, name), f"banner {name} is empty"


# ─── STEP 14.2 — Colormap guard: payoff heatmap avoids jet/rainbow ────
def _fake_payoff(N: int = 3, M: int = 3) -> PayoffMatrix:
    """Small hand-built PayoffMatrix bypassing the sector runner."""
    decisions = [Decision(label=f"D{i}", kind="crop") for i in range(N)]
    states = [
        ClimateState(label=f"S{j}", weight=1.0 / M) for j in range(M)
    ]
    cells = [[None] * M for _ in range(N)]
    p50 = np.array([
        [10000.0, 12000.0, 14000.0],
        [15000.0, 16000.0, 17000.0],
        [8000.0, 9000.0, 10000.0],
    ])
    for i in range(N):
        for j in range(M):
            v = float(p50[i, j])
            cells[i][j] = EconomicOutcome(
                crop="test", season="2024-25",
                region_kind="district", region_id="test",
                gross_revenue_inr_per_ha={"q10": v, "q50": v, "q90": v},
                cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
                net_revenue_inr_per_ha={"q10": v, "q50": v, "q90": v},
                baseline_net_inr_per_ha={"q10": 10000.0, "q50": 10000.0, "q90": 10000.0},
                delta_vs_baseline={},
                price_source="msp",
            )
    return PayoffMatrix(
        decisions=decisions, states=states, cells=cells,
        payoff_p10=p50.copy(), payoff_p50=p50.copy(), payoff_p90=p50.copy(),
        baseline_payoff=np.full_like(p50, 10000.0),
        region_kind="test", region_id="",
    )


def test_payoff_heatmap_avoids_jet_and_rainbow():
    """Rule 4: no jet. The heatmap's colorscale must not be jet /
    rainbow / hsv."""
    fig = payoff_heatmap(_fake_payoff())
    blob = fig.to_json()
    blocklist = ("jet", "rainbow", "hsv", "gnuplot")
    lower = blob.lower()
    for bad in blocklist:
        assert f'"{bad}"' not in lower, (
            f"payoff heatmap uses forbidden colorscale {bad!r}"
        )


# ─── STEP 14.3 — format_inr formatting ────────────────────────────────
def test_format_inr_indian_lakh_grouping():
    """Indian lakh grouping: ₹1,23,400 not ₹123,400."""
    assert format_inr(123456.78) == "₹1,23,457"
    assert format_inr(1000000) == "₹10,00,000"
    assert format_inr(0) == "₹0"
    assert format_inr(-45000) == "-₹45,000"
    assert format_inr(float("nan")) == "—"


# ─── STEP 14.4 — State reproducibility ────────────────────────────────
def test_state_cache_key_is_deterministic():
    """cache_key() depends only on lever fields — not on UI prefs."""
    s1 = WhatIfState()
    s2 = WhatIfState()
    assert s1.cache_key() == s2.cache_key()
    # UI-only preference change does NOT invalidate cache
    s2.show_baseline_overlay = not s1.show_baseline_overlay
    assert s1.cache_key() == s2.cache_key(), (
        "UI-only pref changed cache key — this would force needless reruns"
    )
    # Lever change DOES invalidate
    s3 = WhatIfState(crops_selected=("paddy_kharif",))
    assert s3.cache_key() != s1.cache_key()


def test_state_from_dict_roundtrip():
    """WhatIfState → dict → WhatIfState is byte-identical."""
    s = WhatIfState(
        region_id="cauvery", region_kind="subbasin",
        method="perturbation", method_caveat_ack=True,
        crops_selected=("wheat_rabi",),
        perturbation_rain_scale=0.85,
    )
    d = s.to_dict()
    s2 = WhatIfState.from_dict(d)
    assert s.cache_key() == s2.cache_key()
    assert s.to_dict() == s2.to_dict()


def test_state_from_dict_preserves_tuples():
    """crops_selected + sow_dates come back as tuples, not lists.

    Widgets bind tuples; a list would break equality checks in
    update_state and force cache invalidation on every load."""
    d = {"crops_selected": ["paddy_kharif", "bajra_kharif"],
         "sow_dates": ["2020-06-15"]}
    s = WhatIfState.from_dict(d)
    assert isinstance(s.crops_selected, tuple)
    assert isinstance(s.sow_dates, tuple)


# ─── STEP 14.5 — Recommendation templates ─────────────────────────────
def test_recommendation_copy_variants_render():
    """All three variants render without KeyError; strong/fair share
    the template body; no-strong branch is distinct."""
    slots = dict(
        argmax_label="bajra_kharif sow 2020-06-15",
        ev_p50="₹28,400", ev_p10="₹21,100", ev_p90="₹31,900",
        worst_case="₹18,700",
        delta_vs_baseline="+₹6,900", delta_pct="+24%",
        beats_label="paddy_kharif sow 2020-06-15",
        beats_amount="₹4,200", downside_reduction="+92%",
        confidence_word="Fair",
        analog_year=2002, analog_dist="8.4", analog_tier="fair",
    )
    strong = copy_recommendation.render_recommendation(tier="strong", **slots)
    fair = copy_recommendation.render_recommendation(tier="fair", **slots)
    poor = copy_recommendation.render_recommendation(tier="poor", **slots)
    assert "bajra_kharif" in strong
    assert "bajra_kharif" in fair
    assert "bajra_kharif" in poor
    # Strong vs fair templates are identical body; both are distinct from poor
    assert strong == fair
    assert "No strong or fair" in poor
    assert "closest analog" in strong.lower() or "closest match" in strong.lower()


def test_analog_card_template_slots():
    """The analog card template accepts the expected slots."""
    s = copy_analogs.CARD_TEMPLATE.format(
        year=2002, ya=2.1, pct_of_ymax=54,
        spi=-1.4, tmax=1.1, onset=8,
    )
    assert "2002" in s and "2.10" in s


# ─── STEP 14.6 — Perturbation gate contract ───────────────────────────
def test_perturbation_gate_state_wiring():
    """Turning on ``method='perturbation'`` should clear caveat_ack;
    tick it → gate opens. This is the state invariant that the levers
    panel binds to the checkbox."""
    import streamlit as st
    st.session_state.clear()
    s = get_state()
    assert s.method == "analog"
    assert s.method_caveat_ack is False
    update_state(method="perturbation")
    assert get_state().method == "perturbation"
    assert get_state().method_caveat_ack is False        # cleared
    update_state(method_caveat_ack=True)
    assert get_state().method_caveat_ack is True
    st.session_state.clear()


def test_reset_state_starts_clean():
    """reset_state() wipes every field back to defaults."""
    import streamlit as st
    from climate_twin.whatif.ui.state import reset_state
    st.session_state.clear()
    update_state(region_id="cauvery", crops_selected=("wheat_rabi",))
    assert get_state().region_id == "cauvery"
    reset_state()
    assert get_state().region_id == "vidarbha"
    assert get_state().crops_selected == (
        "paddy_kharif", "bajra_kharif", "arhar_kharif",
    )
    st.session_state.clear()


# ─── STEP 14.7 — AppTest cold start ────────────────────────────────────
@pytest.mark.slow
def test_page_cold_start_no_exception():
    """The page loads without raising. AppTest ships with Streamlit; if
    it's unavailable in this environment, we skip. Slow because it walks
    every import on the page module."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("Streamlit AppTest not available")
    at = AppTest.from_file(
        "climate_twin/pages/30_What_If.py", default_timeout=45,
    )
    at.run()
    # The page must at least render two tabs.
    labels = [t.label for t in at.tabs]
    assert labels == ["Short Term", "Long Term"], (
        f"expected two tabs Short Term / Long Term, got {labels}"
    )


# ─── STEP 14.8 — Save/load YAML roundtrip ──────────────────────────────
def test_save_load_yaml_roundtrip(tmp_path):
    """A WhatIfState written to YAML and read back produces an
    identical cache_key — this is Rule 8, reproducibility from screen."""
    import yaml
    s = WhatIfState(
        region_id="rayalaseema", region_kind="bbox",
        method="analog", analog_target_year=2015,
        analog_weighting="softmax", crops_selected=("paddy_kharif",),
    )
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(s.to_dict()), encoding="utf-8")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    s2 = WhatIfState.from_dict(d)
    assert s.cache_key() == s2.cache_key()
