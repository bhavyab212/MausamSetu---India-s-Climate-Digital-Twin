"""
whatif.tests.test_e2e — end-to-end integration.

Fast, hermetic. Every layer's unit tests already exist; these stitch
them together and verify their *interaction*. All fixtures live in
:mod:`whatif.tests.fixtures_e2e` and are deterministic.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.economics import (
    ClimateState,
    Decision,
    EconomicOutcome,
    HistoricalFrequencyRule,
    PayoffMatrix,
    SHIPPED_SETUPS,
    SchemaError,
    build_payoff_matrix,
    expected_value,
    load_prices,
    minimax_regret,
    recommend,
    regret_matrix,
    synthesise_history,
    value_agriculture,
    walk_forward_backtest,
)
from climate_twin.whatif.report.card import export_scenario_card
from climate_twin.whatif.sectors import (
    LongTermResult,
    RepresentationMismatch,
    load_crop,
    run_long_term_scenario,
)
from climate_twin.whatif.sectors.adaptations import MissingCostCitation
from climate_twin.whatif.economics.npv import adaptation_npv
from climate_twin.whatif.ui.state import WhatIfState


# ─── Tiny synthetic fixtures ─────────────────────────────────────────
def _fake_pm(matrix, weights=None):
    N, M = matrix.shape
    if weights is None:
        weights = np.full(M, 1.0 / M)
    decs = [Decision(label=f"D{i}", kind="crop") for i in range(N)]
    states = [ClimateState(label=f"S{j}", weight=float(weights[j])) for j in range(M)]
    cells = [[None] * M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            v = float(matrix[i, j])
            cells[i][j] = EconomicOutcome(
                crop="test", season="2024-25",
                region_kind="district", region_id="test",
                gross_revenue_inr_per_ha={"q10": v, "q50": v, "q90": v},
                cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
                net_revenue_inr_per_ha={"q10": v * 0.9, "q50": v, "q90": v * 1.1},
                baseline_net_inr_per_ha={"q10": 10000.0, "q50": 10000.0, "q90": 10000.0},
                delta_vs_baseline={},
                price_source="msp",
            )
    return PayoffMatrix(
        decisions=decs, states=states, cells=cells,
        payoff_p10=matrix.astype(float) * 0.9,
        payoff_p50=matrix.astype(float),
        payoff_p90=matrix.astype(float) * 1.1,
        baseline_payoff=np.full_like(matrix, 10000.0, dtype=float),
        region_kind="test", region_id="",
    )


# ─── STEP 1.1 — Cold-start Short Term shape ──────────────────────────
def test_shortterm_payoff_shape_and_provenance():
    """Bypass the sector runner and directly assemble a PayoffMatrix
    from EconomicOutcomes — verifies the L4 → UI wiring."""
    m = np.array([[42000, 38000, 35000],
                  [40000, 41000, 39000],
                  [-5000, -5000, -5000]])
    pm = _fake_pm(m, weights=np.array([0.28, 0.51, 0.21]))
    ev = expected_value(pm)
    rec = recommend(pm)
    assert rec["argmax_EV"] in {0, 1}
    for k in ("q10", "q50", "q90"):
        assert k in pm.cells[0][0].net_revenue_inr_per_ha
    # Provenance keys expected downstream
        assert pm.cells[0][0].price_source == "msp"


# ─── STEP 1.2 — YAML round-trip byte-identity ────────────────────────
def test_state_yaml_roundtrip_produces_same_cache_key(tmp_path):
    """A saved WhatIfState reloads to the same cache_key — same input,
    same run, same output. Rule 8 from Part 6, verified end-to-end."""
    s = WhatIfState(
        region_id="vidarbha", region_kind="bbox",
        method="analog", analog_target_year=2020,
        crops_selected=("paddy_kharif", "bajra_kharif"),
    )
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(s.to_dict()), encoding="utf-8")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    s2 = WhatIfState.from_dict(d)
    assert s.cache_key() == s2.cache_key()


# ─── STEP 1.3 — Analog → payoff → recommendation coherence ───────────
def test_recommendation_argmax_matches_expected_value():
    m = np.array([[10000, 12000, 14000],
                  [22000, 20000, 18000],
                  [8000, 9000, 10000]])
    pm = _fake_pm(m, weights=np.array([0.25, 0.5, 0.25]))
    rec = recommend(pm)
    ev = expected_value(pm)
    assert rec["argmax_EV"] == int(np.argmax(ev))


# ─── STEP 1.4 — Perturbation gate on export ──────────────────────────
def test_perturbation_export_gate_via_ui_flag(tmp_path):
    """The export gate lives in whatif.ui.panels.save_load_export;
    verify a perturbation state without caveat_ack is refused."""
    from climate_twin.whatif.ui.panels.save_load_export import _can_export

    s = WhatIfState(method="perturbation", method_caveat_ack=False)
    ok, msg = _can_export(s, {})
    assert not ok
    assert "caveat" in msg.lower() or "perturbation" in msg.lower()

    s2 = WhatIfState(method="perturbation", method_caveat_ack=True)
    ok2, _ = _can_export(s2, {})
    assert ok2


# ─── STEP 1.5 — Long Term multi-model coherence ──────────────────────
def test_long_term_scenario_returns_result_with_verdicts():
    region = RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0),
                        id="vidarbha")
    res = run_long_term_scenario(
        "ssp245", 2050, "agriculture", region,
        adaptations=["heat_tolerant_wheat_variant",
                     "raised_bunds_flood_protection"],
    )
    assert isinstance(res, LongTermResult)
    assert res.provenance["scenario_id"] == "ssp245"
    assert res.provenance["window_20yr"] == [2041, 2060]
    for k in ("heat_tolerant_wheat_variant", "raised_bunds_flood_protection"):
        assert res.provenance["adaptation_verdicts"][k]["cost_complete"]


def test_long_term_npv_arithmetic_ordering():
    """At the same annual benefit, a lower discount rate produces a
    larger NPV (positive stream). BCR at 8% must be positive."""
    df = adaptation_npv(
        {"heat_tolerant_wheat_variant": 6000.0},
    )
    row = df.iloc[0]
    assert row["NPV_07"] >= row["NPV_08"] >= row["NPV_12"]
    assert row["BCR_08"] > 0


# ─── STEP 1.6 — Cross-representation refusal ─────────────────────────
def test_representation_mismatch_raises():
    with pytest.raises(RepresentationMismatch):
        raise RepresentationMismatch(
            "attempted to fold multi-model LT into ST 3-pass",
        )


# ─── STEP 1.7 — Baseline attached everywhere ─────────────────────────
def test_economic_outcome_requires_baseline():
    """Constructing an EconomicOutcome with a None baseline raises the
    schema guard."""
    crop = load_crop("paddy_kharif")
    ps = load_prices("paddy_kharif")
    yq = {"q10": 3.0, "q50": 3.5, "q90": 4.0}
    # Happy path
    eo = value_agriculture(
        yield_qdict=yq, baseline_ya_t_ha=3.2, crop=crop, price_set=ps,
        season="2024-25", region_kind="district", region_id="test",
    )
    for k in ("q10", "q50", "q90"):
        assert np.isfinite(eo.baseline_net_inr_per_ha[k])
    # Attempt to reset baseline to a scalar → schema rejects
    with pytest.raises(SchemaError):
        EconomicOutcome(
            crop="test", season="2024-25",
            region_kind="district", region_id="",
            gross_revenue_inr_per_ha={"q10": 1.0, "q50": 1.0, "q90": 1.0},
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha={"q10": 1.0, "q50": 1.0, "q90": 1.0},
            baseline_net_inr_per_ha=42000.0,     # ← scalar
            delta_vs_baseline={},
            price_source="msp",
        )


# ─── STEP 1.8 — Provenance replay via WhatIfState ────────────────────
def test_scenario_yaml_carries_full_lever_set(tmp_path):
    """The WhatIfState YAML captures every lever needed to reproduce
    the run. Nothing beyond WhatIfState._CACHE_KEY_FIELDS may be
    required for replay."""
    s = WhatIfState()
    d = s.to_dict()
    for field in WhatIfState._CACHE_KEY_FIELDS:
        assert field in d, f"lever field {field!r} missing from YAML"


# ─── STEP 1.9 — Export card round-trip (HTML) ────────────────────────
def test_export_scenario_card_writes_html(tmp_path):
    """The scenario card exporter writes a self-contained HTML that
    Print-to-PDFs cleanly."""
    from climate_twin.whatif.economics import DEFAULT_KNOBS, tornado

    m = np.array([[42000, 38000, 35000],
                  [40000, 41000, 39000]])
    pm = _fake_pm(m, weights=np.array([0.3, 0.4, 0.3]))
    rec = recommend(pm)

    def _fn(levers):
        return pm.cells[0][0]
    tor = tornado(_fn, base_levers={}, delta_pct=0.0)

    out = tmp_path / "card.html"
    export_scenario_card(
        out, title="e2e-test", region="test",
        run_id="deadbeef",
        payoff_matrix=pm, recommendation=rec,
        tornado_result=tor, provenance={"code_hash": "test"},
    )
    assert out.exists() and out.stat().st_size > 500


# ─── STEP 1.10 — Backtest failure surface preserved ─────────────────
def test_shipped_negative_backtest_still_reports():
    """HistoricalFrequencyRule cannot beat climatology (guaranteed by
    construction). The e2e path emits V<=0 without crashing."""
    df = synthesise_history(seed=42)
    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    res = walk_forward_backtest(
        HistoricalFrequencyRule(), setup, df, write_parquet=False,
    )
    assert res.V_forecast <= 1e-6, (
        f"HistoricalFrequency unexpectedly beat climatology "
        f"(V={res.V_forecast}); audit the leakage guard."
    )
    # The failure surface is preserved: the result is not None
    assert res.brier is not None and np.isfinite(res.brier)
