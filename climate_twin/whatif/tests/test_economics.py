"""
whatif.tests.test_economics — Part 4 unit tests.

Fast, hermetic, deterministic. Every test blocks the branch on failure.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_twin.whatif.economics import (
    BACKTEST_VERSION,
    COST_LOSS_VERSION,
    DEFAULT_KNOBS,
    Decision,
    ClimateState,
    CostLossSetup,
    EconomicOutcome,
    HistoricalFrequencyRule,
    OnsetAnomalyRule,
    PayoffMatrix,
    SHIPPED_SETUPS,
    SchemaError,
    brier_score,
    brier_skill_score,
    build_payoff_matrix,
    expected_value,
    load_prices,
    minimax_regret,
    prices_registry_sha256,
    prices_registry_version,
    recommend,
    regret_matrix,
    reliability_diagram,
    stochastic_dominance,
    synthesise_history,
    tornado,
    value_agriculture,
    value_curve,
    var_cvar,
    walk_forward_backtest,
    worst_case,
)
from climate_twin.whatif.sectors import ResolutionCeilingError, load_crop


# ─── STEP 10.1: schema guard ─────────────────────────────────────────
def test_no_bare_rupee_scalar_rejected():
    """Constructing EconomicOutcome with a scalar in net_revenue raises."""
    with pytest.raises(SchemaError):
        EconomicOutcome(
            crop="paddy_kharif",
            season="2024-25",
            region_kind="district",
            region_id="test",
            gross_revenue_inr_per_ha={"q10": 1.0, "q50": 1.0, "q90": 1.0},
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha=42000.0,        # ← bare scalar
            baseline_net_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            delta_vs_baseline={},
            price_source="msp",
        )


def test_missing_quantile_key_rejected():
    """A dict without q10/q50/q90 fails schema."""
    with pytest.raises(SchemaError):
        EconomicOutcome(
            crop="paddy_kharif", season="2024-25",
            region_kind="district", region_id="test",
            gross_revenue_inr_per_ha={"q50": 1.0},    # ← missing q10, q90
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            baseline_net_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            delta_vs_baseline={},
            price_source="msp",
        )


# ─── STEP 10.2: valuation identity ───────────────────────────────────
def test_valuation_identity_perfect_water_no_cost():
    """Ya == Ymax, cost=0, discounts=0 ⇒ net == Ymax·10·MSP exactly."""
    crop = load_crop("paddy_kharif")
    ps = load_prices("paddy_kharif")
    # Zero out the cost and the discounts for this identity test
    ps_zero = replace(
        ps,
        cost_of_cultivation_inr_per_ha=0.0,
        moisture_discount_pct=0.0,
        transport_marketing_pct=0.0,
    )
    ya = float(crop.ymax_t_per_ha)
    yq = {"q10": ya, "q50": ya, "q90": ya}
    eo = value_agriculture(
        yield_qdict=yq,
        baseline_ya_t_ha=ya,
        crop=crop,
        price_set=ps_zero,
        season="2024-25",
        region_kind="district", region_id="test",
    )
    msp = ps_zero.msp_for("2024-25")
    expected = ya * 10.0 * msp
    for k in ("q10", "q50", "q90"):
        assert eo.net_revenue_inr_per_ha[k] == pytest.approx(expected, abs=1e-6)


# ─── STEP 10.3: baseline / delta invariance ──────────────────────────
def test_delta_equals_net_minus_baseline():
    """EconomicOutcome.delta_vs_baseline is always net - baseline_net."""
    crop = load_crop("wheat_rabi")
    ps = load_prices("wheat_rabi")
    yq = {"q10": 2.0, "q50": 3.0, "q90": 4.0}
    eo = value_agriculture(
        yield_qdict=yq, baseline_ya_t_ha=2.5, crop=crop, price_set=ps,
        season="2024-25", region_kind="district", region_id="test",
    )
    for k in ("q10", "q50", "q90"):
        expected = eo.net_revenue_inr_per_ha[k] - eo.baseline_net_inr_per_ha[k]
        assert eo.delta_vs_baseline[k] == pytest.approx(expected, abs=1e-6)


# ─── STEP 10.4: regret / minimax ─────────────────────────────────────
def _fake_pm(matrix: np.ndarray, weights: np.ndarray) -> PayoffMatrix:
    """Build a PayoffMatrix from a raw payoff array — bypassing the
    scenario runner. Baseline is set to zeros so delta = matrix."""
    N, M = matrix.shape
    dec = [Decision(label=f"D{i}", kind="crop") for i in range(N)]
    st = [ClimateState(label=f"S{j}", weight=float(weights[j])) for j in range(M)]
    cells = [[None] * M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            q = {"q10": float(matrix[i, j]), "q50": float(matrix[i, j]),
                 "q90": float(matrix[i, j])}
            cells[i][j] = EconomicOutcome(
                crop="test", season="2024-25",
                region_kind="district", region_id="test",
                gross_revenue_inr_per_ha=q,
                cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
                net_revenue_inr_per_ha=q,
                baseline_net_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
                delta_vs_baseline={},
                price_source="msp",
            )
    return PayoffMatrix(
        decisions=dec, states=st, cells=cells,
        payoff_p10=matrix.astype(float),
        payoff_p50=matrix.astype(float),
        payoff_p90=matrix.astype(float),
        baseline_payoff=np.zeros_like(matrix, dtype=float),
        region_kind="test", region_id="",
    )


def test_regret_nonneg_and_zero_per_column():
    """regret >= 0 everywhere; each column has at least one zero."""
    m = np.array([[10.0, -5.0], [0.0, 3.0], [4.0, 1.0]])
    pm = _fake_pm(m, weights=np.array([0.5, 0.5]))
    r = regret_matrix(pm)
    assert (r >= -1e-9).all()
    # Each column has at least one zero
    for j in range(m.shape[1]):
        assert np.isclose(r[:, j].min(), 0.0, atol=1e-9)


def test_minimax_regret_analytic_2x2():
    """On [[10,-5],[0,3]], minimax_regret picks the row with smaller max regret."""
    m = np.array([[10.0, -5.0], [0.0, 3.0]])
    pm = _fake_pm(m, weights=np.array([0.5, 0.5]))
    # regret = [[0, 8], [10, 0]]; max per row = [8, 10]; argmin = 0
    assert minimax_regret(pm) == 0


def test_dominance_row_dominates():
    """Row i weakly dominates row k iff pm[i] >= pm[k] and strict somewhere."""
    m = np.array([[5.0, 5.0], [4.0, 5.0], [3.0, 3.0]])
    pm = _fake_pm(m, weights=np.array([0.5, 0.5]))
    dom = stochastic_dominance(pm)
    # Row 0 strictly dominates row 1 (5>=4 & 5>=5, one strict) and row 2 (both strict)
    assert dom[0, 1] and dom[0, 2]
    assert not dom[1, 0]
    assert not dom[2, 0]


# ─── STEP 10.5: cost–loss bounds ─────────────────────────────────────
def test_cost_loss_perfect_and_climatology_bounds():
    """V(perfect)=1; V(climatology)=0; V(random)≈0 within stochastic tol."""
    rng = np.random.default_rng(42)
    n = 200
    obs = rng.integers(0, 2, size=n)          # 0/1
    p_clim = float(obs.mean())
    setup = CostLossSetup(
        action_id="synth", C_inr_per_ha=1000.0, L_inr_per_ha=8000.0,
        event_label="synth_event", citation="synthetic test",
    )
    # Perfect forecast (== obs, so p_star just below 1 → always right)
    perfect = obs.astype(float) * 1.0 + (1 - obs) * 0.0
    vc_perfect = value_curve(setup, pd.Series(perfect), pd.Series(obs), p_clim)
    assert vc_perfect.V.max() == pytest.approx(1.0, abs=1e-9)

    # Climatology-equivalent forecast: constant p_clim
    clim_fcast = np.full_like(obs, p_clim, dtype=float)
    vc_clim = value_curve(setup, pd.Series(clim_fcast), pd.Series(obs), p_clim)
    # V(climatology) should be <= tiny (never > climatology by definition)
    assert vc_clim.V.max() <= 1e-6

    # Randomised forecast (shuffled obs) → V near 0
    shuffled = obs.copy()
    rng.shuffle(shuffled)
    vc_rand = value_curve(setup, pd.Series(shuffled.astype(float)),
                            pd.Series(obs), p_clim)
    # Loose bound: shuffled series scarcely beats climatology
    assert vc_rand.V.max() < 0.35


# ─── STEP 10.6: backtest leakage guard ───────────────────────────────
def test_backtest_leakage_guard():
    """A rule that peeks at year y raises when the harness detects a
    forecast probability derived from year-y data."""
    class PeekingRule:
        rule_id = "peeking_rule_test"

        def __call__(self, history_before_y, y):
            # Try to look at year y in the history
            if (history_before_y["year"] == y).any():
                raise RuntimeError("saw year y in history — should be impossible")
            # Emit a plain probability (still 0..1)
            return float(history_before_y["event"].mean())

    df = synthesise_history(seed=1)
    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    # Guard confirms the harness never leaks year y — the rule never
    # raises because the harness strictly filters year < y.
    res = walk_forward_backtest(PeekingRule(), setup, df, write_parquet=False)
    assert res.V_forecast is not None       # no LeakageError raised


def test_backtest_publishes_negative_result_honestly():
    """Part-4 Rule 6: the framework must be able to ship a rule that
    fails against climatology. HistoricalFrequency (returns event mean
    every year, no signal) does exactly that under a hard 0.5 decision
    threshold — V comes out ≤ 0. This test *proves* the honesty of
    the framework by checking the negative result lands unhidden."""
    df = synthesise_history(seed=2)
    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    res = walk_forward_backtest(
        HistoricalFrequencyRule(), setup, df, write_parquet=False,
    )
    # The rule cannot beat climatology by design. V must be ≤ 0.
    # If it were > 0 something in the leakage guard is broken.
    assert res.V_forecast <= 1e-6, (
        f"HistoricalFrequency unexpectedly beat climatology: "
        f"V={res.V_forecast:.4f} — audit the leakage guard."
    )
    # Also verify Brier score is well-defined (no silent NaNs)
    assert np.isfinite(res.brier)
    # And the parquet-ready per_year DataFrame is populated
    assert not res.per_year.empty


# ─── STEP 10.7: reliability calibration ──────────────────────────────
def test_reliability_calibrated_series():
    """On a well-calibrated series, mean_forecast ~ observed_freq per bin."""
    rng = np.random.default_rng(7)
    n = 2000
    p = rng.uniform(0, 1, size=n)
    obs = (rng.uniform(0, 1, size=n) < p).astype(int)
    rel = reliability_diagram(p, obs, bins=10)
    rel = rel.dropna(subset=["mean_forecast", "observed_freq"])
    err = (rel["mean_forecast"] - rel["observed_freq"]).abs().mean()
    assert err < 0.05


def test_reliability_biased_series_detected():
    """On a systematically biased series the deviation is detected."""
    rng = np.random.default_rng(11)
    n = 2000
    p = rng.uniform(0.6, 1.0, size=n)         # over-confident forecasts
    obs = (rng.uniform(0, 1, size=n) < 0.4).astype(int)      # low base rate
    rel = reliability_diagram(p, obs, bins=10)
    rel = rel.dropna(subset=["mean_forecast", "observed_freq"])
    err = (rel["mean_forecast"] - rel["observed_freq"]).abs().mean()
    assert err > 0.15


# ─── STEP 10.8: tornado determinism ──────────────────────────────────
def test_tornado_zero_delta_is_zero_range():
    """delta_pct=0 ⇒ every knob's range is 0 (deterministic scenario_fn)."""
    base_out = EconomicOutcome(
        crop="paddy_kharif", season="2024-25",
        region_kind="district", region_id="test",
        gross_revenue_inr_per_ha={"q10": 50000.0, "q50": 60000.0, "q90": 70000.0},
        cost_inr_per_ha={"q10": 20000.0, "q50": 20000.0, "q90": 20000.0},
        net_revenue_inr_per_ha={"q10": 30000.0, "q50": 40000.0, "q90": 50000.0},
        baseline_net_inr_per_ha={"q10": 25000.0, "q50": 25000.0, "q90": 25000.0},
        delta_vs_baseline={},
        price_source="msp",
    )
    def _scenario_fn(_levers):
        return base_out
    result = tornado(_scenario_fn, base_levers={}, delta_pct=0.0)
    assert (result.rows["range"] == 0).all()
    assert result.dominant_share_pct == 0.0


def test_tornado_moves_when_knobs_change():
    """delta_pct=0.20 with a scenario_fn that reads overrides should
    produce a non-zero range and a dominant-knob string."""
    def _scenario_fn(levers):
        ov = (levers or {}).get("overrides", {})
        base = 40000.0
        base += 20000.0 * float(ov.get("msp_pct", 0.0))    # price dominates
        base += 2000.0 * float(ov.get("cost_pct", 0.0))    # cost weaker
        return EconomicOutcome(
            crop="paddy_kharif", season="2024-25",
            region_kind="district", region_id="test",
            gross_revenue_inr_per_ha={"q10": base, "q50": base, "q90": base},
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha={"q10": base, "q50": base, "q90": base},
            baseline_net_inr_per_ha={"q10": 25000.0, "q50": 25000.0, "q90": 25000.0},
            delta_vs_baseline={},
            price_source="msp",
        )

    result = tornado(_scenario_fn, base_levers={}, delta_pct=0.20)
    assert (result.rows["range"] > 0).any()
    # MSP dominates by construction
    assert result.rows.iloc[0]["knob"] == "msp"


# ─── STEP 10.9: resolution ceiling ───────────────────────────────────
def test_valuation_refuses_grid_input():
    """value_agriculture raises when given an xr.Dataset with (lat, lon)."""
    crop = load_crop("paddy_kharif")
    ps = load_prices("paddy_kharif")
    ya = 3.0
    yq = {"q10": ya, "q50": ya, "q90": ya}
    # Build a Dataset that still has lat/lon dims — the grid input we
    # must never let through valuation.
    grid_ds = xr.Dataset({
        "Ya": xr.DataArray(np.zeros((4, 4)), dims=("lat", "lon")),
    })
    with pytest.raises(ResolutionCeilingError):
        value_agriculture(
            yield_qdict=yq, baseline_ya_t_ha=ya,
            crop=crop, price_set=ps, season="2024-25",
            region_kind="grid", region_id="test",
            yield_ds_for_shape_check=grid_ds,
        )


# ─── STEP 10.10: brier + payoff sanity ───────────────────────────────
def test_brier_perfect_and_worst():
    """Brier score of perfect forecast is 0; of maximally-wrong is 1."""
    obs = np.array([0, 1, 0, 1, 1], dtype=int)
    assert brier_score(obs.astype(float), obs) == pytest.approx(0.0, abs=1e-12)
    worst = 1.0 - obs                          # opposite of truth
    assert brier_score(worst.astype(float), obs) == pytest.approx(1.0, abs=1e-12)


def test_payoff_weights_must_sum_to_one():
    """Weights that don't sum to 1 raise on payoff-matrix build."""
    dec = [Decision(label="A", kind="crop"), Decision(label="B", kind="crop")]
    st = [ClimateState(label="s1", weight=0.6), ClimateState(label="s2", weight=0.3)]
    def _stub_cell(dec, st):
        return EconomicOutcome(
            crop="test", season="2024-25",
            region_kind="district", region_id="",
            gross_revenue_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            baseline_net_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            delta_vs_baseline={},
            price_source="msp",
        )
    from climate_twin.whatif.config.region import RegionSpec
    with pytest.raises(ValueError):
        build_payoff_matrix(dec, st, RegionSpec("all_india"), run_cell=_stub_cell)


# ─── STEP 10.11: version + citation strings ─────────────────────────
def test_prices_registry_versioning():
    """prices_registry_version + SHA appear + are non-empty."""
    v = prices_registry_version()
    s = prices_registry_sha256()
    assert v.startswith("prices-")
    assert len(s) >= 8
    ps = load_prices("paddy_kharif")
    assert ps.registry_version == v
    assert ps.registry_sha256 == s
    assert ps.citation.get("msp"), "MSP citation missing"


def test_shipped_setups_have_citations():
    """Every shipped setup ships with a non-empty citation string."""
    for k, s in SHIPPED_SETUPS.items():
        assert s.citation and len(s.citation) > 20, f"{k} citation is too short"
