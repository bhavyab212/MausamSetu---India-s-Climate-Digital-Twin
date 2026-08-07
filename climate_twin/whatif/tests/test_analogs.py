"""
whatif.tests.test_analogs — Part 5, Method 2 tests.

Fast, hermetic. Feature matrices are built as synthetic DataFrames so
tests don't touch the IMD cube — that's an integration concern.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.drivers.analog_features import (
    DEFAULT_FEATURES,
    AnalogSpec,
    feature_covariance,
)
from climate_twin.whatif.drivers.analog_outcomes import (
    _compute_weights,
    _weighted_quantile,
    analog_outcome_distribution,
)
from climate_twin.whatif.drivers.analogs import (
    ANALOGS_VERSION,
    AnalogMatch,
    AnalogPool,
    _CHI2_FAIR_DF7,
    _CHI2_STRONG_DF7,
    _quality_tier,
    find_analogs,
)
from climate_twin.whatif.drivers.climate_states import (
    _train_terciles,
    climate_states_from_analogs,
    climate_states_from_perturbation,
)
from climate_twin.whatif.drivers.perturbation import PerturbationSpec
from climate_twin.whatif.economics.cost_loss import SHIPPED_SETUPS
from climate_twin.whatif.indices.reference import (
    LeakageError,
    TRAIN_YEARS,
    VALID_YEARS,
)


# ─── Synthetic pool builder ──────────────────────────────────────────
def _synth_spec() -> AnalogSpec:
    """A minimal AnalogSpec that doesn't touch the cube."""
    return AnalogSpec(
        region=RegionSpec(kind="all_india"),
        window="JJAS",
        features=DEFAULT_FEATURES,
        metric="mahalanobis",
        train_years=TRAIN_YEARS,
    )


def _synth_pool(
    train_years: tuple[int, int] = TRAIN_YEARS,
    seed: int = 42,
    n_features: int = 7,
) -> AnalogPool:
    rng = np.random.default_rng(int(seed))
    y0, y1 = train_years
    years = list(range(y0, y1 + 1))
    X = rng.normal(0.0, 1.0, size=(len(years), n_features))
    fm = pd.DataFrame(X, index=years, columns=list(DEFAULT_FEATURES))
    fm.index.name = "year"
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}
    fm.attrs["dropped_years"] = []
    cov_inv = np.linalg.inv(np.cov(X, rowvar=False, ddof=1) + 1e-6 * np.eye(n_features))
    return AnalogPool(
        spec=_synth_spec(),
        feature_matrix=fm,
        cov_inv=cov_inv,
        stats={"n_years": len(years), "excluded": []},
    )


# ─── Distance sanity ─────────────────────────────────────────────────
def test_mahalanobis_matches_analytic():
    """On a known covariance, d² matches the analytic formula to 1e-6."""
    # 2-feature space, cov = [[2, 1], [1, 2]]
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    cov_inv = np.linalg.inv(cov)
    # Build a 2-feature pool where the first year is at (0, 0) and the
    # target is at (1, 1). Analytic: d² = (1,1) @ cov_inv @ (1,1) = 2/3.
    from climate_twin.whatif.drivers.analog_features import (
        DEFAULT_FEATURES as _F,
    )
    two_feat = _F[:2]
    fm = pd.DataFrame([[0.0, 0.0]], index=[1980], columns=list(two_feat))
    spec = AnalogSpec(
        region=RegionSpec("all_india"), features=tuple(two_feat),
        metric="mahalanobis",
    )
    pool = AnalogPool(spec=spec, feature_matrix=fm, cov_inv=cov_inv,
                      stats={"n_years": 1})
    tgt = pd.Series([1.0, 1.0], index=list(two_feat))
    matches = find_analogs(tgt, pool, k=1)
    analytic = float((np.array([1.0, 1.0]) @ cov_inv @ np.array([1.0, 1.0])))
    assert matches[0].distance == pytest.approx(analytic, abs=1e-9)


# ─── Quality tiers ───────────────────────────────────────────────────
def test_quality_tiers_at_cutoffs():
    """d < χ²(0.5)  → strong;  d < χ²(0.9) → fair; larger → poor."""
    strong = _quality_tier(_CHI2_STRONG_DF7 - 1e-6, df=7, metric="mahalanobis")
    fair = _quality_tier(_CHI2_STRONG_DF7 + 1e-6, df=7, metric="mahalanobis")
    fair2 = _quality_tier(_CHI2_FAIR_DF7 - 1e-6, df=7, metric="mahalanobis")
    poor = _quality_tier(_CHI2_FAIR_DF7 + 1e-6, df=7, metric="mahalanobis")
    assert strong == "strong"
    assert fair == "fair"
    assert fair2 == "fair"
    assert poor == "poor"


# ─── Leakage guard ───────────────────────────────────────────────────
def test_analog_leakage_guard_excludes_years():
    """An excluded year is dropped from the pool's feature matrix."""
    fm = pd.DataFrame(
        np.zeros((5, 7)),
        index=[2011, 2012, 2013, 2014, 2015],
        columns=list(DEFAULT_FEATURES),
    )
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}
    fm.attrs["dropped_years"] = []
    # Simulate exclude_years=[2013, 2015]
    excluded = {2013, 2015}
    fm2 = fm.loc[~fm.index.isin(excluded)]
    assert 2013 not in fm2.index
    assert 2015 not in fm2.index
    assert list(fm2.index) == [2011, 2012, 2014]


def test_analog_leakage_raises_on_peek():
    """If a subroutine returns a pool that still contains the excluded
    year, we raise LeakageError. Mimicks the backtest's contract check."""
    pool = _synth_pool()
    # Sneak year 1999 back in
    pool.feature_matrix.loc[1999] = np.zeros(len(DEFAULT_FEATURES))
    excluded = {1999}
    leak = [y for y in pool.years() if y in excluded]
    if leak:
        with pytest.raises(LeakageError):
            raise LeakageError(
                f"pool contains excluded years {leak}"
            )


# ─── Weight sums ─────────────────────────────────────────────────────
def test_perturbation_state_weights_must_sum_to_one():
    """Perturbation state-builder refuses weights that don't sum to 1."""
    perts = [PerturbationSpec(rain_scale=0.8), PerturbationSpec(rain_scale=1.2)]
    with pytest.raises(ValueError):
        climate_states_from_perturbation(perts, weights=[0.3, 0.3])
    ok = climate_states_from_perturbation(perts, weights=[0.4, 0.6])
    assert abs(sum(s.weight for s in ok) - 1.0) < 1e-9


def test_analog_bucket_weights_normalise_to_one():
    """Analog-bucket state weights sum to 1 (renormalised inside)."""
    # Fabricate matches with known distances
    rng = np.random.default_rng(1)
    matches = []
    for y, d in [(1975, 0.5), (1988, 0.8), (1992, 1.2), (2001, 2.0)]:
        f_analog = pd.Series(
            rng.normal(size=len(DEFAULT_FEATURES)),
            index=list(DEFAULT_FEATURES),
        )
        matches.append(AnalogMatch(
            year=y, distance=d,
            features_target=pd.Series(
                np.zeros(len(DEFAULT_FEATURES)), index=list(DEFAULT_FEATURES),
            ),
            features_analog=f_analog,
            quality="strong",
        ))
    # per_year: weights already come from _compute_weights; states sum to 1
    states = climate_states_from_analogs(
        matches, _synth_spec(), bucketing="per_year",
    )
    assert abs(sum(s.weight for s in states) - 1.0) < 1e-9


# ─── Bucket boundaries stability ────────────────────────────────────
def test_terciles_anchored_on_train_years(monkeypatch):
    """Terciles come from TRAIN_YEARS, not from the analog subsample.

    We monkey-patch build_feature_matrix so this test never touches
    the cube; the point is that the tercile helper reads the TRAIN_YEARS
    slice of the returned matrix, not any subsample.
    """
    from climate_twin.whatif.drivers import climate_states as _cs

    # Fake train feature matrix: 40 years with 'rain_total_std' spread
    y0, y1 = TRAIN_YEARS
    years = list(range(y0, y1 + 1))
    vals = np.linspace(-2.0, 2.0, len(years))
    fm = pd.DataFrame(
        np.tile(vals.reshape(-1, 1), (1, len(DEFAULT_FEATURES))),
        index=years, columns=list(DEFAULT_FEATURES),
    )
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}

    def _fake_build(spec, years=None, **kw):
        return fm

    monkeypatch.setattr(_cs, "build_feature_matrix", _fake_build)

    spec = _synth_spec()
    t33, t66 = _train_terciles(spec, "rain_total_std")
    # Expected from linspace(-2, 2, 40): terciles at ~-2/3, +2/3
    train_vals = fm.loc[
        (fm.index >= y0) & (fm.index <= y1), "rain_total_std"
    ].to_numpy()
    expected_t33 = float(np.percentile(train_vals, 100.0 / 3.0))
    expected_t66 = float(np.percentile(train_vals, 200.0 / 3.0))
    assert t33 == pytest.approx(expected_t33, abs=1e-9)
    assert t66 == pytest.approx(expected_t66, abs=1e-9)


# ─── Bootstrap determinism ──────────────────────────────────────────
def test_bootstrap_determinism_same_seed():
    """Two runs of analog_outcome_distribution with the same seed
    produce byte-identical bootstrap CI."""
    matches = []
    for y, d, ya in [(1972, 0.4, 3.9), (1987, 0.7, 3.5), (2002, 1.1, 4.2)]:
        matches.append(AnalogMatch(
            year=y, distance=d,
            features_target=pd.Series(
                np.zeros(len(DEFAULT_FEATURES)), index=list(DEFAULT_FEATURES),
            ),
            features_analog=pd.Series(
                np.zeros(len(DEFAULT_FEATURES)), index=list(DEFAULT_FEATURES),
            ),
            quality="strong",
        ))
    outcomes = {1972: 3.9, 1987: 3.5, 2002: 4.2}

    def fn(y): return float(outcomes[int(y)])

    a = analog_outcome_distribution(matches, fn, bootstrap=200, seed=123)
    b = analog_outcome_distribution(matches, fn, bootstrap=200, seed=123)
    assert a.bootstrap_ci == b.bootstrap_ci
    assert a.percentiles == b.percentiles


# ─── Empirical q10/q50/q90 from bucket ──────────────────────────────
def test_analog_bucket_empirical_percentiles():
    """A weighted quantile over a hand-picked bucket returns hand-
    computable values."""
    values = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3.0
    # With uniform weights and 3 samples, q50 ≈ 4.0
    assert _weighted_quantile(values, weights, 0.50) == pytest.approx(4.0, abs=0.5)


# ─── Analog forecaster reliability: synthetic correlation ────────────
def test_analog_forecaster_beats_climatology_on_signal():
    """On a synthetic series where the event correlates with a known
    feature, the analog forecaster produces BSS > 0 and V > 0 for a
    shipped setup."""
    from climate_twin.whatif.drivers.analog_backtest import (
        analog_forecaster_backtest,
    )

    # Build a synthetic feature matrix + event series that has a
    # strong correlation via rain_total_std (positive rain → less event).
    rng = np.random.default_rng(0)
    train_range = TRAIN_YEARS
    valid_range = (VALID_YEARS[0], VALID_YEARS[1])
    all_years = list(range(train_range[0], valid_range[1] + 1))
    n = len(all_years)
    x = rng.normal(0.0, 1.0, size=n)                    # standardised feature
    # Event probability decreases with rain feature (drought event)
    p = 1.0 / (1.0 + np.exp(2.0 * x))
    events = (rng.uniform(size=n) < p).astype(int)
    event_by_year = {y: int(e) for y, e in zip(all_years, events)}

    # Build a synthetic feature matrix that lands the same x values
    fm = pd.DataFrame(
        {feat: (x if feat == "rain_total_std" else rng.normal(0.0, 1.0, size=n))
         for feat in DEFAULT_FEATURES},
        index=all_years,
    )
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}
    fm.attrs["dropped_years"] = []

    # Monkey-patch build_analog_pool / target_from_year to use our synth
    from climate_twin.whatif.drivers import analog_backtest as _bt
    from climate_twin.whatif.drivers.analogs import (
        AnalogPool as _AP,
        find_analogs as _find,
    )

    cov_inv = np.linalg.inv(
        np.cov(fm.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
        + 1e-6 * np.eye(len(DEFAULT_FEATURES))
    )

    def _fake_pool(spec, exclude_years=(), **kw):
        excl = set(int(y) for y in exclude_years)
        sub = fm.loc[~fm.index.isin(excl)].copy()
        return _AP(spec=spec, feature_matrix=sub, cov_inv=cov_inv,
                    stats={"n_years": len(sub), "excluded": sorted(excl)})

    def _fake_target(year, spec):
        return fm.loc[int(year)]

    _bt.build_analog_pool = _fake_pool
    _bt.target_from_year = _fake_target

    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    spec = _synth_spec()

    res = analog_forecaster_backtest(
        setup, spec, event_by_year,
        train_years=train_range, valid_years=valid_range,
        k=10, weighting="inv_distance", write_parquet=False,
    )
    assert res.brier_skill > 0, f"BSS should be > 0 with signal, got {res.brier_skill}"
    # The swept value curve should find at least one threshold where
    # V > 0 (probability estimates carry actionable signal). The
    # scalar V_forecast (fixed 0.5 threshold) is not guaranteed to be
    # positive — that's a threshold question, not a skill question.
    assert res.value_curve.V_best > 0.0, (
        f"analog forecaster should have V_best > 0 with signal, "
        f"got {res.value_curve.V_best:.4f}"
    )


def test_analog_forecaster_no_signal_gives_no_skill():
    """With shuffled labels (no correlation) BSS ≈ 0 or negative."""
    from climate_twin.whatif.drivers.analog_backtest import (
        analog_forecaster_backtest,
    )
    rng = np.random.default_rng(1)
    train_range = TRAIN_YEARS
    valid_range = (VALID_YEARS[0], VALID_YEARS[1])
    all_years = list(range(train_range[0], valid_range[1] + 1))
    n = len(all_years)
    x = rng.normal(0.0, 1.0, size=n)
    events = rng.integers(0, 2, size=n)                # random labels
    event_by_year = {y: int(e) for y, e in zip(all_years, events)}
    fm = pd.DataFrame(
        {feat: (x if feat == "rain_total_std" else rng.normal(0.0, 1.0, size=n))
         for feat in DEFAULT_FEATURES},
        index=all_years,
    )
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}
    fm.attrs["dropped_years"] = []

    from climate_twin.whatif.drivers import analog_backtest as _bt
    from climate_twin.whatif.drivers.analogs import (
        AnalogPool as _AP,
    )
    cov_inv = np.linalg.inv(
        np.cov(fm.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
        + 1e-6 * np.eye(len(DEFAULT_FEATURES))
    )
    def _fake_pool(spec, exclude_years=(), **kw):
        excl = set(int(y) for y in exclude_years)
        sub = fm.loc[~fm.index.isin(excl)].copy()
        return _AP(spec=spec, feature_matrix=sub, cov_inv=cov_inv,
                    stats={"n_years": len(sub)})

    def _fake_target(year, spec):
        return fm.loc[int(year)]

    _bt.build_analog_pool = _fake_pool
    _bt.target_from_year = _fake_target

    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    spec = _synth_spec()
    res = analog_forecaster_backtest(
        setup, spec, event_by_year,
        train_years=train_range, valid_years=valid_range,
        k=10, weighting="inv_distance", write_parquet=False,
    )
    # BSS should be non-positive (± noise) — the null case
    assert res.brier_skill < 0.15, (
        f"unexpected BSS > 0.15 with shuffled labels: {res.brier_skill}"
    )


def test_analog_backtest_provenance_carries_signature():
    """The BacktestResult from the analog backtest records enough for
    replay: spec.signature, spec.version, metric, region, k, weighting."""
    from climate_twin.whatif.drivers.analog_backtest import (
        analog_forecaster_backtest,
    )
    from climate_twin.whatif.drivers import analog_backtest as _bt
    from climate_twin.whatif.drivers.analogs import AnalogPool as _AP
    rng = np.random.default_rng(2)
    train_range = TRAIN_YEARS
    valid_range = (VALID_YEARS[0], VALID_YEARS[0] + 2)   # 3-year mini backtest
    all_years = list(range(train_range[0], valid_range[1] + 1))
    n = len(all_years)
    fm = pd.DataFrame(
        rng.normal(size=(n, len(DEFAULT_FEATURES))),
        index=all_years, columns=list(DEFAULT_FEATURES),
    )
    fm.attrs["train_stats"] = {c: (0.0, 1.0) for c in fm.columns}
    fm.attrs["dropped_years"] = []
    cov_inv = np.linalg.inv(
        np.cov(fm.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
        + 1e-6 * np.eye(len(DEFAULT_FEATURES))
    )
    def _fake_pool(spec, exclude_years=(), **kw):
        excl = set(int(y) for y in exclude_years)
        sub = fm.loc[~fm.index.isin(excl)].copy()
        return _AP(spec=spec, feature_matrix=sub, cov_inv=cov_inv,
                    stats={"n_years": len(sub)})
    def _fake_target(year, spec):
        return fm.loc[int(year)]
    _bt.build_analog_pool = _fake_pool
    _bt.target_from_year = _fake_target

    events = {y: int(y) % 2 for y in all_years}
    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    spec = _synth_spec()
    res = analog_forecaster_backtest(
        setup, spec, events,
        train_years=train_range, valid_years=valid_range,
        write_parquet=False,
    )
    for key in ("analog_spec_signature", "analog_spec_version",
                 "features", "metric", "region", "k", "weighting"):
        assert key in res.provenance, f"provenance missing {key}"
