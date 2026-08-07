"""
whatif.drivers.analogs — L0, Method 2: historical-analog retrieval.

Primary sources:
    * Lorenz, E.N. (1969) "Atmospheric predictability as revealed by
      naturally occurring analogues." J. Atmos. Sci. 26:636-646.
    * van den Dool, H.M. (1994) "Searching for analogues, how long must
      we wait?" Tellus A 46:314-324. Statistical realism about analog
      availability in a limited record.
    * Zorita & von Storch (1999); Delle Monache et al. (2013);
      Hamill & Whitaker (2006). See :mod:`analog_features` docstring
      for full citations.

Quality tiers (Mahalanobis distance-squared, 7-feature default vector):
    * strong :  d² < χ²(0.5, df=7)  ≈ 6.35   (median of the null)
    * fair   :  d² < χ²(0.9, df=7)  ≈ 12.02  (top 10 % of the null)
    * poor   :  anything larger. Report anyway — Rule 7: never lie
                about analog count.

The χ² cutoffs follow from the fact that under the null "target drawn
from the same distribution as the pool", the Mahalanobis d² is
χ²-distributed with df = number of features.

Version: ``analogs-v1`` (bumps when the pool builder or metric changes).
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ..config.paths import CACHE_DIR
from ..indices.reference import LeakageError, TRAIN_YEARS
from .analog_features import (
    AnalogSpec,
    build_feature_matrix,
    feature_covariance,
    target_features_from_row,
)

ANALOGS_VERSION = "analogs-v1"

# Chi-squared thresholds for d² under a df=7 null. Values pinned so
# switching to a df != 7 feature vector requires an explicit override.
_CHI2_STRONG_DF7: float = 6.35        # χ²(0.5, df=7)
_CHI2_FAIR_DF7: float = 12.02         # χ²(0.9, df=7)


@dataclass
class AnalogPool:
    """A pool of historical years available as analog matches.

    ``feature_matrix`` is already standardised (produced by
    :func:`whatif.drivers.analog_features.build_feature_matrix`).
    ``cov_inv`` is the inverse feature covariance restricted to the
    training window — used for Mahalanobis distance. Pre-computed +
    cached so queries are O(features²) not O(features³) per call.
    """
    spec: AnalogSpec
    feature_matrix: pd.DataFrame
    cov_inv: np.ndarray | None
    stats: dict
    version: str = ANALOGS_VERSION

    def years(self) -> list[int]:
        return [int(y) for y in self.feature_matrix.index]


@dataclass
class AnalogMatch:
    """One retrieved analog year with its distance + quality tier."""
    year: int
    distance: float
    features_target: pd.Series
    features_analog: pd.Series
    quality: Literal["strong", "fair", "poor"]

    def as_row(self) -> dict:
        row = {"year": int(self.year),
                "distance": float(self.distance),
                "quality": self.quality}
        row.update({f"target_{k}": float(v)
                    for k, v in self.features_target.items()})
        row.update({f"analog_{k}": float(v)
                    for k, v in self.features_analog.items()})
        return row


def _pool_cache_path(spec: AnalogSpec, exclude_years: tuple[int, ...]) -> Path:
    key = f"{spec.signature()}|excl={sorted(exclude_years)}|{ANALOGS_VERSION}"
    d = hashlib.sha256(key.encode()).hexdigest()[:12]
    return CACHE_DIR / "analog_pools" / f"pool_{d}.pkl"


def build_analog_pool(
    spec: AnalogSpec,
    exclude_years: Iterable[int] = (),
    *,
    pool_years: tuple[int, int] | None = None,
    use_cache: bool = True,
) -> AnalogPool:
    """Assemble a pool of candidate analog years.

    ``exclude_years`` is how the walk-forward backtest enforces "no peek
    at year y" (Rule 1). Excluded years are dropped from the feature
    matrix; the standardisation statistics are still fitted on
    ``spec.train_years`` (Rule 2).

    ``pool_years`` — the widest year window from which analogs may be
    drawn. Defaults to ``TRAIN_YEARS`` since Rule 1 forbids using
    VALID_YEARS as analog candidates for forecast targets in those
    years.
    """
    excl = tuple(sorted(set(int(y) for y in exclude_years)))
    years = pool_years or (spec.train_years[0], spec.train_years[1])
    cache = _pool_cache_path(spec, excl)
    if use_cache and cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    fm = build_feature_matrix(spec, years=years, use_cache=use_cache)
    if excl:
        fm = fm.loc[~fm.index.isin(excl)].copy()
    # Sanity: keep only rows inside pool_years
    fm = fm.loc[(fm.index >= years[0]) & (fm.index <= years[1])].copy()

    cov_inv = None
    if spec.metric == "mahalanobis":
        cov_inv = feature_covariance(fm, spec.train_years)
    pool = AnalogPool(
        spec=spec,
        feature_matrix=fm,
        cov_inv=cov_inv,
        stats={
            "n_years": int(len(fm)),
            "excluded": list(excl),
            "train_stats": dict(fm.attrs.get("train_stats", {})),
            "dropped_years": list(fm.attrs.get("dropped_years", [])),
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache, "wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return pool


# ─── Distance metrics ────────────────────────────────────────────────
def _mahalanobis(x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray) -> float:
    d = x - y
    return float(d @ cov_inv @ d)          # returns d² (squared distance)


def _euclidean(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return float(d @ d)                    # d²


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    xn = np.linalg.norm(x)
    yn = np.linalg.norm(y)
    if xn <= 0 or yn <= 0:
        return 2.0                          # maximally dissimilar
    return float(1.0 - (x @ y) / (xn * yn))


def _pair_distance(x: np.ndarray, y: np.ndarray, spec: AnalogSpec,
                    cov_inv: np.ndarray | None) -> float:
    if spec.metric == "mahalanobis":
        if cov_inv is None:
            raise ValueError("Mahalanobis requested but cov_inv is None")
        return _mahalanobis(x, y, cov_inv)
    if spec.metric == "euclidean":
        return _euclidean(x, y)
    if spec.metric == "cosine":
        return _cosine(x, y)
    raise ValueError(f"unknown metric {spec.metric!r}")


def _quality_tier(distance: float, df: int, metric: str) -> Literal["strong", "fair", "poor"]:
    """Return the quality tier for a distance.

    For Mahalanobis (df=7 default) we use the χ² cutoffs pinned above.
    For other metrics we scale the cutoffs by 1.0 (Euclidean on
    standardised features has the same df-χ² interpretation up to
    a rotation) and by 0.5 (Cosine is bounded in [0, 2]).
    """
    if metric == "mahalanobis":
        strong = _CHI2_STRONG_DF7 if df == 7 else float(df) * (
            _CHI2_STRONG_DF7 / 7.0
        )
        fair = _CHI2_FAIR_DF7 if df == 7 else float(df) * (
            _CHI2_FAIR_DF7 / 7.0
        )
    elif metric == "euclidean":
        # Same distribution as Mahalanobis when cov=I (whitened features).
        strong = _CHI2_STRONG_DF7 if df == 7 else float(df) * (
            _CHI2_STRONG_DF7 / 7.0
        )
        fair = _CHI2_FAIR_DF7 if df == 7 else float(df) * (
            _CHI2_FAIR_DF7 / 7.0
        )
    elif metric == "cosine":
        strong, fair = 0.05, 0.15
    else:
        raise ValueError(f"unknown metric {metric!r}")

    if distance < strong:
        return "strong"
    if distance < fair:
        return "fair"
    return "poor"


# ─── Public retrieval ────────────────────────────────────────────────
def find_analogs(
    target_features: pd.Series,
    pool: AnalogPool,
    k: int = 10,
) -> list[AnalogMatch]:
    """Return the ``k`` closest analog years to ``target_features``.

    ``target_features`` must carry the same columns as
    ``pool.feature_matrix``. Returns a list sorted by ascending
    distance. Fewer than ``k`` are returned only if the pool is smaller
    than ``k`` (never silently padded with poor matches — Rule 7).
    """
    spec = pool.spec
    cols = list(spec.features)
    tgt = target_features_from_row(target_features, spec).to_numpy(dtype=np.float64)

    fm = pool.feature_matrix
    if len(fm) == 0:
        return []
    X = fm[cols].to_numpy(dtype=np.float64)

    dists = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        dists[i] = _pair_distance(X[i], tgt, spec, pool.cov_inv)

    order = np.argsort(dists, kind="stable")
    df = len(cols)
    out: list[AnalogMatch] = []
    for i in order[:k]:
        year = int(fm.index[int(i)])
        q = _quality_tier(float(dists[int(i)]), df, spec.metric)
        out.append(AnalogMatch(
            year=year,
            distance=float(dists[int(i)]),
            features_target=pd.Series(tgt, index=cols),
            features_analog=fm[cols].iloc[int(i)].astype(np.float64),
            quality=q,
        ))
    return out


# ─── Target-features helpers ─────────────────────────────────────────
def target_from_year(year: int, spec: AnalogSpec) -> pd.Series:
    """Compute standardised feature vector for one *observed* year.

    Used by the walk-forward backtest — the target for verification
    year y is what actually happened in y, and we ask "what past year
    looks most like y?"
    """
    fm = build_feature_matrix(
        spec,
        years=(year, year),
        use_cache=False,
    )
    if year not in fm.index:
        raise KeyError(
            f"could not build features for year {year} at region "
            f"{spec.region.kind}:{spec.region.id}, window {spec.window}"
        )
    return fm.loc[year].astype(np.float64)


def target_from_forecast(
    forecast_ds, spec: AnalogSpec, quantile: str = "q50",
) -> pd.Series:
    """Compute the feature vector on a forecast Dataset.

    ``forecast_ds`` must expose ``rain``, ``tmax``, ``tmin`` at the
    same (time, lat, lon) grid the historical driver uses. The
    ``quantile`` argument is unused at this layer — the caller is
    expected to have already selected the desired quantile from the
    ensemble; we recompute the region-integrated features on the
    selected series.

    Not implemented — waits for the ensemble → forecast Dataset
    adapter in Part 6. Deferred.
    """
    raise NotImplementedError(
        "target_from_forecast lands with the ensemble→forecast adapter "
        "in Part 6. Use target_from_year in the walk-forward backtest."
    )


def target_from_perturbation(*args, **kwargs) -> pd.Series:
    """Perturbation-flavoured target features.

    The perturbation modifies base data by known deltas. The target
    feature vector is what those features would be if the perturbation
    were applied. Deferred to Part 6; today the payoff-matrix
    perturbation path bypasses this helper by using the perturbation
    directly as a driver.
    """
    raise NotImplementedError(
        "target_from_perturbation is deferred to Part 6."
    )
