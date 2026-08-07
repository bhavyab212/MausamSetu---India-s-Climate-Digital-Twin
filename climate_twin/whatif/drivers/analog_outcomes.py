"""
whatif.drivers.analog_outcomes — outcome distribution from analog years.

The point of finding analogs is not to name years — it's to look up
what actually happened in those years and treat the *empirical
distribution over their outcomes* as a forecast.

Primary sources:
    * Hamill, T.M. & Whitaker, J.S. (2006) "Probabilistic quantitative
      precipitation forecasts based on reforecast analogs: theory and
      application." Mon. Weather Rev. 134:3209-3229. §3 the outcome-
      distribution construction.
    * Delle Monache, L. et al. (2013). §3 the softmax weighting.

Weighting flavours:
    * uniform      : w_i = 1/k          — equal say.
    * inv_distance : w_i ∝ 1/(d_i + ε)  — closer analogs matter more.
                      ε = 1e-3.
    * softmax      : w_i ∝ exp(-d_i²/τ) — Delle Monache 2013 style;
                      τ = 1.0 by default (calibrated per region+window;
                      user override bumps `version`).

Determinism:
    * The bootstrap uses ``numpy.random.default_rng(seed)``; seed is
      recorded in provenance (Rule 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .analogs import AnalogMatch

ANALOG_OUTCOMES_VERSION = "analog-outcomes-v1"
_EPS = 1e-3          # inverse-distance regulariser


Weighting = Literal["uniform", "inv_distance", "softmax"]


@dataclass
class AnalogOutcomeDist:
    """Weighted empirical distribution over analog-year outcomes."""
    matches: list[AnalogMatch]
    outcome_by_year: dict[int, xr.Dataset | xr.DataArray | float]
    weights: dict[int, float]
    percentiles: dict[str, xr.Dataset | xr.DataArray | float]
    bootstrap_ci: dict[str, xr.Dataset | xr.DataArray | float] | None
    weighting: Weighting
    softmax_tau: float
    seed: int
    version: str = ANALOG_OUTCOMES_VERSION
    provenance: dict = field(default_factory=dict)


def _compute_weights(matches: list[AnalogMatch], weighting: Weighting,
                      softmax_tau: float) -> dict[int, float]:
    if not matches:
        return {}
    ds = np.array([m.distance for m in matches], dtype=np.float64)
    years = [m.year for m in matches]
    if weighting == "uniform":
        w = np.ones_like(ds) / len(ds)
    elif weighting == "inv_distance":
        w = 1.0 / (ds + _EPS)
        w = w / w.sum()
    elif weighting == "softmax":
        # d is already d² for Mahalanobis / Euclidean; we exponentiate
        # -d²/τ so a smaller τ sharpens the distribution.
        e = np.exp(-ds / max(softmax_tau, 1e-9))
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones_like(ds) / len(ds)
        else:
            w = e / s
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return {int(y): float(w_i) for y, w_i in zip(years, w)}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                        q: float) -> float:
    """Weighted quantile (Type-7-style linear interpolation).

    ``values`` and ``weights`` must be 1-D and same length. ``q ∈ [0, 1]``.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    cum = np.cumsum(w) - 0.5 * w      # centred at each sample
    return float(np.interp(q, cum, v))


def _percentiles_scalar(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "q10": _weighted_quantile(values, weights, 0.10),
        "q50": _weighted_quantile(values, weights, 0.50),
        "q90": _weighted_quantile(values, weights, 0.90),
    }


def _bootstrap_ci_scalar(values: np.ndarray, weights: np.ndarray,
                          *, n: int, seed: int) -> dict[str, float]:
    """Bootstrap CI on the weighted q50, plus q05 and q95 of the bootstrap
    distribution. Fixed seed → deterministic output."""
    rng = np.random.default_rng(int(seed))
    k = len(values)
    if k < 2:
        return {"q05": float("nan"), "q95": float("nan")}
    q50_samples = np.empty(n, dtype=np.float64)
    idx_bank = rng.integers(0, k, size=(n, k))
    for i in range(n):
        idx = idx_bank[i]
        q50_samples[i] = _weighted_quantile(values[idx], weights[idx], 0.50)
    return {
        "q05": float(np.percentile(q50_samples, 5)),
        "q95": float(np.percentile(q50_samples, 95)),
    }


def analog_outcome_distribution(
    matches: list[AnalogMatch],
    outcome_fn: Callable[[int], float | xr.Dataset | xr.DataArray],
    *,
    weighting: Weighting = "inv_distance",
    softmax_tau: float = 1.0,
    bootstrap: int = 500,
    seed: int = 20260807,
) -> AnalogOutcomeDist:
    """Assemble a weighted outcome distribution over ``matches``.

    ``outcome_fn(year)`` returns the observed outcome for that year at
    the region of interest. For agriculture it wraps
    ``run_agriculture_scenario`` with observed IMD data for the year;
    the caller supplies the wrapper so this module never reaches into
    a sector directly.

    For scalar outcomes (a single ₹/ha or t/ha), ``percentiles`` and
    ``bootstrap_ci`` are plain floats. For xr outcomes the same
    percentiles are computed element-wise using a per-cell weighted
    quantile.
    """
    if not matches:
        raise ValueError("analog_outcome_distribution: matches is empty")

    weights_by_year = _compute_weights(matches, weighting, softmax_tau)
    outcomes: dict[int, float | xr.Dataset | xr.DataArray] = {}
    for m in matches:
        outcomes[m.year] = outcome_fn(m.year)

    # Detect scalar vs xr outcome
    first = outcomes[matches[0].year]
    is_scalar = isinstance(first, (int, float, np.floating))

    if is_scalar:
        vals = np.array([float(outcomes[m.year]) for m in matches], dtype=np.float64)
        ws = np.array([weights_by_year[m.year] for m in matches], dtype=np.float64)
        pcts = _percentiles_scalar(vals, ws)
        boot = _bootstrap_ci_scalar(vals, ws, n=int(bootstrap), seed=int(seed)) \
            if bootstrap > 0 else None
        percentiles: dict = pcts
        boot_ci: dict | None = boot
    else:
        # xr outcomes — per-cell weighted quantile
        # Stack per year into a common 4-D array (year × dims)
        arr_stack = None
        template = None
        for m in matches:
            o = outcomes[m.year]
            if isinstance(o, xr.Dataset):
                # Take the first data_var — caller decides what to look
                # at; for multi-var, they should call this per-var.
                var = list(o.data_vars)[0]
                d = o[var]
            elif isinstance(o, xr.DataArray):
                d = o
            else:
                raise TypeError(f"unsupported outcome type for year {m.year}: {type(o)}")
            if template is None:
                template = d
                arr_stack = np.empty((len(matches),) + d.shape, dtype=np.float64)
            arr_stack[matches.index(m)] = np.asarray(d.values, dtype=np.float64)
        ws = np.array([weights_by_year[m.year] for m in matches], dtype=np.float64)
        # Flatten (H*W) → per-cell weighted quantile
        flat = arr_stack.reshape(len(matches), -1)
        q_out = {"q10": np.empty(flat.shape[1]),
                  "q50": np.empty(flat.shape[1]),
                  "q90": np.empty(flat.shape[1])}
        for j in range(flat.shape[1]):
            col = flat[:, j]
            mask = np.isfinite(col)
            if not mask.any():
                for k in q_out:
                    q_out[k][j] = float("nan")
                continue
            q_out["q10"][j] = _weighted_quantile(col[mask], ws[mask], 0.10)
            q_out["q50"][j] = _weighted_quantile(col[mask], ws[mask], 0.50)
            q_out["q90"][j] = _weighted_quantile(col[mask], ws[mask], 0.90)
        # Reshape back and wrap as DataArrays
        percentiles = {}
        for k, arr in q_out.items():
            percentiles[k] = xr.DataArray(
                arr.reshape(template.shape),
                dims=template.dims, coords=template.coords,
                name=f"{template.name or 'outcome'}_{k}",
            )
        boot_ci = None            # xr bootstrap CI is expensive; deferred

    return AnalogOutcomeDist(
        matches=list(matches),
        outcome_by_year=outcomes,
        weights=weights_by_year,
        percentiles=percentiles,
        bootstrap_ci=boot_ci,
        weighting=weighting,
        softmax_tau=float(softmax_tau),
        seed=int(seed),
        provenance={
            "version": ANALOG_OUTCOMES_VERSION,
            "n_analogs": len(matches),
            "weighting": weighting,
            "softmax_tau": float(softmax_tau),
            "bootstrap": int(bootstrap),
            "seed": int(seed),
            "years": [m.year for m in matches],
            "qualities": [m.quality for m in matches],
        },
    )
