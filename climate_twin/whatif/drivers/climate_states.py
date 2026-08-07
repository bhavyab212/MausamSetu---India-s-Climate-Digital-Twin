"""
whatif.drivers.climate_states — state-builders feeding the payoff matrix.

Two flavours; both return ``list[ClimateState]`` for
``economics.payoff.build_payoff_matrix``:

    * :func:`climate_states_from_perturbation` — one state per delta
      perturbation. Physically-inconsistent by construction; the UI
      shows the mandatory caveat next to any output.
    * :func:`climate_states_from_analogs`      — analog-derived states.
      Two shapes:  ``per_year`` (one state per analog, weight from
      inv-distance / softmax) and ``tercile`` (buckets over a chosen
      feature; bucket boundaries anchor on TRAIN_YEARS, never on the
      analog subsample — otherwise the buckets shrink to nothing).

Version: ``state-builder-v1``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ..economics.payoff import ClimateState
from ..indices.reference import TRAIN_YEARS
from .analog_features import AnalogSpec, build_feature_matrix
from .analog_outcomes import _compute_weights
from .analogs import AnalogMatch
from .perturbation import PerturbationSpec

STATE_BUILDER_VERSION = "state-builder-v1"


# ─── Flavour 1: perturbation ─────────────────────────────────────────
def climate_states_from_perturbation(
    perturbations: list[PerturbationSpec],
    weights: list[float],
    labels: list[str] | None = None,
) -> list[ClimateState]:
    """Return one ClimateState per perturbation.

    Weights must sum to 1.0 within 1e-6; otherwise raises. Labels
    default to "rain × {rain_scale:.2f} · Δtmax {tmax_shift_c:+.1f}°C".
    """
    if len(perturbations) != len(weights):
        raise ValueError(
            f"perturbations and weights must be same length: "
            f"{len(perturbations)} vs {len(weights)}"
        )
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(
            f"weights must sum to 1.0 within 1e-6, got {float(sum(weights)):.6f}"
        )

    labels = labels or [
        f"rain × {p.rain_scale:.2f} · Δtmax {p.tmax_shift_c:+.1f}°C"
        for p in perturbations
    ]
    out: list[ClimateState] = []
    for pert, w, label in zip(perturbations, weights, labels):
        # ClimateState.perturbation carries a serialised form the sector
        # runner reads under ``overrides``. Only rain_scale +
        # tmax_shift_c + tmin_shift_c are wired into the sector runner
        # today; other fields would land in provenance only.
        pkey = (
            ("rain_scale", float(pert.rain_scale)),
            ("tmax_shift_c", float(pert.tmax_shift_c)),
            ("tmin_shift_c", float(pert.tmin_shift_c)),
            ("_kind", "perturbation"),
            ("_state_builder", STATE_BUILDER_VERSION),
        )
        out.append(ClimateState(
            label=label, weight=float(w), perturbation=pkey,
        ))
    return out


# ─── Flavour 2: analogs ──────────────────────────────────────────────
def _train_terciles(spec: AnalogSpec, bucket_variable: str) -> tuple[float, float]:
    """Return (t33, t66) tercile boundaries for ``bucket_variable``,
    computed on TRAIN_YEARS. Anchoring on TRAIN_YEARS (not on the
    analog subsample) is Part-5 Rule: bucket boundaries stay stable
    as the analog set shrinks."""
    fm = build_feature_matrix(
        spec, years=spec.train_years, use_cache=True,
    )
    if bucket_variable not in fm.columns:
        raise KeyError(
            f"bucket_variable {bucket_variable!r} not in features "
            f"{list(fm.columns)}"
        )
    tmask = (fm.index >= spec.train_years[0]) & (fm.index <= spec.train_years[1])
    train_vals = fm.loc[tmask, bucket_variable].to_numpy(dtype=np.float64)
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size < 3:
        raise ValueError(
            f"not enough train rows ({train_vals.size}) to compute terciles "
            f"of {bucket_variable}"
        )
    return (
        float(np.percentile(train_vals, 100.0 / 3.0)),
        float(np.percentile(train_vals, 200.0 / 3.0)),
    )


def _train_quartiles(spec: AnalogSpec, bucket_variable: str) -> tuple[float, float, float]:
    fm = build_feature_matrix(spec, years=spec.train_years, use_cache=True)
    tmask = (fm.index >= spec.train_years[0]) & (fm.index <= spec.train_years[1])
    v = fm.loc[tmask, bucket_variable].to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    return (
        float(np.percentile(v, 25.0)),
        float(np.percentile(v, 50.0)),
        float(np.percentile(v, 75.0)),
    )


def climate_states_from_analogs(
    matches: list[AnalogMatch],
    spec: AnalogSpec,
    *,
    bucketing: Literal["per_year", "tercile", "quartile"] = "tercile",
    weighting: Literal["uniform", "inv_distance", "softmax"] = "inv_distance",
    softmax_tau: float = 1.0,
    bucket_variable: str = "rain_total_std",
) -> list[ClimateState]:
    """Convert analog matches into ClimateStates for the payoff matrix.

    - ``per_year``  : one state per analog. Weight from ``weighting``.
                      A large matrix with narrative depth.
    - ``tercile``  : bucket analogs into low / normal / high on
                      ``bucket_variable``. Boundaries come from
                      TRAIN_YEARS terciles of that variable. State
                      weight = sum of within-bucket analog weights.
    - ``quartile`` : four buckets by TRAIN_YEARS quartiles.
    """
    if not matches:
        raise ValueError("climate_states_from_analogs: matches is empty")
    weights_by_year = _compute_weights(matches, weighting, softmax_tau)

    if bucketing == "per_year":
        return [
            ClimateState(
                label=f"analog {m.year} (d={m.distance:.2f}, {m.quality})",
                weight=float(weights_by_year[m.year]),
                perturbation=(
                    ("_kind", "analog_bucket"),
                    ("_state_builder", STATE_BUILDER_VERSION),
                    ("_weighting", weighting),
                ),
                analog_years=(int(m.year),),
            )
            for m in matches
        ]

    if bucketing == "tercile":
        t33, t66 = _train_terciles(spec, bucket_variable)
        buckets = {"low": [], "normal": [], "high": []}
        for m in matches:
            v = float(m.features_analog[bucket_variable])
            if v < t33:
                buckets["low"].append(m)
            elif v < t66:
                buckets["normal"].append(m)
            else:
                buckets["high"].append(m)
        labels = {
            "low":    f"low {bucket_variable} (< {t33:+.2f})",
            "normal": f"normal {bucket_variable} ({t33:+.2f} to {t66:+.2f})",
            "high":   f"high {bucket_variable} (≥ {t66:+.2f})",
        }
        out = []
        total_w = sum(weights_by_year[m.year] for m in matches)
        for bucket_key in ("low", "normal", "high"):
            bms = buckets[bucket_key]
            if not bms:
                # Empty bucket → 0 weight but still emit a state so the
                # payoff matrix has stable shape. Downstream code must
                # tolerate w=0 (already does — cells with 0 weight
                # contribute 0 to EV).
                out.append(ClimateState(
                    label=labels[bucket_key], weight=0.0,
                    perturbation=(
                        ("_kind", "analog_bucket"),
                        ("_state_builder", STATE_BUILDER_VERSION),
                        ("_bucketing", "tercile"),
                        ("_bucket_variable", bucket_variable),
                        ("_bucket_key", bucket_key),
                        ("_empty_bucket", 1),
                    ),
                    analog_years=(),
                ))
                continue
            bw = sum(weights_by_year[m.year] for m in bms) / max(total_w, 1e-12)
            out.append(ClimateState(
                label=labels[bucket_key], weight=float(bw),
                perturbation=(
                    ("_kind", "analog_bucket"),
                    ("_state_builder", STATE_BUILDER_VERSION),
                    ("_bucketing", "tercile"),
                    ("_bucket_variable", bucket_variable),
                    ("_bucket_key", bucket_key),
                    ("_weighting", weighting),
                ),
                analog_years=tuple(int(m.year) for m in bms),
            ))
        # Renormalise so weights sum to 1 exactly (bookkeeping)
        wsum = sum(s.weight for s in out)
        if wsum > 0:
            out = [replace(s, weight=s.weight / wsum) for s in out]
        return out

    if bucketing == "quartile":
        q25, q50, q75 = _train_quartiles(spec, bucket_variable)
        buckets = {"q1": [], "q2": [], "q3": [], "q4": []}
        for m in matches:
            v = float(m.features_analog[bucket_variable])
            if v < q25:   buckets["q1"].append(m)
            elif v < q50: buckets["q2"].append(m)
            elif v < q75: buckets["q3"].append(m)
            else:         buckets["q4"].append(m)
        total_w = sum(weights_by_year[m.year] for m in matches)
        out = []
        for key in ("q1", "q2", "q3", "q4"):
            bms = buckets[key]
            bw = sum(weights_by_year[m.year] for m in bms) / max(total_w, 1e-12)
            out.append(ClimateState(
                label=f"{key} {bucket_variable}",
                weight=float(bw),
                perturbation=(
                    ("_kind", "analog_bucket"),
                    ("_state_builder", STATE_BUILDER_VERSION),
                    ("_bucketing", "quartile"),
                    ("_bucket_variable", bucket_variable),
                    ("_bucket_key", key),
                ),
                analog_years=tuple(int(m.year) for m in bms),
            ))
        wsum = sum(s.weight for s in out)
        if wsum > 0:
            out = [replace(s, weight=s.weight / wsum) for s in out]
        return out

    raise ValueError(f"unknown bucketing {bucketing!r}")
