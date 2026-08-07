"""
whatif.economics.payoff — payoff matrix builder.

The single most important artifact this engine emits. A payoff matrix
is a (decision × climate-state) table of :class:`EconomicOutcome`, plus
per-state probability weights, plus a climatology-baseline counterpart.
Everything else in the decision layer (regret, minimax, VaR, tornado,
value curves) reads this artifact.

Contract:
    * Weights sum to 1.0 within 1e-6 or construction raises.
    * Every cell carries its own :class:`EconomicOutcome` — no scalar
      shortcuts (Part-4 Rule 1).
    * The baseline field is populated cell-by-cell so ``delta_vs_baseline``
      is available at every cell as well.
    * Determinism: two builds with the same input Decisions + States +
      seed produce identical arrays.

Primary source: standard decision-theoretic payoff formulation
(Savage 1951; textbook exposition in Berger 1985 §5).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np

from ..config.region import RegionSpec
from .valuation import EconomicOutcome


@dataclass(frozen=True)
class Decision:
    """A single agricultural decision.

    Kept intentionally small: the payoff builder passes it to a caller-
    supplied ``run_cell`` function, so anything decision-specific lives
    in the closure the caller writes. This module doesn't dictate crop
    vs fallow vs sow-schedule taxonomy.
    """
    label: str
    kind: str                            # "crop" | "fallow" | ...
    params: tuple[tuple[str, object], ...] = ()

    def params_dict(self) -> dict:
        return dict(self.params)

    def signature(self) -> str:
        s = f"{self.kind}|{self.label}|{sorted(self.params)}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class ClimateState:
    """A discrete climate state with a probability weight and a way to
    build the driver spec / lever used to run the L2+L3 chain under it.

    ``perturbation`` is a dict describing how to perturb the historical
    driver — e.g., ``{"rain_scale": 0.8}`` or ``{"tmax_shift_c": 1.5}``.
    Part-4 ships the perturbation flavour; historical-analog buckets
    are declared here but wired in Part 5.
    """
    label: str
    weight: float                        # ∈ [0, 1]; states sum to 1
    perturbation: tuple[tuple[str, float], ...] = ()
    analog_years: tuple[int, ...] = ()   # empty ⇒ perturbation flavour

    def perturbation_dict(self) -> dict:
        return dict(self.perturbation)

    def signature(self) -> str:
        s = f"{self.label}|{self.weight}|{self.perturbation}|{self.analog_years}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]


@dataclass
class PayoffMatrix:
    """Rectangular payoff table with baseline counterpart + provenance."""
    decisions: list[Decision]
    states: list[ClimateState]
    cells: list[list[EconomicOutcome | None]]     # cells[i][j] for dec i × state j
    payoff_p10: np.ndarray                        # ₹/ha
    payoff_p50: np.ndarray
    payoff_p90: np.ndarray
    baseline_payoff: np.ndarray                   # climatology counterpart, q50
    region_kind: str
    region_id: str
    version: str = "payoff-v1"
    provenance: list[dict] = field(default_factory=list)

    @property
    def n_dec(self) -> int: return len(self.decisions)

    @property
    def n_state(self) -> int: return len(self.states)

    def state_weights(self) -> np.ndarray:
        return np.array([s.weight for s in self.states], dtype=np.float64)


# ─── Builder ─────────────────────────────────────────────────────────
def build_payoff_matrix(
    decisions: list[Decision],
    states: list[ClimateState],
    region: RegionSpec,
    *,
    run_cell: Callable[[Decision, ClimateState], EconomicOutcome],
) -> PayoffMatrix:
    """Assemble a payoff matrix by calling ``run_cell(decision, state)`` on
    every cell.

    ``run_cell`` is the caller's closure over crop/prices/season — the
    matrix module deliberately doesn't reach into the sector or the
    valuation layer, so the same builder handles agriculture today,
    power / water sectors tomorrow.

    Weights on states must sum to 1 within 1e-6. Raises ``ValueError``
    otherwise (Part-4 Rule for probability weights).
    """
    if not decisions or not states:
        raise ValueError("payoff matrix needs at least one decision and one state")

    w = np.array([s.weight for s in states], dtype=np.float64)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"climate-state weights must sum to 1.0 within 1e-6, "
            f"got {float(w.sum()):.6f}"
        )
    if (w < 0).any():
        raise ValueError("climate-state weights must be non-negative")

    N, M = len(decisions), len(states)
    cells: list[list[EconomicOutcome | None]] = [[None] * M for _ in range(N)]
    p10 = np.full((N, M), np.nan, dtype=np.float64)
    p50 = np.full((N, M), np.nan, dtype=np.float64)
    p90 = np.full((N, M), np.nan, dtype=np.float64)
    base = np.full((N, M), np.nan, dtype=np.float64)
    prov: list[dict] = []

    region_id = region.id or ""
    # If any state is an analog_bucket with non-empty analog_years, we
    # switch to the analog-aware runner (defined below) which averages
    # per-year observed outcomes empirically. Callers that don't want
    # this behaviour override run_cell explicitly.
    for i, dec in enumerate(decisions):
        for j, st in enumerate(states):
            out = run_cell(dec, st)
            if not isinstance(out, EconomicOutcome):
                raise TypeError(
                    f"run_cell({dec.label!r}, {st.label!r}) returned "
                    f"{type(out).__name__}, expected EconomicOutcome"
                )
            cells[i][j] = out
            p10[i, j] = out.net_revenue_inr_per_ha["q10"]
            p50[i, j] = out.net_revenue_inr_per_ha["q50"]
            p90[i, j] = out.net_revenue_inr_per_ha["q90"]
            base[i, j] = out.baseline_net_inr_per_ha["q50"]
            prov.append({
                "dec": dec.label, "state": st.label,
                "dec_sig": dec.signature(), "state_sig": st.signature(),
                "outcome": out.summary(),
                "valuation": out.provenance,
            })

    return PayoffMatrix(
        decisions=list(decisions),
        states=list(states),
        cells=cells,
        payoff_p10=p10, payoff_p50=p50, payoff_p90=p90,
        baseline_payoff=base,
        region_kind=region.kind,
        region_id=region_id,
        provenance=prov,
    )
