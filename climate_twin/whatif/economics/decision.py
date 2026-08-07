"""
whatif.economics.decision — decision rules over a PayoffMatrix.

Primary sources:
    * Savage, L.J. (1951) "The theory of statistical decision", JASA 46,
      55-67 — minimax regret.
    * Rockafellar & Uryasev (2000) "Optimization of conditional value-
      at-risk", J. Risk 2:21-41 — CVaR.
    * Hadar & Russell (1969) "Rules for ordering uncertain prospects",
      AER 59:25-34 — stochastic dominance.

Every function here consumes a :class:`PayoffMatrix` from
:mod:`whatif.economics.payoff`; nothing here dips back into the
scenario runner.
"""
from __future__ import annotations

import numpy as np

from .payoff import PayoffMatrix


# ── Expected value ───────────────────────────────────────────────────
def expected_value(pm: PayoffMatrix) -> np.ndarray:
    """Probability-weighted expected net revenue per decision, at q50."""
    w = pm.state_weights()
    return pm.payoff_p50 @ w                       # (N_dec,)


def expected_value_with_uncertainty(pm: PayoffMatrix) -> dict[str, np.ndarray]:
    """Same expected-value calc across all three quantile matrices.
    The UI displays the resulting EV as a q10/q50/q90 band."""
    w = pm.state_weights()
    return {
        "q10": pm.payoff_p10 @ w,
        "q50": pm.payoff_p50 @ w,
        "q90": pm.payoff_p90 @ w,
    }


# ── Regret ───────────────────────────────────────────────────────────
def regret_matrix(pm: PayoffMatrix) -> np.ndarray:
    """Savage regret: ``regret[i, j] = max_k pm[k, j] - pm[i, j]``.

    Property: ``regret[i, j] >= 0`` everywhere, and for every column j
    there is at least one row i* with regret == 0 (the best decision
    against that state has zero regret against itself).
    """
    per_state_max = pm.payoff_p50.max(axis=0, keepdims=True)  # (1, M)
    return per_state_max - pm.payoff_p50                       # (N, M)


def minimax_regret(pm: PayoffMatrix) -> int:
    """Return the index of the decision that minimises the maximum regret."""
    r = regret_matrix(pm)
    return int(np.argmin(r.max(axis=1)))


# ── Worst-case / robust ──────────────────────────────────────────────
def worst_case(pm: PayoffMatrix) -> np.ndarray:
    """Worst payoff each decision faces across the state grid, at q50."""
    return pm.payoff_p50.min(axis=1)


# ── Value-at-risk / CVaR ─────────────────────────────────────────────
def var_cvar(
    pm: PayoffMatrix, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted VaR and CVaR of the payoff distribution per decision.

    * ``VaR_alpha`` = α-quantile of the payoff distribution over states
      (weights = state.weight). Negative losses (regrets) live in the
      left tail, so the α-quantile with small α returns the poor-outcome
      threshold.
    * ``CVaR_alpha`` = expected value conditional on payoff ≤ VaR_α.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    w = pm.state_weights()
    N = pm.n_dec
    var_arr = np.empty(N, dtype=np.float64)
    cvar_arr = np.empty(N, dtype=np.float64)

    for i in range(N):
        payoffs = pm.payoff_p50[i]
        order = np.argsort(payoffs)
        sorted_p = payoffs[order]
        sorted_w = w[order]
        cum_w = np.cumsum(sorted_w)
        # First index where cumulative weight crosses alpha
        idx = int(np.searchsorted(cum_w, alpha))
        idx = min(idx, len(sorted_p) - 1)
        var_arr[i] = sorted_p[idx]
        # CVaR = weighted mean of payoffs ≤ VaR
        mask = sorted_p <= sorted_p[idx]
        if mask.any() and sorted_w[mask].sum() > 0:
            cvar_arr[i] = float(np.sum(sorted_p[mask] * sorted_w[mask])
                                / sorted_w[mask].sum())
        else:
            cvar_arr[i] = sorted_p[idx]
    return var_arr, cvar_arr


# ── Stochastic dominance ─────────────────────────────────────────────
def stochastic_dominance(pm: PayoffMatrix) -> np.ndarray:
    """First-order stochastic dominance across states.

    ``dom[i, k] = True`` iff decision ``i`` weakly dominates ``k``: for
    every state, ``pm[i, j] >= pm[k, j]``, with strict inequality in at
    least one state. This is a strong, conservative statement; use for
    pruning obviously worse decisions before ranking.
    """
    N = pm.n_dec
    dom = np.zeros((N, N), dtype=bool)
    p = pm.payoff_p50
    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            row_i = p[i]
            row_k = p[k]
            if (row_i >= row_k).all() and (row_i > row_k).any():
                dom[i, k] = True
    return dom


# ── Combined recommendation ──────────────────────────────────────────
def recommend(pm: PayoffMatrix) -> dict:
    """Compact summary the UI can render as a recommendation card."""
    ev_bands = expected_value_with_uncertainty(pm)
    ev = ev_bands["q50"]
    wc = worst_case(pm)
    var_arr, cvar_arr = var_cvar(pm, alpha=0.10)
    reg = regret_matrix(pm)
    mm = minimax_regret(pm)

    i_star = int(np.argmax(ev))
    # Baseline EV: probability-weighted mean of the column-mean baseline
    # (which is the same value across rows since baseline is the
    # climatology counterpart for the cell, not the decision).
    baseline_ev = float(pm.baseline_payoff.mean(axis=0) @ pm.state_weights())
    delta = float(ev[i_star]) - baseline_ev

    return {
        "argmax_EV": i_star,
        "argmax_EV_label": pm.decisions[i_star].label,
        "EV_q50": float(ev[i_star]),
        "EV_q10": float(ev_bands["q10"][i_star]),
        "EV_q90": float(ev_bands["q90"][i_star]),
        "worst_case_q50": float(wc[i_star]),
        "VaR_10": float(var_arr[i_star]),
        "CVaR_10": float(cvar_arr[i_star]),
        "argmin_maxRegret": mm,
        "argmin_maxRegret_label": pm.decisions[mm].label,
        "maxRegret_at_argmin": float(reg[mm].max()),
        "delta_vs_baseline_EV": delta,
        "baseline_EV": baseline_ev,
    }
