"""
whatif.economics.cost_loss — Murphy 1977 cost–loss + forecast value.

Primary sources:
    * Murphy, A.H. (1977). "The value of climatological, categorical,
      and probabilistic forecasts in the cost–loss ratio situation."
      Mon. Weather Rev. 105:803–816.
    * Richardson, D.S. (2000). "Skill and relative economic value of
      the ECMWF ensemble prediction system."  Q.J.R. Meteorol. Soc.
      126:649–667.  §2 rewrites Murphy's V in the ensemble era.
    * Brier, G.W. (1950).  Mon. Weather Rev. 78:1-3 — Brier score.

Setup:  a binary protective action.
    C = cost of protective action (₹/ha).
    L = loss if event occurs AND no action was taken (₹/ha).
    p_climatology = the base-rate probability of the event.

Three strategies and their mean-expense forms:
    * Perfect      — act iff event will occur.
    * Climatology  — act every year iff  p_clim > C/L.  (once-and-for-all)
    * Probabilistic forecast at threshold p* — act iff  p_forecast > p*.

Relative economic value (Murphy 1977 eq. 5; Richardson 2000 eq. 2):

    V(strategy) =  ME_climatology  −  ME_strategy
                  ─────────────────────────────────
                   ME_climatology  −  ME_perfect

Ranges from 0 (no better than climatology) to 1 (as good as perfect).

Version: ``cost-loss-murphy-1977-v1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

COST_LOSS_VERSION = "cost-loss-murphy-1977-v1"


# ─── Setup ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CostLossSetup:
    """One protective-action decision.

    ``event_definition`` is any 0/1-valued observed indicator over the
    verification cases (year, district, whatever). We keep it a
    ``pd.Series`` for testability instead of a callable — the backtest
    supplies the series each year.
    """
    action_id: str
    C_inr_per_ha: float
    L_inr_per_ha: float
    event_label: str
    citation: str
    complete: bool = True     # False ⇒ TODO cited-value ⇒ excluded from default runs

    def cost_loss_ratio(self) -> float:
        if self.L_inr_per_ha <= 0:
            raise ValueError("L must be positive")
        return float(self.C_inr_per_ha) / float(self.L_inr_per_ha)


# Shipped setups — every C, L is a defensible mid-range estimate cited
# in-line. Where a value is not yet cited from a primary source, the
# setup ships with ``complete=False`` and is excluded from the default
# backtest.  This is intentional: half-cited numbers must never bleed
# into a live comparison.
SHIPPED_SETUPS: dict[str, CostLossSetup] = {
    "sow_delay_10d_if_late_onset": CostLossSetup(
        action_id="sow_delay_10d_if_late_onset",
        C_inr_per_ha=3000.0,       # incremental replant / delayed field-prep cost
        L_inr_per_ha=25000.0,      # partial-season failure loss (~40% of paddy Ymax net)
        event_label="monsoon_onset_delay_gt_10d",
        citation=(
            "C: ICAR-CRIDA weekly agromet advisory; incremental cost of "
            "10-day sowing delay for kharif paddy. "
            "L: DES-CoC net revenue implied by a 40% partial-season shortfall."
        ),
    ),
    "preventive_irrigation_if_dry_spell_forecast": CostLossSetup(
        action_id="preventive_irrigation_if_dry_spell_forecast",
        C_inr_per_ha=1800.0,       # one supplemental irrigation event
        L_inr_per_ha=12000.0,      # yield shortfall from a 10-day dry spell during flowering
        event_label="dry_spell_ge_10d_at_flowering",
        citation=(
            "C: ICAR-Directorate of Water Management; energy + water "
            "charge for one supplemental irrigation cycle, 5 ha basis. "
            "L: FAO-33 Ky-implied shortfall for a 10-day dry spell at "
            "flowering, Ymax anchor 4.5 t/ha."
        ),
    ),
    "fungicide_spray_if_humid_flowering": CostLossSetup(
        action_id="fungicide_spray_if_humid_flowering",
        C_inr_per_ha=1200.0,
        L_inr_per_ha=8000.0,
        event_label="humid_flowering_wk_ge_5d",
        citation=(
            "C: ICAR-NRRI recommended input rate + labour, single spray. "
            "L: yield loss under blast pressure per NRRI advisory "
            "(mid-range for kharif paddy)."
        ),
    ),
    "preposition_relief_if_extreme_10day_rainfall": CostLossSetup(
        action_id="preposition_relief_if_extreme_10day_rainfall",
        C_inr_per_ha=500.0,        # per-ha allocation of pre-positioning logistics
        L_inr_per_ha=0.0,          # loss context is disaster response, not yield
        event_label="rain_10day_ge_204mm",
        citation="NDMA disaster-response cost norm; L to be set by "
                  "Water/Disaster sector (Part 6). Excluded from default backtest.",
        complete=False,
    ),
}


# ─── Strategy expected-expense formulas (Murphy 1977) ───────────────
def _mean_expense_perfect(
    obs: np.ndarray, C: float, L: float,
) -> float:
    """Perfect strategy: act iff event occurs.
    ME_perfect = C * P(event) + 0 (no L incurred)."""
    p_event = float(obs.mean())
    return C * p_event


def _mean_expense_climatology(
    obs: np.ndarray, C: float, L: float, p_clim: float,
) -> float:
    """Climatology strategy: act every year iff  p_clim > C/L.
    If we always act: ME = C.  If we never act: ME = L · P(event)."""
    if p_clim > (C / L):
        return C
    p_event = float(obs.mean())
    return L * p_event


def _mean_expense_forecast(
    obs: np.ndarray, fcast: np.ndarray, C: float, L: float, threshold: float,
) -> float:
    """Probabilistic-forecast strategy at threshold ``p*``.
    Act iff fcast > p*. Then the case incurs C. Otherwise it incurs L*event."""
    act = fcast > threshold
    expenses = np.where(act, C, L * obs)
    return float(expenses.mean())


# ─── Verification metrics ────────────────────────────────────────────
def brier_score(fcast: np.ndarray, obs: np.ndarray) -> float:
    """BS = mean((f - o)^2). Perfect = 0, worst = 1."""
    fcast = np.asarray(fcast, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if fcast.shape != obs.shape:
        raise ValueError(f"shape mismatch: fcast {fcast.shape} vs obs {obs.shape}")
    return float(np.mean((fcast - obs) ** 2))


def brier_skill_score(
    fcast: np.ndarray, obs: np.ndarray, climatology_prob: float,
) -> float:
    """BSS = 1 - BS_forecast / BS_climatology. Positive ⇒ better than climatology."""
    bs_f = brier_score(fcast, obs)
    obs = np.asarray(obs, dtype=np.float64)
    bs_c = float(np.mean((climatology_prob - obs) ** 2))
    if bs_c <= 0:
        return float("nan")
    return 1.0 - bs_f / bs_c


def reliability_diagram(
    fcast: np.ndarray, obs: np.ndarray, bins: int = 10,
) -> pd.DataFrame:
    """Bin the forecast into ``bins`` equal-width buckets on [0,1],
    and for each bucket compute the mean forecast prob (x) and the
    observed frequency (y). Perfect calibration ⇒ y == x."""
    fcast = np.asarray(fcast, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(fcast, edges, right=False) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1],
                          "mean_forecast": float("nan"),
                          "observed_freq": float("nan"), "count": 0})
            continue
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "mean_forecast": float(fcast[mask].mean()),
            "observed_freq": float(obs[mask].mean()),
            "count": n,
        })
    return pd.DataFrame(rows)


def roc_auc(fcast: np.ndarray, obs: np.ndarray) -> float:
    """Simple Mann-Whitney-U-based AUC. Handles ties by averaging."""
    fcast = np.asarray(fcast, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.int32)
    pos = fcast[obs == 1]
    neg = fcast[obs == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-based
    all_vals = np.concatenate([pos, neg])
    order = np.argsort(all_vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_vals) + 1, dtype=np.float64)
    # Handle ties → average ranks
    df = pd.DataFrame({"v": all_vals, "r": ranks})
    df["r"] = df.groupby("v")["r"].transform("mean")
    ranks = df["r"].to_numpy()
    r_pos_sum = ranks[:pos.size].sum()
    U = r_pos_sum - pos.size * (pos.size + 1) / 2
    return float(U / (pos.size * neg.size))


def sharpness_histogram(fcast: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Distribution of forecast probabilities."""
    fcast = np.asarray(fcast, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(fcast, bins=edges)
    return pd.DataFrame({
        "bin_low": edges[:-1],
        "bin_high": edges[1:],
        "count": counts,
    })


# ─── Value curve ─────────────────────────────────────────────────────
@dataclass
class ValueCurve:
    """The V(p*) sweep across p* ∈ [0, 1], plus the verification bundle."""
    setup_id: str
    thresholds: np.ndarray
    V: np.ndarray                           # relative economic value
    ME_perfect: float
    ME_climatology: float
    best_threshold: float
    V_best: float
    brier: float
    brier_skill: float
    reliability: pd.DataFrame
    sharpness: pd.DataFrame
    roc_auc: float
    n_cases: int
    version: str = COST_LOSS_VERSION


def value_curve(
    setup: CostLossSetup,
    forecast_prob_series: pd.Series,
    observed_event_series: pd.Series,
    climatology_prob: float,
    *,
    n_thresholds: int = 21,
) -> ValueCurve:
    """Sweep ``V(p*)`` across ``p* ∈ [0, 1]`` and return the full bundle."""
    if not setup.complete:
        raise ValueError(
            f"setup {setup.action_id!r} is marked incomplete "
            "(complete=False); value_curve refuses to run it."
        )
    fcast = np.asarray(forecast_prob_series.to_numpy(), dtype=np.float64)
    obs = np.asarray(observed_event_series.to_numpy(), dtype=np.int32)
    if fcast.shape != obs.shape:
        raise ValueError(
            f"forecast/obs length mismatch: {fcast.shape} vs {obs.shape}"
        )
    if ((obs != 0) & (obs != 1)).any():
        raise ValueError("observed_event_series must be 0/1")
    C, L = float(setup.C_inr_per_ha), float(setup.L_inr_per_ha)

    ME_perfect = _mean_expense_perfect(obs, C, L)
    ME_clim = _mean_expense_climatology(obs, C, L, climatology_prob)
    denom = ME_clim - ME_perfect
    if denom <= 0:
        # Degenerate case: climatology already ≤ perfect → forecast can't
        # possibly help. Emit V ≡ 0 across all thresholds.
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        V = np.zeros_like(thresholds)
        best_t, V_best = float(thresholds[0]), 0.0
    else:
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        V = np.empty_like(thresholds)
        for k, p_star in enumerate(thresholds):
            ME_f = _mean_expense_forecast(obs, fcast, C, L, float(p_star))
            V[k] = (ME_clim - ME_f) / denom
        best_k = int(np.argmax(V))
        best_t = float(thresholds[best_k])
        V_best = float(V[best_k])

    bs = brier_score(fcast, obs)
    bss = brier_skill_score(fcast, obs, climatology_prob)
    return ValueCurve(
        setup_id=setup.action_id,
        thresholds=thresholds,
        V=V,
        ME_perfect=float(ME_perfect),
        ME_climatology=float(ME_clim),
        best_threshold=best_t,
        V_best=V_best,
        brier=bs,
        brier_skill=bss,
        reliability=reliability_diagram(fcast, obs, bins=10),
        sharpness=sharpness_histogram(fcast, bins=10),
        roc_auc=roc_auc(fcast, obs),
        n_cases=int(len(obs)),
    )
