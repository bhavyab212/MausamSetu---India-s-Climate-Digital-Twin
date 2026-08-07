"""
whatif.economics.sensitivity — one-factor-at-a-time tornado.

For every ``EconomicOutcome`` the engine emits, the honest question is
which input moved the number the most. This module answers it via a
sorted knob-ranking that the UI renders as a tornado plot.

Primary source: standard one-factor-at-a-time (OFAT) sensitivity —
see Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer",
§1.2 for the classic tornado protocol and its documented weakness
(does not capture interaction effects). We ship OFAT because it's the
right level of detail for a district officer's dashboard; Sobol'
indices are a Part 7 add-on if the risk group asks.

Contract:
    * ``tornado(scenario_fn, knobs, delta_pct)`` calls ``scenario_fn``
      once per knob × direction (up/down), records the resulting
      ``net_p50`` change, and returns a sorted table.
    * With ``delta_pct=0``, every range is 0 — used as a determinism
      / sanity test.
    * Every entry carries the knob's ``units`` so the UI can print
      "Assumption {X} accounts for {Y}% of the range in net revenue."
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .valuation import EconomicOutcome


@dataclass(frozen=True)
class Knob:
    """A single input the tornado will nudge.

    ``kind`` is one of "pct" or "abs":
        * "pct" → nudge by ± delta_pct as a fraction of the current value.
        * "abs" → nudge by ± absolute_delta in the input's own units.

    ``apply`` receives the base ``levers`` dict and a signed delta,
    returns a mutated copy the scenario_fn will run with.
    """
    knob_id: str
    label: str
    kind: str                                     # "pct" | "abs"
    units: str
    absolute_delta: float = 0.0
    apply: Callable[[dict, float], dict] = None   # (levers, signed_delta) → new levers


# ── Default knobs for the agriculture scenario ───────────────────────
def _set(levers: dict, key_path: tuple[str, ...], signed: float) -> dict:
    """Write ``signed`` at ``key_path`` in a deep-copy of ``levers``.

    The sector runner interprets ``overrides[<key>]`` as an additive
    bias (for _pct-style keys the bias is a fractional multiplier
    delta, e.g., 0.20 = +20 %; for _abs-style keys the bias is in
    the key's own units, e.g., °C, days).
    """
    d = copy.deepcopy(levers)
    cur = d
    for k in key_path[:-1]:
        cur = cur.setdefault(k, {})
    cur[key_path[-1]] = float(signed)
    return d


DEFAULT_KNOBS: tuple[Knob, ...] = (
    Knob("msp",  "MSP (₹/qt)",              "pct", "₹/qt",
          apply=lambda l, s: _set(l, ("overrides", "msp_pct"), s)),
    Knob("cost", "Cost of cultivation",     "pct", "₹/ha",
          apply=lambda l, s: _set(l, ("overrides", "cost_pct"), s)),
    Knob("ymax", "Ymax (t/ha)",             "pct", "t/ha",
          apply=lambda l, s: _set(l, ("overrides", "ymax_pct"), s)),
    Knob("ky",   "Ky yield-response",       "pct", "1",
          apply=lambda l, s: _set(l, ("overrides", "ky_pct"), s)),
    Knob("kc",   "Kc (mid)",                "pct", "1",
          apply=lambda l, s: _set(l, ("overrides", "kc_mid_pct"), s)),
    Knob("awc",  "AWC (mm/m)",              "pct", "mm/m",
          apply=lambda l, s: _set(l, ("overrides", "awc_pct"), s)),
    Knob("rain", "Rainfall driver bias",    "pct", "mm",
          apply=lambda l, s: _set(l, ("overrides", "rain_pct"), s)),
    Knob("tmax", "Tmax driver bias",        "abs", "°C", absolute_delta=1.0,
          apply=lambda l, s: _set(l, ("overrides", "tmax_shift_c"), s)),
    Knob("sow",  "Sow date shift",          "abs", "days", absolute_delta=7.0,
          apply=lambda l, s: _set(l, ("overrides", "sow_shift_days"), s)),
)


@dataclass
class TornadoResult:
    net_p50_base: float                              # ₹/ha, base case
    rows: pd.DataFrame                               # sorted knob table
    dominant_knob: str
    dominant_share_pct: float

    def sentence(self) -> str:
        return (
            f"Assumption {self.dominant_knob!r} accounts for "
            f"{self.dominant_share_pct:.0f}% of the range in net revenue "
            "(one-factor-at-a-time sensitivity)."
        )


def tornado(
    scenario_fn: Callable[[dict], EconomicOutcome],
    base_levers: dict | None = None,
    knobs: tuple[Knob, ...] = DEFAULT_KNOBS,
    *,
    delta_pct: float = 0.20,
) -> TornadoResult:
    """Run OFAT sensitivity.

    ``scenario_fn`` takes a ``levers`` dict (may include an
    ``"overrides"`` sub-dict that the sector runner reads and applies
    as bias to prices / driver / crop params) and returns an
    :class:`EconomicOutcome`.
    """
    base_levers = dict(base_levers or {})
    base = scenario_fn(base_levers)
    base_p50 = float(base.net_revenue_inr_per_ha["q50"])

    rows: list[dict] = []
    for k in knobs:
        if k.kind == "pct":
            signed_up = float(delta_pct)
            signed_dn = -float(delta_pct)
        elif k.kind == "abs":
            signed_up = float(k.absolute_delta) * (0.0 if delta_pct == 0 else 1.0)
            signed_dn = -signed_up
        else:
            raise ValueError(f"unknown knob.kind={k.kind!r}")

        up_levers = k.apply(base_levers, signed_up)
        dn_levers = k.apply(base_levers, signed_dn)
        up_out = scenario_fn(up_levers)
        dn_out = scenario_fn(dn_levers)
        up = float(up_out.net_revenue_inr_per_ha["q50"])
        dn = float(dn_out.net_revenue_inr_per_ha["q50"])
        rows.append({
            "knob": k.knob_id,
            "label": k.label,
            "units": k.units,
            "kind": k.kind,
            "delta_signed": signed_up,
            "net_p50_up": up,
            "net_p50_dn": dn,
            "range": abs(up - dn),
            "signed_range": up - dn,
        })

    df = pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
    total_range = float(df["range"].sum())
    if total_range > 0:
        share = float(df["range"].iloc[0] / total_range) * 100.0
        dominant = str(df["label"].iloc[0])
    else:
        share = 0.0
        dominant = "none (no movement — check delta_pct or scenario_fn)"

    return TornadoResult(
        net_p50_base=base_p50,
        rows=df,
        dominant_knob=dominant,
        dominant_share_pct=share,
    )
