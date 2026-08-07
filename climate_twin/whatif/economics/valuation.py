"""
whatif.economics.valuation — physical yield → rupees, three passes.

Primary sources:
    * CACP MSP notifications (Government of India) — the default price
      lookup lives in :mod:`prices` / ``prices.yaml``.
    * DES Cost of Cultivation of Principal Crops — cost anchors.
    * FAO-56/33 physical yield is upstream (Part 3).

Contract (Part 4 Golden rules):
    1. **No bare rupees.** Every net-revenue field is a
       ``{"q10","q50","q90"}`` dict. Constructing :class:`EconomicOutcome`
       with a scalar raises :class:`SchemaError`.
    2. **Every ₹ output ships with its climatology counterpart** — the
       ``baseline_net_inr_per_ha`` and ``delta_vs_baseline`` fields are
       mandatory.
    3. **Prices are a lever, not a fact.** The MSP season is passed in;
       defaulting silently to a specific year is disallowed.
    4. **Aggregation happens at district.** Grid-cell inputs raise
       :class:`ResolutionCeilingError`.

Formula:
    gross_inr_per_ha = Ya_t_per_ha × 10 × MSP_inr_per_qt
                        × (1 − moisture_discount − transport_pct)
    net_inr_per_ha   = gross − cost_of_cultivation_inr_per_ha

    (1 tonne = 10 quintals; MSP is quoted per quintal.)

Version: ``valuation-v1``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..sectors.agriculture import ResolutionCeilingError

VALUATION_VERSION = "valuation-v1"
_TONNE_TO_QUINTAL = 10.0


class SchemaError(TypeError):
    """Raised when :class:`EconomicOutcome` is constructed with malformed
    fields. Enforces Part-4 Rule 1 (no bare rupees)."""


def _validate_qdict(field_name: str, val: Any) -> dict[str, float]:
    """Validate that ``val`` is a dict with keys {q10, q50, q90} whose
    values are finite floats. Scalars are rejected."""
    if isinstance(val, (int, float, np.floating)):
        raise SchemaError(
            f"{field_name} is a scalar ({val!r}). Part-4 Rule 1: every "
            f"₹ output must be a dict with keys q10, q50, q90. Use "
            f"EconomicOutcome.point_estimate(x) to opt-in explicitly."
        )
    if not isinstance(val, dict):
        raise SchemaError(
            f"{field_name} must be dict, got {type(val).__name__}"
        )
    missing = {"q10", "q50", "q90"} - set(val)
    if missing:
        raise SchemaError(
            f"{field_name} missing quantile keys {sorted(missing)}"
        )
    for k in ("q10", "q50", "q90"):
        v = val[k]
        if not np.isfinite(v):
            raise SchemaError(f"{field_name}[{k}] is not finite: {v}")
    return {k: float(val[k]) for k in ("q10", "q50", "q90")}


@dataclass
class EconomicOutcome:
    """L4 outcome for one decision × one climate state.

    All ₹/ha fields are three-quantile dicts. The class refuses
    construction with a scalar in any of them.
    """
    crop: str
    season: str
    region_kind: str
    region_id: str
    gross_revenue_inr_per_ha: dict[str, float]
    cost_inr_per_ha: dict[str, float]
    net_revenue_inr_per_ha: dict[str, float]
    baseline_net_inr_per_ha: dict[str, float]
    delta_vs_baseline: dict[str, float]
    price_source: str                              # "msp" | "mandi"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce quantile-dict shape on every ₹ field
        self.gross_revenue_inr_per_ha = _validate_qdict(
            "gross_revenue_inr_per_ha", self.gross_revenue_inr_per_ha
        )
        self.cost_inr_per_ha = _validate_qdict(
            "cost_inr_per_ha", self.cost_inr_per_ha
        )
        self.net_revenue_inr_per_ha = _validate_qdict(
            "net_revenue_inr_per_ha", self.net_revenue_inr_per_ha
        )
        self.baseline_net_inr_per_ha = _validate_qdict(
            "baseline_net_inr_per_ha", self.baseline_net_inr_per_ha
        )
        # delta = net - baseline_net, per quantile — recompute so the
        # invariant can't drift under user mutation.
        self.delta_vs_baseline = {
            k: float(self.net_revenue_inr_per_ha[k]
                     - self.baseline_net_inr_per_ha[k])
            for k in ("q10", "q50", "q90")
        }
        if self.price_source not in ("msp", "mandi"):
            raise SchemaError(
                f"price_source must be 'msp' or 'mandi', got {self.price_source!r}"
            )

    @classmethod
    def point_estimate(cls, x: float) -> dict[str, float]:
        """Explicit opt-in for a point estimate — same value at all three
        quantiles. Use only when a downstream test needs a degenerate
        distribution; production runs must feed genuine q10/q50/q90."""
        return {"q10": float(x), "q50": float(x), "q90": float(x)}

    def summary(self) -> dict[str, Any]:
        """Flat dict for logging / provenance."""
        return {
            "crop": self.crop,
            "season": self.season,
            "region": f"{self.region_kind}:{self.region_id}",
            "net_p50": self.net_revenue_inr_per_ha["q50"],
            "net_p10": self.net_revenue_inr_per_ha["q10"],
            "net_p90": self.net_revenue_inr_per_ha["q90"],
            "delta_p50": self.delta_vs_baseline["q50"],
            "price_source": self.price_source,
        }


# ── Yield → rupees helpers ──────────────────────────────────────────
def _reject_grid_input(y: Any, arg_name: str) -> None:
    """Refuse xr objects that still carry (lat, lon) — Part-4 Rule 5:
    aggregate at district *before* valuation."""
    dims = None
    if isinstance(y, (xr.Dataset, xr.DataArray)):
        dims = set(y.dims)
    if dims and ("lat" in dims or "lon" in dims):
        raise ResolutionCeilingError(
            f"{arg_name} still has grid-cell dims {sorted(dims)}. "
            "Aggregate to district (via sectors.to_district) BEFORE "
            "valuation. Part-4 Rule 5: no village-level rupees."
        )


def _ya_qdict(yields: dict[str, float | dict]) -> dict[str, float]:
    """Coerce a yields spec to a q10/q50/q90 dict of Ya (t/ha).

    Accepts either
      * ``{"q10": 3.1, "q50": 3.8, "q90": 4.4}``   (three-pass output), OR
      * ``{"district": <name>, "Ya_q10": ..., "Ya_q50": ..., "Ya_q90": ...}``
        (single-district row from Part 3's sowing-window optimiser).
    """
    if all(k in yields for k in ("q10", "q50", "q90")):
        out = {k: float(yields[k]) for k in ("q10", "q50", "q90")}
    elif all(k in yields for k in ("Ya_q10", "Ya_q50", "Ya_q90")):
        out = {"q10": float(yields["Ya_q10"]),
                "q50": float(yields["Ya_q50"]),
                "q90": float(yields["Ya_q90"])}
    else:
        raise ValueError(
            "yields dict must supply q10/q50/q90 (or Ya_q10/50/90); "
            f"got keys {sorted(yields)}"
        )
    for k, v in out.items():
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"yield[{k}] = {v} is non-finite or negative")
    return out


def value_agriculture(
    yield_qdict: dict[str, float] | dict[str, Any],
    baseline_ya_t_ha: float,
    *,
    crop,
    price_set,
    season: str,
    region_kind: str = "district",
    region_id: str = "",
    use_mandi_prices: bool = False,
    yield_ds_for_shape_check: Any = None,
) -> EconomicOutcome:
    """Convert district-level three-pass yield into an :class:`EconomicOutcome`.

    Parameters
    ----------
    yield_qdict :
        District-level yield in t/ha, keyed q10/q50/q90 (or the Part-3
        sowing-window row's Ya_q10/50/90 form).
    baseline_ya_t_ha :
        District-level climatology-baseline yield in t/ha (single value —
        the baseline is deterministic).
    crop :
        :class:`whatif.sectors.crops.Crop`.
    price_set :
        :class:`whatif.economics.prices.PriceSet`.
    season :
        MSP season key, e.g., ``"2024-25"``. Required — no silent
        fallback (Part-4 Rule 2).
    region_kind, region_id :
        For provenance labelling. ``region_kind="district"`` is the
        expected value; anything with ``"grid"`` in it raises.
    use_mandi_prices :
        Set True to swap MSP for live mandi prices. Not implemented
        (Part 7) — raises NotImplementedError on True today.
    yield_ds_for_shape_check :
        Optional xr Dataset to trip :func:`_reject_grid_input`. Callers
        that go straight to the district DataFrame don't need this.
    """
    if yield_ds_for_shape_check is not None:
        _reject_grid_input(yield_ds_for_shape_check, "yield_ds_for_shape_check")
    if "grid" in region_kind.lower() or region_kind == "cell":
        raise ResolutionCeilingError(
            f"region_kind={region_kind!r} is finer than district — Rule 5"
        )

    if use_mandi_prices:
        raise NotImplementedError(
            "use_mandi_prices=True lands in Part 7. Set to False (default) "
            "to run on MSP from prices.yaml."
        )

    ya_q = _ya_qdict(yield_qdict)
    msp = price_set.msp_for(season)                        # ₹/qt
    discount = (
        float(price_set.moisture_discount_pct)
        + float(price_set.transport_marketing_pct)
    )
    cost = float(price_set.cost_of_cultivation_inr_per_ha)

    def _gross(ya: float) -> float:
        return ya * _TONNE_TO_QUINTAL * msp * (1.0 - discount)

    gross = {k: _gross(v) for k, v in ya_q.items()}
    cost_q = {k: cost for k in ("q10", "q50", "q90")}
    net = {k: gross[k] - cost for k in ("q10", "q50", "q90")}
    baseline_gross = _gross(float(baseline_ya_t_ha))
    baseline_net = baseline_gross - cost
    baseline_q = {k: baseline_net for k in ("q10", "q50", "q90")}

    prov = {
        "valuation_version": VALUATION_VERSION,
        "prices_registry_version": price_set.registry_version,
        "prices_registry_sha256": price_set.registry_sha256,
        "crop_registry_version": crop.registry_version,
        "crop_registry_sha256": crop.registry_sha256,
        "msp_inr_per_qt": msp,
        "msp_season": season,
        "moisture_discount_pct": price_set.moisture_discount_pct,
        "transport_marketing_pct": price_set.transport_marketing_pct,
        "cost_of_cultivation_inr_per_ha": cost,
        "price_source": "msp",
    }

    return EconomicOutcome(
        crop=crop.key,
        season=season,
        region_kind=region_kind,
        region_id=region_id,
        gross_revenue_inr_per_ha=gross,
        cost_inr_per_ha=cost_q,
        net_revenue_inr_per_ha=net,
        baseline_net_inr_per_ha=baseline_q,
        delta_vs_baseline={},   # recomputed in __post_init__
        price_source="msp",
        provenance=prov,
    )
