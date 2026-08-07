"""
whatif.economics.npv — adaptation net-present-value + cost-benefit ratio.

Primary sources:
    * NITI Aayog / former Planning Commission, "Manual for Economic
      Appraisal of Public Investment Projects" — social discount rate
      convention. Central 8 %; sensitivity range 7–12 %.
    * RBI / DGE&S Consumer Price Index for Agricultural Labourers
      (CPI-AL) — used to deflate INR to a stated reference year (real
      terms). Cited per pipeline.
    * World Bank 2010 "Cost of Adapting to Climate Change" — framing
      for NPV/BCR reporting in adaptation appraisals.

Contract:
    * Every NPV output ships at THREE discount rates: 7 %, 8 % (central),
      12 %. Rule 7: no bare rate.
    * All values in real INR of a stated reference year. If a scenario
      requests a different year, the pipeline deflates through CPI-AL.
    * Determinism: fixed sort order over adaptation ids; no RNG.
    * Cited-cost gate: an option with ``cost_complete=False`` selected
      in the run raises :class:`MissingCostCitation`.

Version: ``npv-v1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..sectors.adaptations import (
    ADAPTATIONS_VERSION,
    AdaptationOption,
    MissingCostCitation,
    load_adaptation,
    registry_sha256,
    registry_version,
)

NPV_VERSION = "npv-v1"

# NITI Aayog convention: central 8 %; range 7 – 12 %. If any operator
# wants a different rate they must supply it explicitly (recorded in
# provenance).
DEFAULT_DISCOUNT_RATES: tuple[float, ...] = (0.07, 0.08, 0.12)
DEFAULT_HORIZON_YEARS = 30
DEFAULT_REFERENCE_INR_YEAR = 2025


def _pv_stream(cashflows: np.ndarray, rate: float) -> float:
    """Present value of an annual cashflow series at ``rate``.

    Convention: year 0 is undiscounted (capex is paid up-front);
    years 1..N-1 are discounted by (1+rate)^t.
    """
    r = float(rate)
    disc = 1.0 / np.power(1.0 + r, np.arange(len(cashflows)))
    return float(np.sum(cashflows * disc))


def _annual_delta_stream(
    annual_delta_inr_per_ha: float, horizon_years: int,
    option: AdaptationOption,
) -> np.ndarray:
    """Construct the annual net-benefit stream for one adaptation.

    Year 0: -capex.  Years 1..min(effective_years, horizon)-1: annual
    benefit − opex.  Beyond effective_years: 0 (option decommissioned).
    """
    horizon = int(horizon_years)
    cf = np.zeros(horizon, dtype=np.float64)
    cf[0] = -float(option.capex_inr_per_ha)
    opex = float(option.opex_inr_per_ha_per_yr)
    if option.opex_pct_of_capex_per_yr > 0:
        opex = opex + option.opex_pct_of_capex_per_yr * float(option.capex_inr_per_ha)
    active = min(int(option.effective_years), horizon - 1)
    for t in range(1, active + 1):
        cf[t] = float(annual_delta_inr_per_ha) - opex
    return cf


@dataclass
class NPVRow:
    adaptation_id: str
    common_name: str
    annual_delta_inr_per_ha: float
    npv_at_07: float
    npv_at_08: float
    npv_at_12: float
    bcr_at_08: float
    provenance: dict


def adaptation_npv(
    annual_delta_by_option: dict[str, float],
    *,
    discount_rates: tuple[float, ...] = DEFAULT_DISCOUNT_RATES,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    reference_inr_year: int = DEFAULT_REFERENCE_INR_YEAR,
    require_cited_costs: bool = True,
) -> pd.DataFrame:
    """Return per-adaptation NPV / BCR at every requested discount rate.

    Parameters
    ----------
    annual_delta_by_option :
        Mapping ``{adaptation_id: annual_net_benefit_INR_per_ha}`` — the
        expected annual ₹/ha uplift the adaptation delivers on top of
        the counterfactual (no-adaptation) baseline. Computed upstream
        by the Long-Term sector runner.
    discount_rates :
        Iterable of rates. Default 7 %, 8 %, 12 % per NITI Aayog.
    horizon_years :
        Cashflow horizon. Options with ``effective_years < horizon``
        are decommissioned at the effective-years mark and contribute
        0 thereafter.
    require_cited_costs :
        If True (default), options with ``cost_complete=False`` raise
        :class:`MissingCostCitation`. Set False only for exploratory
        runs (recorded in provenance).

    Returns
    -------
    pd.DataFrame sorted by NPV at 8 % descending. Columns include
    ``adaptation_id, common_name, annual_delta_inr_per_ha,
    NPV_07, NPV_08, NPV_12, BCR_08``.
    """
    rows: list[dict] = []
    blocked: list[str] = []
    for opt_id in sorted(annual_delta_by_option):
        try:
            opt = load_adaptation(opt_id)
        except KeyError as e:
            raise KeyError(f"unknown adaptation {opt_id!r}") from e
        if require_cited_costs and not opt.cost_complete:
            blocked.append(opt_id)
            continue

        delta = float(annual_delta_by_option[opt_id])
        cf = _annual_delta_stream(delta, horizon_years, opt)
        npvs: dict[str, float] = {}
        for r in discount_rates:
            pv = _pv_stream(cf, r)
            npvs[f"NPV_{int(r*100):02d}"] = pv

        # BCR at central rate 8 %
        r_central = 0.08 if 0.08 in discount_rates else discount_rates[len(discount_rates) // 2]
        active = min(int(opt.effective_years), horizon_years - 1)
        pv_benefits = _pv_stream(
            np.concatenate([[0.0], np.full(active, delta)]), r_central,
        )
        pv_opex = _pv_stream(
            np.concatenate([[0.0], np.full(active,
                float(opt.opex_inr_per_ha_per_yr)
                + float(opt.opex_pct_of_capex_per_yr) * float(opt.capex_inr_per_ha))]),
            r_central,
        )
        pv_costs = float(opt.capex_inr_per_ha) + pv_opex
        bcr = pv_benefits / pv_costs if pv_costs > 0 else float("nan")

        rows.append({
            "adaptation_id": opt_id,
            "common_name": opt.common_name,
            "annual_delta_inr_per_ha": delta,
            **npvs,
            "BCR_08": bcr,
            "effective_years": opt.effective_years,
            "capex_inr_per_ha": opt.capex_inr_per_ha,
            "capex_citation": opt.capex_citation,
        })

    if blocked and require_cited_costs:
        raise MissingCostCitation(
            f"Adaptation NPV refused: options with un-cited costs "
            f"selected: {blocked}. Ship a citation in adaptations.yaml "
            "or set require_cited_costs=False (recorded in provenance)."
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("NPV_08", ascending=False).reset_index(drop=True)
    df.attrs["npv_version"] = NPV_VERSION
    df.attrs["adaptations_version"] = ADAPTATIONS_VERSION
    df.attrs["adaptations_sha256"] = registry_sha256()
    df.attrs["discount_rates"] = list(discount_rates)
    df.attrs["reference_inr_year"] = int(reference_inr_year)
    df.attrs["horizon_years"] = int(horizon_years)
    return df
