"""
whatif.ui.engine_adapter — thin cached wrapper between the page and
the scenario engine.

The page never calls the sector or the driver modules directly. It
calls :func:`run_scenario_for_state` which:
    1. Builds Decisions + ClimateStates from the WhatIfState.
    2. Runs the full L2+L3+L4 chain via ``run_decision_scenario``.
    3. Attaches analog matches, tornado, backtest artifact.
    4. Returns a plain dict a panel can slice into.

Cache: keyed on ``state.cache_key()`` alone (see :class:`WhatIfState`).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd


def _build_decisions(state):
    from ..economics.payoff import Decision
    decs = []
    for crop in state.crops_selected:
        decs.append(Decision(
            label=f"{crop} sow {state.sow_dates[0]}",
            kind="crop",
            params=(("crop", crop), ("sow_date", state.sow_dates[0])),
        ))
    if state.include_fallow:
        decs.append(Decision(
            label="Fallow", kind="fallow",
            params=(("crop", "paddy_kharif"),),
        ))
    return decs


def _build_states_analog(state, region):
    """Analog-tercile states. Requires the analog pool to be buildable
    at the region+window; if not (e.g., cube not available), fall back
    to a synthetic 3-state perturbation grid so the UI is still
    responsive on cold-start."""
    from ..drivers.analog_features import AnalogSpec
    from ..drivers.analogs import (
        build_analog_pool,
        find_analogs,
        target_from_year,
    )
    from ..drivers.climate_states import climate_states_from_analogs

    spec = AnalogSpec(region=region, window="JJAS")
    try:
        excl = tuple(range(int(state.analog_target_year),
                            spec.train_years[1] + 20))
        pool = build_analog_pool(
            spec, exclude_years=excl, pool_years=spec.train_years,
        )
        if not pool.years():
            raise RuntimeError("empty pool")
        target = target_from_year(int(state.analog_target_year), spec)
        matches = find_analogs(target, pool, k=int(state.analog_k))
        states = climate_states_from_analogs(
            matches, spec,
            bucketing=state.analog_bucketing,
            weighting=state.analog_weighting,
        )
        return states, matches
    except Exception:
        return _build_states_perturbation_fallback(state), []


def _build_states_perturbation_fallback(state):
    from ..drivers.climate_states import climate_states_from_perturbation
    from ..drivers.perturbation import PerturbationSpec
    perts = [
        PerturbationSpec(rain_scale=0.8, caveat_acknowledged=True),
        PerturbationSpec(rain_scale=1.0, caveat_acknowledged=True),
        PerturbationSpec(rain_scale=1.2, caveat_acknowledged=True),
    ]
    return climate_states_from_perturbation(
        perts, weights=[0.28, 0.51, 0.21],
        labels=["low rain (fallback)", "normal (fallback)", "high rain (fallback)"],
    )


def _build_states_perturbation(state):
    from ..drivers.climate_states import climate_states_from_perturbation
    from ..drivers.perturbation import PerturbationSpec

    rs = float(state.perturbation_rain_scale)
    perts = [
        PerturbationSpec(
            rain_scale=max(rs - 0.2, 0.05),
            tmax_shift_c=float(state.perturbation_tmax_shift),
            tmin_shift_c=float(state.perturbation_tmin_shift),
            caveat_acknowledged=state.method_caveat_ack,
        ),
        PerturbationSpec(
            rain_scale=rs,
            tmax_shift_c=float(state.perturbation_tmax_shift),
            tmin_shift_c=float(state.perturbation_tmin_shift),
            caveat_acknowledged=state.method_caveat_ack,
        ),
        PerturbationSpec(
            rain_scale=min(rs + 0.2, 3.0),
            tmax_shift_c=float(state.perturbation_tmax_shift),
            tmin_shift_c=float(state.perturbation_tmin_shift),
            caveat_acknowledged=state.method_caveat_ack,
        ),
    ]
    return climate_states_from_perturbation(
        perts, weights=[0.28, 0.51, 0.21],
    )


def _region_from_state(state):
    from ..config.region import RegionSpec
    if state.region_kind == "all_india":
        return RegionSpec(kind="all_india")
    if state.region_kind == "bbox":
        # Default bbox for Vidarbha; the state stores just an id today
        return RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0),
                          id=state.region_id or "vidarbha")
    if state.region_kind == "subbasin":
        return RegionSpec(kind="subbasin", id=state.region_id or "cauvery")
    if state.region_kind == "zone":
        return RegionSpec(kind="zone", id=state.region_id)
    return RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0), id="vidarbha")


def _driver_spec_from_state(state, region):
    from ..drivers.driver import DriverSpec
    sow = date.fromisoformat(state.sow_dates[0])
    return DriverSpec(
        mode="historical", var="rain",
        dates=(sow.replace(month=6, day=1),
                sow.replace(month=11, day=30)),
        region=region,
    )


def _tornado_stub_from_result(scenario_dict) -> "TornadoResult":
    """Cheap tornado: perturb the payoff-matrix's argmax-EV net revenue
    with a set of scalar multipliers on Ya, price, cost. Runs the
    valuation formula directly on the pre-computed q50 rather than
    re-doing the whole L2 chain — the UI wants a fast tornado.
    """
    from ..economics.sensitivity import DEFAULT_KNOBS, TornadoResult
    from ..economics.valuation import EconomicOutcome

    pm = scenario_dict.get("payoff_matrix")
    rec = scenario_dict.get("recommendation") or {}
    if pm is None:
        return None
    argmax = int(rec.get("argmax_EV", 0))
    j50 = int(np.argmax(pm.payoff_p50[argmax]))
    base_cell = pm.cells[argmax][j50]
    if base_cell is None:
        return None
    base_p50 = float(base_cell.net_revenue_inr_per_ha["q50"])

    def _mock_out(delta_rupees_signed: float) -> EconomicOutcome:
        v = base_p50 + delta_rupees_signed
        return EconomicOutcome(
            crop=base_cell.crop, season=base_cell.season,
            region_kind="district", region_id=base_cell.region_id,
            gross_revenue_inr_per_ha={"q10": v, "q50": v, "q90": v},
            cost_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            net_revenue_inr_per_ha={"q10": v, "q50": v, "q90": v},
            baseline_net_inr_per_ha={"q10": base_p50, "q50": base_p50, "q90": base_p50},
            delta_vs_baseline={},
            price_source="msp",
        )

    # Cheap sensitivities: each knob shifts the cell by a fixed fraction
    # of the base value. These are illustrative — the "real" tornado
    # (Part 4 sensitivity.tornado) would re-run the sector runner.
    _SENSITIVITY_MULTIPLIER = {
        "msp":  1.00,        # linear in MSP
        "cost": 0.35,
        "ymax": 1.00,
        "ky":   0.15,
        "kc":   0.20,
        "awc":  0.25,
        "rain": 0.40,
        "tmax": 0.30,
        "sow":  0.10,
    }
    rows = []
    for k in DEFAULT_KNOBS:
        mult = _SENSITIVITY_MULTIPLIER.get(k.knob_id, 0.10)
        delta_r = 0.20 * base_p50 * mult
        up = base_p50 + delta_r
        dn = base_p50 - delta_r
        rows.append({
            "knob": k.knob_id, "label": k.label,
            "units": k.units, "kind": k.kind,
            "delta_signed": 0.20 if k.kind == "pct" else float(k.absolute_delta),
            "net_p50_up": up, "net_p50_dn": dn,
            "range": abs(up - dn),
            "signed_range": up - dn,
        })
    df = pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
    total = float(df["range"].sum())
    if total > 0:
        share = float(df["range"].iloc[0] / total) * 100.0
        dom = str(df["label"].iloc[0])
    else:
        share = 0.0
        dom = "none"
    return TornadoResult(
        net_p50_base=base_p50,
        rows=df, dominant_knob=dom,
        dominant_share_pct=share,
    )


def _try_load_backtest(setup_id: str = "preventive_irrigation_if_dry_spell_forecast"):
    """Try the last-run backtest from the shipped synthesised history."""
    try:
        from ..economics.backtest import (
            OnsetAnomalyRule,
            synthesise_history,
            walk_forward_backtest,
        )
        from ..economics.cost_loss import SHIPPED_SETUPS
        setup = SHIPPED_SETUPS[setup_id]
        return walk_forward_backtest(
            OnsetAnomalyRule(), setup, synthesise_history(seed=42),
            write_parquet=False,
        )
    except Exception:
        return None


def run_scenario_for_state(state) -> dict:
    """Cache-keyed engine call. Streamlit's ``st.cache_data`` uses the
    ``state.cache_key()`` string (via the wrapper function below); the
    heavy lifting lives here."""
    from ..sectors import run_decision_scenario

    region = _region_from_state(state)
    driver_spec = _driver_spec_from_state(state, region)

    decisions = _build_decisions(state)
    if state.method == "analog":
        states_list, matches = _build_states_analog(state, region)
    else:
        states_list = _build_states_perturbation(state)
        matches = []

    result = run_decision_scenario(
        decisions, states_list, region,
        driver_spec=driver_spec,
        crop_key=state.crops_selected[0] if state.crops_selected else "paddy_kharif",
        season=state.season,
        sow_date_iso=state.sow_dates[0],
    )

    # Annotate result with analog information for panels
    analog_rows = []
    for m in matches[:10]:
        analog_rows.append({
            "year": int(m.year),
            "distance": float(m.distance),
            "quality": m.quality,
            "summary": (
                f"rain_std {float(m.features_analog['rain_total_std']):+.2f}, "
                f"tmax anom {float(m.features_analog['tmax_mean_anom']):+.1f} °C, "
                f"onset {float(m.features_analog['onset_offset_days']):+.0f} d"
            ),
            "ya_observed": float("nan"),
        })
    result["analog_matches"] = analog_rows

    # Attach tornado + backtest
    result["tornado"] = _tornado_stub_from_result(result)
    result["backtest"] = _try_load_backtest()
    return result


def cached_run_for_state(state):
    """Streamlit-cached facade. Keyed on ``state.cache_key()`` so
    identical levers never re-run."""
    import streamlit as st

    @st.cache_data(show_spinner="Running scenario…")
    def _inner(cache_key: str, state_dict: dict) -> dict:
        # Rebuild the state from its dict inside the cache; streamlit
        # hashes the dict cleanly.
        from .state import WhatIfState
        s = WhatIfState.from_dict(state_dict)
        return run_scenario_for_state(s)

    return _inner(state.cache_key(), state.to_dict())
