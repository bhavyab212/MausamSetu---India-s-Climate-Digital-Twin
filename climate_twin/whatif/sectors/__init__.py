"""
whatif.sectors — L3 sector aggregators + L4 economics handlers.

Sector modules compose L1 + L2 outputs into a district / basin level
summary tuned to end-user questions. Every sector emits a summary
DataFrame or Dataset + a scenario provenance record.

The **SECTOR_REGISTRY** is the public entry point: downstream code
never imports sector internals directly.

Registered sectors:
    * ``agriculture`` — FAO-56 water balance + FAO-33 multi-stage
      yield model.  ``run_agriculture_scenario(driver_spec, levers)``
      returns a dict with all the L2 / L3 outputs needed for L4
      economics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agriculture import (
    AGRICULTURE_VERSION,
    DistrictRegistry,
    ResolutionCeilingError,
    to_district,
    yield_baseline,
    yield_water_limited,
)
from .crops import Crop, list_crops, load_crop, registry_sha256, registry_version
from .sowing_window import (
    QUANTILE_PROBABILITIES,
    QuantileDriverBundle,
    build_deterministic_bundle,
    optimize_sowing_window,
)
from .validation_apy import apy_available, get_last_validation, run_validation


@dataclass(frozen=True)
class SectorSpec:
    """Registry entry describing one L3 sector + its L4 economics handler."""
    version: str
    entry: Callable[..., Any]
    required_layers: tuple[str, ...]
    required_data: tuple[str, ...]
    economics: Callable[..., Any] | None = None      # Part 4: L4 handler


def run_agriculture_scenario(driver_spec, levers: dict | None = None) -> dict[str, Any]:
    """Run L0 → L1 (ET0) → L2 (water balance) → L3 (yield) → optional L4.

    Parameters
    ----------
    driver_spec : DriverSpec — a historical or forecast driver.
    levers      : dict with optional keys:
        ``crop``           — crop key (default: paddy_kharif)
        ``sow_date``       — ISO date string (default: driver start)
        ``irrigation``     — IrrigationSchedule or None
        ``candidate_sows`` — list of ISO date strings for the
                             sowing-window optimiser (optional)
        ``season``         — MSP season, e.g., "2024-25". If present,
                             an L4 EconomicOutcome is attached under
                             key ``economic_outcome`` and the ``crop``
                             key must exist in prices.yaml.
        ``price_perturbation`` — dict of scalar biases the sector
                             applies to numbers before valuation
                             (used by tornado / decision scenarios).

    Returns
    -------
    dict with keys::

        crop, sow_date, water_balance, yield_grid, yield_baseline,
        sowing_window (only if candidate_sows was provided),
        economic_outcome (only if season was provided),
        provenance
    """
    from datetime import date as _date

    from ..biophysical.water_balance import water_balance
    from ..drivers.driver import load_driver
    from ..indices.et0_hargreaves import et0_hargreaves

    levers = dict(levers or {})
    crop_key = levers.get("crop", "paddy_kharif")
    crop = load_crop(crop_key)

    sow_iso = levers.get("sow_date")
    sow = _date.fromisoformat(sow_iso) if sow_iso else driver_spec.start
    irrigation = levers.get("irrigation")

    # Pull rain, tmax, tmin all on the driver spec's dates + region.
    from dataclasses import replace
    rain_spec = replace(driver_spec, var="rain")
    tmax_spec = replace(driver_spec, var="tmax")
    tmin_spec = replace(driver_spec, var="tmin")

    rain = load_driver(rain_spec)
    tmax = load_driver(tmax_spec)
    tmin = load_driver(tmin_spec)

    # Apply climate-state perturbations if present (used by decision
    # scenarios). Every perturbation is a scalar multiplier or shift.
    overrides = dict(levers.get("overrides", {}) or {})
    if "rain_scale" in overrides:
        rain = rain * float(overrides["rain_scale"])
        rain.attrs.setdefault("perturbation", "").__class__
        rain.attrs["perturbation"] = f"rain*={overrides['rain_scale']}"
    if "rain_pct" in overrides:
        rain = rain * (1.0 + float(overrides["rain_pct"]))
        rain.attrs["perturbation"] = f"rain*={1.0 + overrides['rain_pct']:.3f}"
    if "tmax_shift_c" in overrides:
        tmax = tmax + float(overrides["tmax_shift_c"])
        tmax.attrs["perturbation"] = f"tmax+={overrides['tmax_shift_c']}"

    et0 = et0_hargreaves(tmax, tmin)
    wb = water_balance(crop, rain, et0, sow, driver_spec.region,
                        irrigation=irrigation)
    y_grid = yield_water_limited(crop, wb, tmax=tmax)
    y_base = yield_baseline(crop, driver_spec.region, sow)

    prov = {
        "crop_registry_version": crop.registry_version,
        "crop_registry_sha256": crop.registry_sha256,
        "agriculture_version": AGRICULTURE_VERSION,
        "water_balance_version": wb.attrs.get("version", ""),
        "soil_source": wb.attrs.get("soil_source", ""),
        "soil_warning": wb.attrs.get("soil_warning", ""),
    }
    out: dict[str, Any] = {
        "crop": crop.key,
        "sow_date": sow.isoformat(),
        "water_balance": wb,
        "yield_grid": y_grid,
        "yield_baseline": y_base,
        "overrides_applied": overrides,
        "provenance": prov,
    }

    cand = levers.get("candidate_sows") or []
    if cand:
        cand_dates = [_date.fromisoformat(c) if isinstance(c, str) else c for c in cand]
        bundle = build_deterministic_bundle(rain, tmax, tmin)
        out["sowing_window"] = optimize_sowing_window(
            crop, driver_spec.region, bundle, cand_dates,
            irrigation=irrigation,
        )

    # ── L4 economics (optional) ──
    season = levers.get("season")
    if season:
        from ..economics.valuation import value_agriculture
        from ..economics.prices import load_prices
        import numpy as _np
        price_set = load_prices(crop.key)
        # Apply overrides that touch the ₹ side directly. These are
        # simple biases used by the tornado; they never mutate the
        # price registry on disk.
        cost_override = float(price_set.cost_of_cultivation_inr_per_ha)
        msp_multiplier = 1.0
        ymax_multiplier = 1.0
        if "cost_pct" in overrides:
            cost_override = cost_override * (1.0 + float(overrides["cost_pct"]))
        if "msp_pct" in overrides:
            msp_multiplier = 1.0 + float(overrides["msp_pct"])
        if "ymax_pct" in overrides:
            ymax_multiplier = 1.0 + float(overrides["ymax_pct"])

        ya_mean = float(_np.nanmean(y_grid["Ya"].values)) * ymax_multiplier
        yb_mean = float(_np.nanmean(y_base["Ya"].values)) * ymax_multiplier
        # Degenerate q10/q50/q90 today; Part 5 supplies genuine
        # quantile drivers that will differentiate them.
        yq = {"q10": ya_mean, "q50": ya_mean, "q90": ya_mean}

        # Apply price-multiplier by re-instantiating a shadow PriceSet
        from dataclasses import replace as _replace
        shadow_ps = _replace(
            price_set,
            msp_by_season={
                k: v * msp_multiplier for k, v in price_set.msp_by_season.items()
            },
            cost_of_cultivation_inr_per_ha=cost_override,
        )
        eo = value_agriculture(
            yield_qdict=yq,
            baseline_ya_t_ha=yb_mean,
            crop=crop,
            price_set=shadow_ps,
            season=str(season),
            region_kind="district",
            region_id=str(driver_spec.region.id or driver_spec.region.kind),
        )
        out["economic_outcome"] = eo
        out["provenance"]["prices_registry_version"] = price_set.registry_version
        out["provenance"]["prices_registry_sha256"] = price_set.registry_sha256
        out["provenance"]["msp_season"] = str(season)
        out["provenance"]["valuation_version"] = eo.provenance["valuation_version"]

    return out


# Lazy import to avoid a circular import at package load time.
def _lazy_value_agri():
    from ..economics.valuation import value_agriculture
    return value_agriculture


def _run_analog_bucket_cell(dec, st, driver_spec, levers: dict) -> Any:
    """Analog-bucket cell runner: for each analog year in the bucket,
    run the sector against observed IMD data for that year (with the
    sow_date shifted into that calendar year), collect Ya samples,
    then produce an EconomicOutcome whose q10/q50/q90 come from the
    empirical distribution across analog years.

    This is the *whole point* of Method 2: analog forecast = weighted
    average of the observed outcomes of similar past years."""
    from dataclasses import replace as _replace
    from datetime import date as _date

    import numpy as _np

    from ..economics.prices import load_prices
    from ..economics.valuation import EconomicOutcome, value_agriculture

    years = list(st.analog_years or ())
    if not years:
        # Empty bucket → surface a zero-weight cell without exploding
        ps = load_prices(levers["crop"])
        cost = float(ps.cost_of_cultivation_inr_per_ha)
        zero = {"q10": 0.0, "q50": 0.0, "q90": 0.0}
        return EconomicOutcome(
            crop=levers["crop"], season=levers["season"],
            region_kind="district",
            region_id=str(driver_spec.region.id or driver_spec.region.kind),
            gross_revenue_inr_per_ha=zero,
            cost_inr_per_ha={"q10": cost, "q50": cost, "q90": cost},
            net_revenue_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
            baseline_net_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
            delta_vs_baseline={},
            price_source="msp",
            provenance={"decision_kind": "analog_bucket_empty",
                         "n_analog_years": 0},
        )

    if dec.kind == "fallow":
        ps = load_prices(levers["crop"])
        cost = float(ps.cost_of_cultivation_inr_per_ha) * 0.10
        return EconomicOutcome(
            crop=levers["crop"], season=levers["season"],
            region_kind="district",
            region_id=str(driver_spec.region.id or driver_spec.region.kind),
            gross_revenue_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
            cost_inr_per_ha={"q10": cost, "q50": cost, "q90": cost},
            net_revenue_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
            baseline_net_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
            delta_vs_baseline={},
            price_source="msp",
            provenance={"decision_kind": "fallow-in-analog-bucket",
                         "analog_years": years},
        )

    # For each analog year: build a shifted driver spec and run the
    # sector with observed data for that year. We reuse the same
    # sow_date month/day, only the year changes.
    sow_iso = levers.get("sow_date") or driver_spec.start.isoformat()
    sow_ref = _date.fromisoformat(sow_iso)

    crop = load_crop(levers["crop"])
    # Duration = crop.total_days
    dur = int(crop.total_days)
    ya_samples: list[float] = []
    ya_baseline_samples: list[float] = []
    for y in years:
        # Skip leap-day edge case: if sow_ref is Feb 29 in a non-leap year
        try:
            sow_y = _date(int(y), sow_ref.month, sow_ref.day)
        except ValueError:
            sow_y = _date(int(y), sow_ref.month, 28)
        end_y = _date.fromordinal(sow_y.toordinal() + dur - 1)
        year_spec = _replace(driver_spec, dates=(sow_y, end_y), var="rain")
        year_levers = dict(levers)
        year_levers["sow_date"] = sow_y.isoformat()
        # Analog scenarios use pristine observed data — strip
        # perturbation-style overrides that would otherwise scale rain
        # / shift tmax. Analog bucketing IS the state definition; we
        # must not double-perturb.
        year_levers.pop("overrides", None)
        try:
            result = run_agriculture_scenario(year_spec, year_levers)
        except Exception:
            continue
        yg = result.get("yield_grid")
        yb = result.get("yield_baseline")
        if yg is None or yb is None:
            continue
        ya_samples.append(float(_np.nanmean(yg["Ya"].values)))
        ya_baseline_samples.append(float(_np.nanmean(yb["Ya"].values)))

    if not ya_samples:
        raise RuntimeError(
            f"analog-bucket runner produced no valid Ya samples for "
            f"years {years} — check driver availability"
        )

    ya_arr = _np.asarray(ya_samples, dtype=_np.float64)
    yb_mean = float(_np.mean(ya_baseline_samples)) if ya_baseline_samples else float(_np.mean(ya_arr))
    # Empirical q10/q50/q90 from the *observed* outcomes of the bucket's
    # analog years. This is the whole point of Method 2.
    yq = {
        "q10": float(_np.percentile(ya_arr, 10)),
        "q50": float(_np.percentile(ya_arr, 50)),
        "q90": float(_np.percentile(ya_arr, 90)),
    }

    ps = load_prices(levers["crop"])
    from dataclasses import replace as _replace2
    # No shadow-multipliers here: analog states don't perturb prices
    shadow_ps = ps
    eo = value_agriculture(
        yield_qdict=yq,
        baseline_ya_t_ha=yb_mean,
        crop=crop,
        price_set=shadow_ps,
        season=levers["season"],
        region_kind="district",
        region_id=str(driver_spec.region.id or driver_spec.region.kind),
    )
    eo.provenance.update({
        "analog_bucket_years": [int(y) for y in years],
        "n_analog_years_effective": len(ya_samples),
        "ya_samples_t_ha": [float(x) for x in ya_arr],
    })
    return eo


SECTOR_REGISTRY: dict[str, SectorSpec] = {
    "agriculture": SectorSpec(
        version=AGRICULTURE_VERSION,
        entry=run_agriculture_scenario,
        required_layers=("driver", "et0_hargreaves", "gdd"),
        required_data=("crops.yaml", "soil.awc", "prices.yaml"),
        economics=_lazy_value_agri,
    ),
}


def run_decision_scenario(
    decisions: list,           # list[Decision] from economics.payoff
    states: list,              # list[ClimateState] from economics.payoff
    region,                    # RegionSpec
    *,
    driver_spec,               # DriverSpec — the historical anchor for the run
    crop_key: str = "paddy_kharif",
    season: str = "2024-25",
    sow_date_iso: str | None = None,
) -> dict[str, Any]:
    """Assemble a full decision scenario: build a PayoffMatrix by running
    ``run_agriculture_scenario`` per (decision, state), then compute
    ranking / regret / minimax / VaR / CVaR summaries.

    Each Decision.params must supply ``sow_date`` (ISO) and may supply
    ``crop``, ``irrigation`` overrides. Each ClimateState carries a
    ``perturbation`` dict that lands in the sector runner's ``overrides``
    key.

    Returns
    -------
    dict::

        payoff_matrix, recommendation (dict), expected_value,
        regret_matrix, minimax_regret, worst_case,
        var_10, cvar_10, stochastic_dominance,
        provenance (list of per-cell records)
    """
    from ..economics.decision import (
        expected_value,
        expected_value_with_uncertainty,
        minimax_regret,
        recommend,
        regret_matrix,
        stochastic_dominance,
        var_cvar,
        worst_case,
    )
    from ..economics.payoff import build_payoff_matrix
    from ..economics.valuation import EconomicOutcome
    import numpy as _np

    def _run_cell(dec, st) -> "EconomicOutcome":
        # Merge params + perturbation → levers dict
        params = dec.params_dict()
        levers: dict[str, Any] = {
            "crop": params.get("crop", crop_key),
            "sow_date": params.get("sow_date", sow_date_iso),
            "season": season,
            "overrides": st.perturbation_dict(),
        }
        # ── Analog-bucket path: iterate observed years empirically ──
        pert_kind = str(st.perturbation_dict().get("_kind", ""))
        if pert_kind == "analog_bucket" and st.analog_years:
            return _run_analog_bucket_cell(dec, st, driver_spec, levers)
        # Some Decisions are "fallow" or "skip" — return a zero-yield
        # outcome without running the whole chain.
        if dec.kind == "fallow":
            from ..economics.prices import load_prices
            ps = load_prices(levers["crop"])
            cost = float(ps.cost_of_cultivation_inr_per_ha) * 0.10  # keeper-cost only
            return EconomicOutcome(
                crop=levers["crop"],
                season=season,
                region_kind="district",
                region_id=str(region.id or region.kind),
                gross_revenue_inr_per_ha={"q10": 0.0, "q50": 0.0, "q90": 0.0},
                cost_inr_per_ha={"q10": cost, "q50": cost, "q90": cost},
                net_revenue_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
                baseline_net_inr_per_ha={"q10": -cost, "q50": -cost, "q90": -cost},
                delta_vs_baseline={},
                price_source="msp",
                provenance={
                    "decision_kind": "fallow",
                    "prices_registry_version": ps.registry_version,
                    "prices_registry_sha256": ps.registry_sha256,
                    "note": "fallow: no revenue, 10% keeper-cost booked",
                },
            )
        result = run_agriculture_scenario(driver_spec, levers)
        eo = result.get("economic_outcome")
        if eo is None:
            raise RuntimeError(
                f"decision {dec.label}: sector runner returned no "
                "economic_outcome (missing season lever?)"
            )
        return eo

    pm = build_payoff_matrix(decisions, states, region, run_cell=_run_cell)

    ev_bands = expected_value_with_uncertainty(pm)
    var_arr, cvar_arr = var_cvar(pm, alpha=0.10)
    return {
        "payoff_matrix": pm,
        "recommendation": recommend(pm),
        "expected_value": expected_value(pm),
        "expected_value_bands": ev_bands,
        "regret_matrix": regret_matrix(pm),
        "minimax_regret": minimax_regret(pm),
        "worst_case": worst_case(pm),
        "var_10": var_arr,
        "cvar_10": cvar_arr,
        "stochastic_dominance": stochastic_dominance(pm),
        "provenance": pm.provenance,
    }


def run(*, sector: str, driver, indices, biophysical, levers) -> dict[str, Any]:
    """Adapter for :func:`whatif.scenarios.engine.run_scenario`.

    Looks up the sector, calls its entry with the DriverSpec + levers.
    The engine already produced ``driver`` (a DataArray) — but the
    agriculture sector needs the DriverSpec too, so callers who want
    a full L3 run should call ``run_agriculture_scenario`` directly
    with the spec. This adapter is a graceful no-op until the engine
    is refactored to pass specs through Part 4."""
    if sector not in SECTOR_REGISTRY:
        raise KeyError(
            f"sector {sector!r} not registered. Known: {list(SECTOR_REGISTRY)}"
        )
    return {
        "sector": sector,
        "note": (
            "engine → sector adapter is scaffolded; call "
            "run_agriculture_scenario(spec, levers) directly for now."
        ),
    }


class RepresentationMismatch(RuntimeError):
    """Raised when a Long-Term multi-model ensemble is fed into the
    Short-Term three-pass q10/q50/q90 codepath. Part-7 Rule (§9c):
    the two uncertainty representations don't mix — a Long-Term run
    surfaces a ``model`` dim, a Short-Term run surfaces q10/q50/q90
    on the ``quantile`` attribute. Mixing them silently would be a
    category error."""


@dataclass
class LongTermResult:
    """Result of a Long-Term scenario run.

    ``sector_out`` and ``economics`` carry an extra ``model`` dim (one
    per GCM in the ensemble). Consumers summarise across the model dim
    via ensemble mean, spread, and agreement counts — never by picking
    a single model.
    """
    scenario_id: str
    year_center: int
    region: object
    sector: str
    sector_out: Any = None
    economics: Any = None
    provenance: dict = None
    version: str = "long-term-v1"


def run_long_term_scenario(
    scenario_id: str,
    target_center_year: int,
    sector: str,
    region,
    adaptations: list[str] | None = None,
) -> LongTermResult:
    """Compose a Long-Term scenario end-to-end.

    Contract:
        * Uses NEX-GDDP × downscale × sector runner → multi-model stack.
        * Adaptation NPV is computed via the L4 pipeline over the
          multi-model ensemble.  Rule 3: uncertainty is decomposed
          before reporting.

    Today's Part-7 scaffold prepares the plumbing.  Actual sector-
    runner-over-model-dim inference is data-heavy (needs NEX-GDDP on
    disk); we surface a provenance record that includes the SSP,
    downscaling method, adaptation ids, and cost-completeness
    verdicts, and defer the heavy pass to the UI when data is
    available. This behaviour mirrors Part 3's soil-default banner:
    the pipeline never fabricates numbers when the raster isn't there.
    """
    from ..drivers.ssp import (
        SSP_REGISTRY_VERSION,
        baseline_period,
        load_scenario,
        window_for_center,
        registry_sha256 as _ssp_sha,
    )

    if sector != "agriculture":
        raise NotImplementedError(
            f"Long-Term sector {sector!r} lands in a later part. "
            "Agriculture is the shipped sector for Part 7."
        )
    ssp = load_scenario(scenario_id)
    window = window_for_center(target_center_year)

    # Cost-completeness gate for selected adaptations
    from .adaptations import load_adaptation
    adaptation_verdicts = {}
    for a in adaptations or []:
        try:
            opt = load_adaptation(a)
            adaptation_verdicts[a] = {
                "cost_complete": bool(opt.cost_complete),
                "effective_years": int(opt.effective_years),
                "capex_inr_per_ha": float(opt.capex_inr_per_ha),
                "capex_citation": opt.capex_citation,
            }
        except KeyError:
            adaptation_verdicts[a] = {"error": "unknown adaptation id"}

    prov = {
        "scenario_id": scenario_id,
        "scenario_label": ssp.label,
        "target_center_year": int(target_center_year),
        "window_20yr": list(window),
        "baseline_period": list(baseline_period()),
        "ssp_registry_version": SSP_REGISTRY_VERSION,
        "ssp_registry_sha256": _ssp_sha(),
        "representation": "multi_model_ensemble",
        "adaptations_selected": list(adaptations or []),
        "adaptation_verdicts": adaptation_verdicts,
    }
    return LongTermResult(
        scenario_id=scenario_id,
        year_center=int(target_center_year),
        region=region, sector=sector,
        sector_out=None, economics=None,
        provenance=prov,
    )


__all__ = [
    "SectorSpec", "SECTOR_REGISTRY", "run",
    "run_agriculture_scenario", "run_decision_scenario",
    "run_long_term_scenario", "LongTermResult", "RepresentationMismatch",
    "Crop", "load_crop", "list_crops", "registry_version", "registry_sha256",
    "AGRICULTURE_VERSION", "yield_water_limited", "yield_baseline",
    "to_district", "DistrictRegistry", "ResolutionCeilingError",
    "QuantileDriverBundle", "optimize_sowing_window",
    "build_deterministic_bundle", "QUANTILE_PROBABILITIES",
    "apy_available", "get_last_validation", "run_validation",
]
