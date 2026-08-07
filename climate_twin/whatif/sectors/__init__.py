"""
whatif.sectors — L3, sector aggregators over biophysical outputs.

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
    """Registry entry describing one L3 sector."""
    version: str
    entry: Callable[..., Any]
    required_layers: tuple[str, ...]
    required_data: tuple[str, ...]


def run_agriculture_scenario(driver_spec, levers: dict | None = None) -> dict[str, Any]:
    """Run L0 → L1 (ET0) → L2 (water balance) → L3 (yield) for one crop.

    Parameters
    ----------
    driver_spec : DriverSpec — a historical or forecast driver.
    levers      : dict with optional keys:
        ``crop``           — crop key (default: paddy_kharif)
        ``sow_date``       — ISO date string (default: driver start)
        ``irrigation``     — IrrigationSchedule or None
        ``candidate_sows`` — list of ISO date strings for the
                             sowing-window optimiser (optional)

    Returns
    -------
    dict with keys::

        crop, sow_date, water_balance, yield_grid, yield_baseline,
        sowing_window (only if candidate_sows was provided),
        provenance (crop.registry_sha256, agri version, wb version)
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

    et0 = et0_hargreaves(tmax, tmin)
    wb = water_balance(crop, rain, et0, sow, driver_spec.region,
                        irrigation=irrigation)
    y_grid = yield_water_limited(crop, wb, tmax=tmax)
    y_base = yield_baseline(crop, driver_spec.region, sow)

    out = {
        "crop": crop.key,
        "sow_date": sow.isoformat(),
        "water_balance": wb,
        "yield_grid": y_grid,
        "yield_baseline": y_base,
        "provenance": {
            "crop_registry_version": crop.registry_version,
            "crop_registry_sha256": crop.registry_sha256,
            "agriculture_version": AGRICULTURE_VERSION,
            "water_balance_version": wb.attrs.get("version", ""),
            "soil_source": wb.attrs.get("soil_source", ""),
            "soil_warning": wb.attrs.get("soil_warning", ""),
        },
    }

    cand = levers.get("candidate_sows") or []
    if cand:
        cand_dates = [_date.fromisoformat(c) if isinstance(c, str) else c for c in cand]
        bundle = build_deterministic_bundle(rain, tmax, tmin)
        out["sowing_window"] = optimize_sowing_window(
            crop, driver_spec.region, bundle, cand_dates,
            irrigation=irrigation,
        )
    return out


SECTOR_REGISTRY: dict[str, SectorSpec] = {
    "agriculture": SectorSpec(
        version=AGRICULTURE_VERSION,
        entry=run_agriculture_scenario,
        required_layers=("driver", "et0_hargreaves", "gdd"),
        required_data=("crops.yaml", "soil.awc"),
    ),
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


__all__ = [
    "SectorSpec", "SECTOR_REGISTRY", "run", "run_agriculture_scenario",
    "Crop", "load_crop", "list_crops", "registry_version", "registry_sha256",
    "AGRICULTURE_VERSION", "yield_water_limited", "yield_baseline",
    "to_district", "DistrictRegistry", "ResolutionCeilingError",
    "QuantileDriverBundle", "optimize_sowing_window",
    "build_deterministic_bundle", "QUANTILE_PROBABILITIES",
    "apy_available", "get_last_validation", "run_validation",
]
