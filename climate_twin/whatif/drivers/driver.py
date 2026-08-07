"""
whatif.drivers.driver — the L0 façade.

Everything above L0 (indices, biophysical, sectors, economics) calls
``load_driver(DriverSpec)`` and never touches a specific driver module
directly. This lets us swap or extend drivers (perturbation, analogs,
SSP) without touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import xarray as xr

from ..config.region import RegionSpec, apply_region
from ._common import stamp_attrs
from .ensemble import get_ensemble
from .historical import get_historical


Mode = Literal["historical", "forecast", "perturbation", "analog", "ssp"]
Var  = Literal["rain", "tmax", "tmin", "tmean"]


@dataclass(frozen=True)
class DriverSpec:
    """Immutable declaration of an L0 scenario input."""
    mode: Mode
    var: Var
    dates: tuple[date, date]
    region: RegionSpec
    quantile: float | None = None            # forecast mode only
    extras: tuple[tuple[str, Any], ...] = ()

    @property
    def start(self) -> date: return self.dates[0]

    @property
    def end(self) -> date: return self.dates[1]

    def extras_dict(self) -> dict:
        return dict(self.extras)


def load_driver(spec: DriverSpec) -> xr.DataArray:
    """Dispatch a DriverSpec to the concrete L0 driver.

    Returns a single ``xr.DataArray`` (never a Dataset) on the master
    grid, IST-aware, with ``attrs["units"|"source"|"source_version"|
    "quantile"]`` set. For ``forecast`` mode, the array is the single
    quantile requested via ``spec.quantile`` (default 0.50).
    """
    if spec.mode == "historical":
        da = get_historical(spec.var, spec.start, spec.end)
        return apply_region(da, spec.region)

    if spec.mode == "forecast":
        # Forecast is single-day: end == start (multi-day forecasts are the
        # sum of Part-6's rolled-out predictions; not implemented here).
        if spec.start != spec.end:
            raise NotImplementedError(
                "multi-day forecast rollout is a Part 6 feature; "
                "for now request one date at a time."
            )
        q = spec.quantile if spec.quantile is not None else 0.50
        qkey = f"q{int(q * 100):02d}"
        ds = get_ensemble(spec.var if spec.var != "tmean" else "tmax",
                            spec.start, quantiles=(q,))
        if qkey not in ds:
            raise KeyError(f"ensemble returned no {qkey}: keys={list(ds.data_vars)}")
        # Promote the single quantile to a (time,lat,lon) shape for
        # engine consistency
        from datetime import datetime
        one = ds[qkey].expand_dims(time=[np.datetime64(spec.start)])
        one = stamp_attrs(
            one.rename(spec.var),
            var=spec.var,
            source=str(ds.attrs.get("source", "mausamsetu_ensemble")),
            source_version=str(ds.attrs.get("run_id", "?")),
            quantile=qkey,
        )
        return apply_region(one, spec.region)

    if spec.mode == "perturbation":
        raise NotImplementedError("perturbation driver arrives in Part 5.")
    if spec.mode == "analog":
        raise NotImplementedError("analog driver arrives in Part 5.")
    if spec.mode == "ssp":
        # Long-Term multi-model driver. Extras must supply
        # scenario, year_center; downscaling method is optional
        # (default 'qdm').
        from .ssp import load_ssp_driver
        extras = spec.extras_dict()
        scenario_id = extras.get("scenario") or extras.get("scenario_id")
        year_center = extras.get("year_center")
        if not scenario_id or not year_center:
            raise ValueError(
                "DriverSpec(mode='ssp') requires "
                "extras={'scenario': 'ssp245', 'year_center': 2050}"
            )
        models = extras.get("models")
        da = load_ssp_driver(
            scenario_id=str(scenario_id),
            year_center=int(year_center),
            region=spec.region,
            variable=spec.var,
            models=list(models) if models else None,
        )
        # Note: SSP driver returns (model, time, lat, lon); the
        # engine's load_driver contract promises xr.DataArray, so
        # the caller must be aware. The dedicated
        # run_long_term_scenario orchestrator handles the model dim.
        da.attrs["driver_mode"] = "ssp"
        da.attrs["scenario"] = str(scenario_id)
        da.attrs["year_center"] = int(year_center)
        da.attrs["representation"] = "multi_model_ensemble"
        return da

    raise ValueError(f"unknown DriverSpec.mode={spec.mode!r}")
