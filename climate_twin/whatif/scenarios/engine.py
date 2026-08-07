"""
whatif.scenarios.engine — the scenario orchestrator.

Walks L0 → L1 → L2 → L3 → L4 by calling each layer's public function.
Any layer that hasn't been implemented yet is skipped with a
``layer_pending`` entry in the provenance log; the run still returns a
valid ``ScenarioResult`` carrying the layers that did complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from ..drivers.driver import DriverSpec, load_driver
from .provenance import (
    ProvenanceRecord,
    record_add_layer,
    record_close,
    record_open,
)


@dataclass
class ScenarioResult:
    run_id: str
    driver: xr.DataArray | xr.Dataset
    indices: dict[str, xr.DataArray] = field(default_factory=dict)
    biophysical: dict[str, xr.DataArray] = field(default_factory=dict)
    sector_out: dict[str, Any] = field(default_factory=dict)
    economics: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceRecord | None = None
    layer_pending: list[str] = field(default_factory=list)


def _driver_summary(da: xr.DataArray | xr.Dataset) -> dict:
    if isinstance(da, xr.Dataset):
        return {"kind": "Dataset", "data_vars": list(da.data_vars)}
    return {
        "kind": "DataArray",
        "name": str(da.name),
        "dims": dict(da.sizes),
        "quantile": da.attrs.get("quantile", "?"),
        "source": da.attrs.get("source", "?"),
        "source_version": da.attrs.get("source_version", "?"),
    }


def run_scenario(
    driver: DriverSpec,
    sector: str | None = None,
    levers: dict | None = None,
) -> ScenarioResult:
    """Orchestrate one scenario end-to-end.

    Only the L0 → driver layer is wired in Part 1. Downstream layers
    (indices / biophysical / sectors / economics) are appended in the
    respective Part-2+ builds; each of them, when present, exposes a
    ``run(spec, levers, driver_array) → dict`` entry point which this
    orchestrator calls in order.

    The provenance record is opened before L0, updated at every layer
    boundary, and closed with the outputs summary at the end.
    """
    rec = record_open(driver, levers=levers)

    # ── L0 driver ──
    da = load_driver(driver)
    record_add_layer(
        rec,
        name="drivers.load_driver",
        version=f"{driver.mode}-v1",
        inputs_summary=_driver_summary(da),
    )

    result = ScenarioResult(run_id=rec.run_id, driver=da, provenance=rec)

    # ── L1 indices (Part 2) ──
    try:
        from .. import indices as _idx  # noqa: F401
        if hasattr(_idx, "run"):
            result.indices = _idx.run(driver, da, levers or {})
            record_add_layer(rec, "indices.run", "v1",
                              {"n": len(result.indices)})
        else:
            result.layer_pending.append("indices")
            record_add_layer(rec, "indices", "pending", {})
    except Exception as e:  # noqa: BLE001
        result.layer_pending.append(f"indices ({type(e).__name__}: {e})")
        record_add_layer(rec, "indices", "error", {"error": repr(e)})

    # ── L2 biophysical (Part 3) ──
    try:
        from .. import biophysical as _bp  # noqa: F401
        if hasattr(_bp, "run"):
            result.biophysical = _bp.run(driver, result.indices, levers or {})
            record_add_layer(rec, "biophysical.run", "v1",
                              {"n": len(result.biophysical)})
        else:
            result.layer_pending.append("biophysical")
            record_add_layer(rec, "biophysical", "pending", {})
    except Exception as e:  # noqa: BLE001
        result.layer_pending.append(f"biophysical ({type(e).__name__}: {e})")
        record_add_layer(rec, "biophysical", "error", {"error": repr(e)})

    # ── L3 sectors (Part 3+) ──
    try:
        from .. import sectors as _sec  # noqa: F401
        if sector and hasattr(_sec, "run"):
            result.sector_out = _sec.run(
                sector=sector,
                driver=da,
                indices=result.indices,
                biophysical=result.biophysical,
                levers=levers or {},
            )
            record_add_layer(rec, f"sectors.{sector}", "v1",
                              {"n": len(result.sector_out)})
        else:
            result.layer_pending.append(f"sectors.{sector or 'none-selected'}")
            record_add_layer(rec, "sectors", "pending", {})
    except Exception as e:  # noqa: BLE001
        result.layer_pending.append(f"sectors ({type(e).__name__}: {e})")
        record_add_layer(rec, "sectors", "error", {"error": repr(e)})

    # ── L4 economics (Part 4) ──
    try:
        from .. import economics as _econ  # noqa: F401
        if hasattr(_econ, "run"):
            result.economics = _econ.run(result.sector_out, levers or {})
            record_add_layer(rec, "economics.run", "v1",
                              {"n": len(result.economics)})
        else:
            result.layer_pending.append("economics")
            record_add_layer(rec, "economics", "pending", {})
    except Exception as e:  # noqa: BLE001
        result.layer_pending.append(f"economics ({type(e).__name__}: {e})")
        record_add_layer(rec, "economics", "error", {"error": repr(e)})

    # ── close provenance ──
    outputs = {
        "layer_pending": list(result.layer_pending),
        "driver": _driver_summary(da),
        "n_indices": len(result.indices),
        "n_biophysical": len(result.biophysical),
        "sector_selected": sector,
    }
    record_close(rec, outputs_summary=outputs)
    return result
