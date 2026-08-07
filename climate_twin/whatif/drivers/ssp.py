"""
whatif.drivers.ssp — SSP scenario registry + long-term driver façade.

Primary sources:
    * IPCC AR6 WG1 SPM (2021), Table SPM.1 — assessed global warming
      ranges per SSP × 2081-2100 window vs the 1850-1900 baseline.
    * Riahi, K. et al. (2017) "The Shared Socioeconomic Pathways and
      their energy, land use, and greenhouse gas emissions
      implications", Global Env. Change 42:153-168.

Contract:
    * Every scenario carries its citation, the narrative (paraphrased
      from AR6 SPM), and the assessed warming range.
    * ``load_ssp_driver(scenario_id, year_center, region, variable)``
      returns a 20-year window (multi-model stack) on the master grid.
    * The 20-year window follows Rule 6: ``center-9 → center+10``,
      e.g., 2050 → 2041-2060. State the window in every UI caption.

Version: ``ssp-registry-v1``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import xarray as xr
import yaml

from ..config.region import RegionSpec
from .nex_gddp import assemble_multi_model, list_models

SSP_REGISTRY_VERSION = "ssp-registry-v1"
_REGISTRY_PATH = Path(__file__).resolve().parent / "ssp_registry.yaml"


@dataclass(frozen=True)
class SSPScenario:
    id: str
    label: str
    forcing_2100_Wm2: float
    global_warming_2081_2100_C: tuple[float, float]
    narrative: str
    citation: str


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    return yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_version() -> str:
    return str(_load_registry()["version"])


def registry_sha256() -> str:
    return hashlib.sha256(_REGISTRY_PATH.read_bytes()).hexdigest()[:16]


def list_scenarios() -> list[str]:
    return [s["id"] for s in _load_registry()["scenarios"]]


def default_display_scenarios() -> list[str]:
    return list(_load_registry().get("default_display", []))


def baseline_period() -> tuple[int, int]:
    p = _load_registry().get("baseline_period", [1971, 2000])
    return int(p[0]), int(p[1])


def load_scenario(scenario_id: str) -> SSPScenario:
    reg = _load_registry()
    for s in reg["scenarios"]:
        if s["id"] == scenario_id:
            gr = s["global_warming_2081_2100_C"]
            return SSPScenario(
                id=s["id"], label=s["label"],
                forcing_2100_Wm2=float(s["forcing_2100_Wm2"]),
                global_warming_2081_2100_C=(float(gr[0]), float(gr[1])),
                narrative=str(s["narrative"]).strip(),
                citation=str(s["citation"]),
            )
    raise KeyError(
        f"scenario {scenario_id!r} not in registry "
        f"(known: {list_scenarios()})"
    )


def window_for_center(center_year: int) -> tuple[int, int]:
    """20-year window centred on ``center_year`` per Rule 6.

    2030 → 2021-2040
    2050 → 2041-2060
    2075 → 2066-2085
    """
    return (int(center_year) - 9, int(center_year) + 10)


def load_ssp_driver(
    scenario_id: str, year_center: int, region: RegionSpec,
    variable: str, *, models: list[str] | None = None,
) -> xr.DataArray:
    """Return the 20-year multi-model stack on the master grid.

    Dims: ``(model, time, lat, lon)``. Variable is the engine's
    convention (``rain`` → NEX-GDDP ``pr``; ``tmax`` → ``tasmax``;
    ``tmin`` → ``tasmin``).
    """
    _ = load_scenario(scenario_id)     # validate scenario_id
    var_map = {"rain": "pr", "tmax": "tasmax", "tmin": "tasmin",
                "pr": "pr", "tasmax": "tasmax", "tasmin": "tasmin"}
    if variable not in var_map:
        raise ValueError(
            f"SSP driver supports {list(var_map)}, not {variable!r}"
        )
    year_range = window_for_center(year_center)
    return assemble_multi_model(
        variable=var_map[variable],
        ssp=scenario_id,
        period="future",
        year_range=year_range,
        region=region,
        models=models or list_models(),
    )
