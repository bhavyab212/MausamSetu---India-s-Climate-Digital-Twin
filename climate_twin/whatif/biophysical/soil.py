"""
whatif.biophysical.soil — available water capacity (AWC) map.

Provides the total-available-water (TAW) and readily-available-water
(RAW) inputs to the FAO-56 water balance.

Priority order (deterministic; higher wins):
    1. ISRIC SoilGrids 250m → texture fractions → AWC via Saxton & Rawls
       2006 pedotransfer function.
    2. FAO Harmonized World Soil Database v2 (HWSD) texture classes →
       AWC via class lookup.
    3. **Fallback:** a constant map of 140 mm/m across the master grid.
       This is the median of Indian soils; results computed on it are
       INDICATIVE-ONLY and every scenario using this path records a
       WARNING in provenance.

Primary sources:
    * Saxton & Rawls (2006) "Soil water characteristic estimates by
      texture and organic matter for hydrologic solutions", Soil Sci.
      Soc. Am. J. 70:1569-1578.
    * FAO Harmonized World Soil Database v2 (2023 update).
    * ISRIC SoilGrids 250m v2.0 (Poggio et al. 2021, SOIL 7:217-240).

Contract:
    * ``awc_mm_per_m(region)`` returns AWC (mm of water per m of soil)
      on the master 0.25° grid, masked to the region.
    * ``taw(crop, awc_map, stage)`` = ``AWC × Zr(stage)`` in mm.
    * ``raw(crop, taw_map)`` = ``p × TAW`` in mm.

None of the numeric constants here are tuned — they are literature
lookups. The default value (140 mm/m) is cited by FAO IDP-56 Table 19
as a mid-range value for medium-textured soils.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from ..config.constants import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from ..config.paths import CACHE_DIR
from ..config.region import RegionSpec, apply_region, master_axes


# FAO-56 IDP Table 19 — medium-textured soils: AWC ≈ 140 mm per m of soil
DEFAULT_AWC_MM_PER_M: float = 140.0

_SOIL_SOURCE_VERSION = "awc-default-140mm-per-m-v1"


@dataclass(frozen=True)
class SoilSourceInfo:
    """What backed the AWC map for a given call."""
    source: Literal["soilgrids", "hwsd", "default"]
    version: str
    warning: str | None       # non-None ⇒ append to provenance


def _make_default_awc_da() -> xr.DataArray:
    """AWC constant map on the master grid."""
    lat, lon = master_axes()
    arr = np.full((N_LAT, N_LON), DEFAULT_AWC_MM_PER_M, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="awc_mm_per_m",
    )
    da.attrs.update({
        "units": "mm/m",
        "source": "default",
        "source_version": _SOIL_SOURCE_VERSION,
        "citation": "FAO IDP-56 Table 19; medium-texture midpoint",
        "note": "INDICATIVE-ONLY — no SoilGrids/HWSD raster on disk",
    })
    return da


def _try_load_soilgrids() -> xr.DataArray | None:
    """Placeholder: SoilGrids ingest lives with the data pipeline.
    Returns None until the on-disk raster + PTF pipeline are wired up."""
    # Cache directory would be CACHE_DIR / "soil" / "soilgrids_awc.nc"
    p = CACHE_DIR / "soil" / "soilgrids_awc.nc"
    if not p.exists():
        return None
    try:
        return xr.open_dataarray(p)
    except Exception:
        return None


def _try_load_hwsd() -> xr.DataArray | None:
    """Placeholder: HWSD ingest lives with the data pipeline."""
    p = CACHE_DIR / "soil" / "hwsd_awc.nc"
    if not p.exists():
        return None
    try:
        return xr.open_dataarray(p)
    except Exception:
        return None


def awc_mm_per_m(region: RegionSpec | None = None) -> tuple[xr.DataArray, SoilSourceInfo]:
    """Return AWC (mm/m) on the master grid, masked to ``region``.

    Also returns a :class:`SoilSourceInfo` telling the caller which
    backend produced the map — the scenario orchestrator uses this to
    stamp a WARNING onto provenance when the default is used.
    """
    da = _try_load_soilgrids()
    if da is not None:
        info = SoilSourceInfo(source="soilgrids",
                                version=str(da.attrs.get("source_version", "soilgrids-v?")),
                                warning=None)
    else:
        da = _try_load_hwsd()
        if da is not None:
            info = SoilSourceInfo(source="hwsd",
                                    version=str(da.attrs.get("source_version", "hwsd-v?")),
                                    warning=None)
        else:
            da = _make_default_awc_da()
            info = SoilSourceInfo(
                source="default",
                version=_SOIL_SOURCE_VERSION,
                warning=(
                    f"AWC map defaulted to constant "
                    f"{DEFAULT_AWC_MM_PER_M:.0f} mm/m — no SoilGrids/HWSD "
                    "raster is available on disk. Yield estimates are "
                    "INDICATIVE-ONLY."
                ),
            )

    if region is not None:
        da = apply_region(da, region)
    return da, info


def taw(crop, awc_map: xr.DataArray, stage: str = "max") -> xr.DataArray:
    """Total available water in the root zone (mm) = AWC × Zr(stage).

    ``stage`` is either "ini" or "max"; production code uses "max" (root
    at full extent) for stability, and "ini" only for early-season
    diagnostics.
    """
    if stage not in crop.root_depth_m:
        raise KeyError(
            f"crop.root_depth_m has no stage {stage!r}; "
            f"available: {list(crop.root_depth_m)}"
        )
    zr = float(crop.root_depth_m[stage])
    out = awc_map * zr
    out.name = "taw"
    out.attrs.update({
        "units": "mm",
        "root_depth_m": zr,
        "stage": stage,
        "citation": awc_map.attrs.get("citation", ""),
        "source_chain": f"{awc_map.attrs.get('source_version','?')} × Zr[{stage}]",
    })
    return out


def raw(crop, taw_map: xr.DataArray) -> xr.DataArray:
    """Readily available water (mm) = p × TAW."""
    p = float(crop.depletion_p)
    out = taw_map * p
    out.name = "raw"
    out.attrs.update({
        "units": "mm",
        "depletion_p": p,
        "citation": "FAO-56 eq. 82 (RAW = p·TAW)",
        "source_chain": f"{taw_map.attrs.get('source_chain','?')} × p={p}",
    })
    return out
