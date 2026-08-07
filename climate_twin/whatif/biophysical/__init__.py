"""
whatif.biophysical — L2, process models operating on L1 indices.

Public entry points:
    * :func:`water_balance`    — FAO-56 single-Kc daily bucket.
    * :func:`etc_series`       — ETc = Kc × ET0 broadcast helper.
    * :func:`awc_mm_per_m`     — soil available-water-capacity map.
    * :func:`taw`, :func:`raw` — root-zone water pools.

Deterministic; no model-in-the-loop training here — the trained pieces
live in ``climate_twin.train``.
"""
from __future__ import annotations

from .crop_water import (
    etc_series,
    gdd_stage_schedule,
    kc_curve,
    stage_of_day,
)
from .soil import (
    DEFAULT_AWC_MM_PER_M,
    SoilSourceInfo,
    awc_mm_per_m,
    raw,
    taw,
)
from .water_balance import (
    WATER_BALANCE_VERSION,
    IrrigationSchedule,
    water_balance,
)

__all__ = [
    "water_balance", "WATER_BALANCE_VERSION", "IrrigationSchedule",
    "kc_curve", "stage_of_day", "etc_series", "gdd_stage_schedule",
    "awc_mm_per_m", "taw", "raw", "SoilSourceInfo", "DEFAULT_AWC_MM_PER_M",
]
