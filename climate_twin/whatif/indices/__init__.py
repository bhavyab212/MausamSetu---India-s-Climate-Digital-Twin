"""
whatif.indices — L1 climate indices, plus the ``INDEX_REGISTRY``.

Every entry in the registry ships with:
    * ``version`` — bumped whenever the formula / constants change; the
      version appears in every scenario's provenance so replays can
      detect drift.
    * ``inputs`` — the driver variables required to compute it.
    * ``units``  — the physical unit the output carries.
    * ``needs_fit`` — whether a fit artifact (SPI, SPEI, R95p, GEV RL)
      lives under ``CACHE_DIR/fits/`` and must be produced first.
    * ``citation`` — primary reference. Reviewers read these.

Downstream sectors ask for ``INDEX_REGISTRY["et0_hargreaves"]`` and
never import the concrete function directly. That indirection is what
lets us slot in Penman-Monteith later without touching sector code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .aridity import aridity_index, anomaly, pctile_anomaly, UNEP_CLASSES
from .degree_days import cdd, hdd
from .dry_spell import DRY_THRESHOLD_MM, cdd_wmo, longest_dry_spell
from .et0_hargreaves import et0_hargreaves
from .extremes import r95p, return_level, rx1day, rx5day
from .gdd import gdd, gdd_for_crop
from .heat_stress import hot_day_count, imd_heatwave_days
from .onset import (
    IMD_KERALA_CRITERION,
    OnsetCriterion,
    imd_kerala_onset,
    regional_onset,
)
from .radiation import broadcast_ra_to, ra_table
from .reference import (
    TRAIN_YEARS,
    VALID_YEARS,
    LeakageError,
    assert_train_only,
    climatology,
    valid_fraction,
    zscore,
)
from .spei import SPEIFit, fit_spei, spei
from .spi import SPIFit, fit_spi, fit_spi_from_cube, spi


@dataclass(frozen=True)
class IndexSpec:
    """Registry entry describing one L1 index."""
    fn: Callable[..., Any]
    version: str
    inputs: tuple[str, ...]           # driver var names
    units: str
    needs_fit: bool
    citation: str


INDEX_REGISTRY: dict[str, IndexSpec] = {
    "et0_hargreaves": IndexSpec(
        fn=et0_hargreaves,
        version="hargreaves-v1",
        inputs=("tmax", "tmin"),
        units="mm/day",
        needs_fit=False,
        citation="FAO-56 §3; Hargreaves & Samani 1985",
    ),
    "gdd": IndexSpec(
        fn=gdd,
        version="mcmaster-wilhelm-1997-m1-v1",
        inputs=("tmax", "tmin"),
        units="°C·day",
        needs_fit=False,
        citation="McMaster & Wilhelm 1997",
    ),
    "cdd_24c": IndexSpec(
        fn=lambda tmean: cdd(tmean, t_base_c=24.0),
        version="cdd@24C-v1",
        inputs=("tmean",),
        units="°C·day",
        needs_fit=False,
        citation="ASHRAE Fundamentals 2021 Ch. 14; CEA India 24 °C base",
    ),
    "hdd_18c": IndexSpec(
        fn=lambda tmean: hdd(tmean, t_base_c=18.0),
        version="hdd@18C-v1",
        inputs=("tmean",),
        units="°C·day",
        needs_fit=False,
        citation="ASHRAE Fundamentals 2021 Ch. 14",
    ),
    "hot_days_40c": IndexSpec(
        fn=lambda tmax: hot_day_count(tmax, threshold_c=40.0, freq="ME"),
        version="hotdays@40-monthly-v1",
        inputs=("tmax",),
        units="days",
        needs_fit=False,
        citation="ETCCDI; Karl et al. 1999",
    ),
    "rx1day": IndexSpec(
        fn=lambda rain: rx1day(rain, freq="ME"),
        version="rx1day-monthly-v1",
        inputs=("rain",),
        units="mm",
        needs_fit=False,
        citation="ETCCDI Rx1day; Zhang et al. 2011",
    ),
    "rx5day": IndexSpec(
        fn=lambda rain: rx5day(rain, freq="ME"),
        version="rx5day-monthly-v1",
        inputs=("rain",),
        units="mm",
        needs_fit=False,
        citation="ETCCDI Rx5day; Zhang et al. 2011",
    ),
    "r95p": IndexSpec(
        fn=r95p,
        version="r95p@train71-10-v1",
        inputs=("rain",),
        units="mm",
        needs_fit=True,
        citation="ETCCDI R95p; percentile fit uses TRAIN_YEARS",
    ),
    "longest_dry_spell": IndexSpec(
        fn=longest_dry_spell,
        version="lds-v1",
        inputs=("rain",),
        units="days",
        needs_fit=False,
        citation=f"IMD rain-day (< {DRY_THRESHOLD_MM} mm dry); run-length",
    ),
    "cdd_wmo": IndexSpec(
        fn=cdd_wmo,
        version="etccdi-cdd-v1",
        inputs=("rain",),
        units="days",
        needs_fit=False,
        citation="ETCCDI Consecutive Dry Days; Zhang et al. 2011",
    ),
    "spi_3": IndexSpec(
        fn=spi,                         # requires a pre-fit SPIFit
        version="spi3-gamma-mixed-v1",
        inputs=("rain",),
        units="σ",
        needs_fit=True,
        citation="McKee et al. 1993; WMO-No. 1090",
    ),
    "spi_6": IndexSpec(
        fn=spi,
        version="spi6-gamma-mixed-v1",
        inputs=("rain",),
        units="σ",
        needs_fit=True,
        citation="McKee et al. 1993; WMO-No. 1090",
    ),
    "spi_12": IndexSpec(
        fn=spi,
        version="spi12-gamma-mixed-v1",
        inputs=("rain",),
        units="σ",
        needs_fit=True,
        citation="McKee et al. 1993; WMO-No. 1090",
    ),
    "spei_6": IndexSpec(
        fn=spei,
        version="spei6-fisk-v1",
        inputs=("rain", "tmax", "tmin"),
        units="σ",
        needs_fit=True,
        citation="Vicente-Serrano et al. 2010",
    ),
    "aridity": IndexSpec(
        fn=aridity_index,
        version="unep-1997-v1",
        inputs=("rain", "et0"),
        units="dimensionless",
        needs_fit=False,
        citation="UNEP 1997 World Atlas of Desertification (2nd ed.)",
    ),
}


__all__ = [
    "IndexSpec",
    "INDEX_REGISTRY",
    "TRAIN_YEARS",
    "VALID_YEARS",
    "LeakageError",
    "assert_train_only",
    "climatology",
    "zscore",
    "valid_fraction",
    "et0_hargreaves",
    "gdd",
    "gdd_for_crop",
    "cdd",
    "hdd",
    "hot_day_count",
    "imd_heatwave_days",
    "rx1day",
    "rx5day",
    "r95p",
    "return_level",
    "longest_dry_spell",
    "cdd_wmo",
    "DRY_THRESHOLD_MM",
    "imd_kerala_onset",
    "regional_onset",
    "OnsetCriterion",
    "IMD_KERALA_CRITERION",
    "spi", "fit_spi", "fit_spi_from_cube", "SPIFit",
    "spei", "fit_spei", "SPEIFit",
    "aridity_index",
    "anomaly",
    "pctile_anomaly",
    "UNEP_CLASSES",
    "ra_table",
    "broadcast_ra_to",
]
