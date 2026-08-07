"""
whatif.indices.aridity — aridity index + anomaly helpers.

Primary sources:
    * UNEP (1997), 'World Atlas of Desertification, 2nd ed.'.
      Aridity Index AI = P_annual / PET_annual with UNEP class
      thresholds:

        AI          class
        < 0.05      hyper-arid
        0.05 – 0.20 arid
        0.20 – 0.50 semi-arid
        0.50 – 0.65 dry sub-humid
        > 0.65      humid

Anomalies use the DOY climatology from
:mod:`whatif.indices.reference.climatology` — TRAIN_YEARS only.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .reference import _propagate_attrs


UNEP_CLASSES = [
    (0.00, 0.05, "hyper-arid"),
    (0.05, 0.20, "arid"),
    (0.20, 0.50, "semi-arid"),
    (0.50, 0.65, "dry sub-humid"),
    (0.65, np.inf, "humid"),
]


def aridity_index(p_annual: xr.DataArray, et0_annual: xr.DataArray) -> xr.DataArray:
    """P/ET0 with a zero-safe denominator. UNEP classes cited above."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p_annual / et0_annual.where(et0_annual > 0)
    out.name = "aridity_index"
    out = _propagate_attrs(out, p_annual, "aridity-index-v1")
    out.attrs["units"] = "dimensionless"
    out.attrs["method"] = "unep-1997"
    return out


def anomaly(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Signed anomaly wrt DOY climatology."""
    doys = da["time.dayofyear"]
    out = da - clim.sel(dayofyear=doys)
    out.name = f"{da.name}_anom"
    out = _propagate_attrs(out, da, "anomaly-v1")
    return out


def pctile_anomaly(da: xr.DataArray, clim_pctiles: xr.DataArray) -> xr.DataArray:
    """Empirical percentile of ``da`` relative to a DOY-percentile
    reference (``clim_pctiles`` with an extra ``quantile`` dim ranging
    0..1). Useful for narrative UI ('this is a 1-in-15-year dry day')."""
    doys = da["time.dayofyear"]
    ref = clim_pctiles.sel(dayofyear=doys)
    # For each cell/day find the fraction of ref quantiles that are ≤ da
    quantile_axis = ref["quantile"]
    diff = ref - da
    below = (diff <= 0).astype("float32")
    frac = below.mean(dim="quantile")
    frac.name = f"{da.name}_pctile"
    frac = _propagate_attrs(frac, da, "pctile-anomaly-v1")
    frac.attrs["units"] = "percentile"
    return frac
