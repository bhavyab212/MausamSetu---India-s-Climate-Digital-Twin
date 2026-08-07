"""
whatif.indices.degree_days — cooling / heating degree days.

Primary source: ASHRAE Handbook — Fundamentals (2021), Ch. 14 §14.6
'Degree-days and hourly cooling and heating loads'. Definitions:

    CDD(T_base) = max(0, Tmean − T_base)         °C·day  (cooling load driver)
    HDD(T_base) = max(0, T_base − Tmean)         °C·day  (heating load driver)

Base temperatures used in practice (all cited in each argument's
docstring so users see them at the call site):

    * 24 °C — CEA / MoP India building-cooling normalisation
      (Central Electricity Authority guidance).
    * 18 °C — international HVAC / ASHRAE default.
    * 15.5 °C — UK Met Office HDD convention (60 °F).

Base is an explicit argument; the defaults reflect the India-first
context. Any downstream comparison across studies must state the base
used or the numbers are meaningless.
"""
from __future__ import annotations

import xarray as xr

from .reference import _propagate_attrs


def cdd(tmean: xr.DataArray, t_base_c: float = 24.0) -> xr.DataArray:
    """Cooling degree days [°C·day]. ``t_base_c=24`` matches CEA India;
    ``t_base_c=18`` matches ASHRAE default."""
    out = (tmean - float(t_base_c)).clip(min=0.0)
    out.name = "cdd"
    out = _propagate_attrs(out, tmean, f"cdd@Tb{t_base_c}-v1")
    out.attrs["units"] = "°C·day"
    out.attrs["t_base_c"] = float(t_base_c)
    out.attrs["method"] = "ashrae-fundamentals-2021-ch14"
    return out


def hdd(tmean: xr.DataArray, t_base_c: float = 18.0) -> xr.DataArray:
    """Heating degree days [°C·day]. ``t_base_c=18`` is the ASHRAE default;
    ``t_base_c=15.5`` matches the UK Met Office (60 °F) convention."""
    out = (float(t_base_c) - tmean).clip(min=0.0)
    out.name = "hdd"
    out = _propagate_attrs(out, tmean, f"hdd@Tb{t_base_c}-v1")
    out.attrs["units"] = "°C·day"
    out.attrs["t_base_c"] = float(t_base_c)
    out.attrs["method"] = "ashrae-fundamentals-2021-ch14"
    return out
