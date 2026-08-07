"""
whatif.indices.heat_stress — hot-day counts + IMD heatwave days.

Primary sources:

    * ETCCDI hot-day / summer-day counts —
      Karl, Nicholls & Ghazi (1999), *Climatic Change* 42(1), 3-7;
      Zhang et al. (2011), *Wiley Interdisciplinary Reviews: Climate
      Change* 2, 851-870.
    * IMD heatwave criteria —
      IMD's operational definition (`imd.gov.in/section/nhac/dynamic/
      heatwave-criteria.pdf`), reproduced verbatim below:

        HEATWAVE (declared when Tmax ≥ 40 °C in plains, ≥ 30 °C in
        hilly regions, ≥ 37 °C in coastal areas AND departure from
        normal is 4.5–6.4 °C).

        SEVERE HEATWAVE — same regional bases; departure > 6.4 °C OR
        Tmax ≥ 47 °C (absolute criterion, plains).

    ``normal`` here is the DOY climatology from
    :func:`whatif.indices.reference.climatology` — fit on
    ``TRAIN_YEARS`` only, no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .reference import _propagate_attrs


def hot_day_count(tmax: xr.DataArray, threshold_c: float,
                   freq: str = "YE") -> xr.DataArray:
    """Number of days per resample bin with ``tmax > threshold_c``.

    ``freq``: any pandas offset (default annual: "YE"). Use "ME" for
    monthly counts.  Returns dim (time, lat, lon) with the resampled
    time axis.
    """
    is_hot = (tmax > float(threshold_c)).astype("int8")
    out = is_hot.resample(time=freq).sum(skipna=False)
    out.name = f"hotdays_gt{int(threshold_c)}"
    out = _propagate_attrs(out, tmax, f"hotdays@{threshold_c}-v1")
    out.attrs["units"] = "days"
    out.attrs["threshold_c"] = float(threshold_c)
    out.attrs["method"] = "etccdi-hotday-count"
    return out


def imd_heatwave_days(
    tmax: xr.DataArray,
    normal: xr.DataArray,
    plains_absolute_c: float = 40.0,
    hills_absolute_c: float = 30.0,
    coastal_absolute_c: float = 37.0,
    departure_c: float = 4.5,
) -> xr.DataArray:
    """Daily boolean mask of IMD-defined heatwave days.

    Because the master grid doesn't carry a plains/hills/coastal
    classification, this function applies the plains criterion
    (`tmax ≥ 40 °C` AND departure ≥ 4.5 °C). Callers who need the
    zonal thresholds should pass ``plains_absolute_c`` per zone via
    the sector façade (Part 3+) — this keeps the primitive honest
    about not silently swapping zone rules.

    ``normal`` is the DOY climatology; aligned on ``tmax.time.dayofyear``.
    """
    doys = tmax["time.dayofyear"]
    aligned_normal = normal.sel(dayofyear=doys)
    departure = tmax - aligned_normal
    is_hw = (tmax >= float(plains_absolute_c)) & (departure >= float(departure_c))
    is_hw = is_hw.astype("int8")
    is_hw.name = "imd_heatwave_days"
    is_hw = _propagate_attrs(is_hw, tmax, "imd_hw@plains-v1")
    is_hw.attrs["units"] = "boolean (0/1)"
    is_hw.attrs["method"] = "imd-heatwave-plains"
    is_hw.attrs["plains_absolute_c"] = float(plains_absolute_c)
    is_hw.attrs["departure_c"] = float(departure_c)
    return is_hw
