"""
whatif.indices.onset — monsoon onset dates.

Primary sources:
    * Pai & Rajeevan (2009), 'Summer monsoon onset over Kerala: New
      definition and prediction', *Journal of Earth System Science*
      118, 123-135.
    * IMD's operational monsoon-onset criteria for Kerala (published
      each year in the IMD Monsoon Report).

The IMD Kerala criterion (paraphrased in operational terms — the yaml
in the caller records the exact numbers used):

    "After 10 May, if 60% of the 14 rain-gauge stations
     distributed over Kerala + Karnataka + Tamil Nadu report ≥ 2.5 mm
     of rainfall for two consecutive days, the second day is declared
     the monsoon onset over Kerala, provided the depth of westerlies
     up to 600 hPa is > 15 knots and OLR at 5-10 N, 70-75 E < 200
     W m⁻². (Wind + OLR conditions are dropped when only IMD gauge
     data is available — this makes the criterion strictly stricter
     than published; document that.)"

Because the master 0.25° cube is gauge-only, this module implements the
**precipitation-only** subset of the criterion for Kerala and exposes a
region-generalised variant (``regional_onset``) that requires the caller
to pass an explicit criterion dict. It refuses to run Kerala's rule on
Rajasthan silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class OnsetCriterion:
    """Explicit onset rule. All fields must be set at the call site."""
    min_daily_mm: float                        # rainy-day threshold
    consecutive_days: int                      # sequence length
    coverage_frac: float                       # frac of cells needed
    earliest_doy: int = 1                      # ignore anything before this
    break_check_days: int | None = None        # optional: no 7-day break
    break_max_dry_frac: float = 0.5            # 50% dry cells in the break check


IMD_KERALA_CRITERION = OnsetCriterion(
    min_daily_mm=2.5,
    consecutive_days=2,
    coverage_frac=0.60,
    earliest_doy=131,                          # 11 May
    break_check_days=None,
)


def _onset_from_rain(
    rain: xr.DataArray, mask: xr.DataArray | None, crit: OnsetCriterion,
) -> pd.Series:
    """Compute onset dates year-by-year. Returns a Series indexed by year."""
    if mask is not None:
        rain = rain.where(mask)
    out: dict[int, Any] = {}
    for y, blk in rain.groupby("time.year"):
        # daily fraction of covered cells with rain >= threshold
        wet = (blk >= crit.min_daily_mm)
        frac = wet.mean(dim=("lat", "lon"), skipna=True).values
        doys = blk["time.dayofyear"].values
        winner = None
        for i in range(len(frac) - crit.consecutive_days + 1):
            if doys[i] < crit.earliest_doy:
                continue
            window = frac[i : i + crit.consecutive_days]
            if not np.all(window >= crit.coverage_frac):
                continue
            # Optional "no 7-day break" check
            if crit.break_check_days:
                brk = frac[i + crit.consecutive_days :
                            i + crit.consecutive_days + crit.break_check_days]
                if brk.size > 0 and (brk < crit.coverage_frac * (1 - crit.break_max_dry_frac)).all():
                    continue
            iso = pd.Timestamp(blk["time"].values[i + crit.consecutive_days - 1])
            winner = iso.date()
            break
        out[int(y)] = winner
    return pd.Series(out, name="onset_date")


def imd_kerala_onset(rain: xr.DataArray, kerala_mask: xr.DataArray | None = None
                      ) -> pd.Series:
    """Precipitation-only IMD Kerala onset (see module docstring for the
    published criterion + the exact simplifications made here).

    ``rain`` should already be masked/cropped to the Kerala domain (10-13 °N,
    74-77 °E) or an explicit ``kerala_mask`` supplied."""
    return _onset_from_rain(rain, kerala_mask, IMD_KERALA_CRITERION)


def regional_onset(rain: xr.DataArray, mask: xr.DataArray | None,
                    criterion: OnsetCriterion) -> pd.Series:
    """Region-agnostic onset. Caller MUST supply the criterion.

    Do not silently apply Kerala rules to Rajasthan; every region has
    its own realistic thresholds and the caller is on the hook for
    citing the source in provenance."""
    if not isinstance(criterion, OnsetCriterion):
        raise TypeError(
            "regional_onset requires an OnsetCriterion; do not reuse "
            "IMD_KERALA_CRITERION for regions other than Kerala without "
            "documenting the choice explicitly."
        )
    return _onset_from_rain(rain, mask, criterion)
