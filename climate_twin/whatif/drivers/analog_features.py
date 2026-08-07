"""
whatif.drivers.analog_features — feature construction for analog distance.

Primary sources:
    * Zorita, E. & von Storch, H. (1999) "The analog method as a simple
      statistical downscaling technique." J. Climate 12:2474-2489.
      §3 explains that features must be region-integrated, not per-cell,
      to avoid noise-drowning.
    * Delle Monache, L. et al. (2013) "Probabilistic weather prediction
      with an analog ensemble." Mon. Weather Rev. 141:3498-3516. §2.2
      on physical feature choice.

Contract:
    * Distance is computed on a per-region, per-window feature vector
      — never per grid cell.
    * Standardisation uses ``TRAIN_YEARS`` statistics only. Fitting on
      VALID_YEARS is silent leakage (Rule 2).
    * Missing values in a year → the year is DROPPED with a WARNING
      in provenance. No imputation.

Version: ``analogfeat-v1`` (bumps when default features change).
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..config.paths import CACHE_DIR
from ..config.region import RegionSpec, apply_region
from ..indices.reference import TRAIN_YEARS

Window = Literal["JJAS", "JJA", "JF", "MAM", "ON", "annual", "custom"]


WINDOW_MONTHS: dict[str, tuple[int, ...]] = {
    "JJAS":   (6, 7, 8, 9),
    "JJA":    (6, 7, 8),
    "JF":     (1, 2),
    "MAM":    (3, 4, 5),
    "ON":     (10, 11),
    "annual": tuple(range(1, 13)),
}


DEFAULT_FEATURES: tuple[str, ...] = (
    "rain_total_std",
    "rain_rx5day_std",
    "cdd_wmo_std",
    "tmax_mean_anom",
    "tmin_mean_anom",
    "onset_offset_days",
    "spi3_regional",
)


@dataclass(frozen=True)
class AnalogSpec:
    """Immutable declaration of an analog problem.

    ``metric``:
        - "euclidean"  — L2 on standardised features
        - "mahalanobis" — L2 in the whitened basis; handles correlated
                           features (a low-rain year is often also a
                           low-SPI year, so mahalanobis is the natural
                           choice for the default 7-feature vector).
        - "cosine"     — 1 - cosine similarity; scale-free.
    """
    region: RegionSpec
    window: Window = "JJAS"
    custom_months: tuple[int, ...] = ()
    features: tuple[str, ...] = DEFAULT_FEATURES
    metric: Literal["euclidean", "mahalanobis", "cosine"] = "mahalanobis"
    train_years: tuple[int, int] = TRAIN_YEARS
    version: str = "analogfeat-v1"

    def months(self) -> tuple[int, ...]:
        if self.window == "custom":
            if not self.custom_months:
                raise ValueError("AnalogSpec(window='custom') requires custom_months")
            return tuple(self.custom_months)
        if self.window not in WINDOW_MONTHS:
            raise ValueError(f"unknown window {self.window!r}")
        return WINDOW_MONTHS[self.window]

    def signature(self) -> str:
        parts = [
            self.region.signature(), self.window,
            ",".join(map(str, self.custom_months)),
            "|".join(self.features), self.metric,
            f"{self.train_years[0]}-{self.train_years[1]}", self.version,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


# ─── Per-year feature extractors ─────────────────────────────────────
def _region_mean_over_window(
    var: str, year: int, months: tuple[int, ...], region: RegionSpec,
    *, agg: Literal["sum", "mean"] = "mean",
) -> float:
    """Compute a single scalar per (year, window, region) from cached
    historical data. Uses IMD sentinel-masked values (drivers.historical
    already masks sentinels)."""
    from datetime import date as _date

    from .historical import get_historical

    start = _date(year, months[0], 1)
    # Last month's last day — cheap way: pick the 28th → get_historical
    # returns a wider window if start > end, so use a safe end.
    end_month = months[-1]
    # Use pandas to get the last day of end_month
    end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    end_d = _date(end.year, end.month, end.day)
    da = get_historical(var, start, end_d)
    # Apply region
    da = apply_region(da, region)
    # Filter to the exact months (get_historical returns a contiguous span)
    times = pd.DatetimeIndex(da["time"].values)
    keep = np.isin(times.month, months)
    da = da.isel(time=np.flatnonzero(keep))
    arr = da.values
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    # Spatial mean per day, then temporal aggregate over days
    with np.errstate(invalid="ignore"):
        daily = np.nanmean(arr, axis=(1, 2))            # (T,)
    finite_daily = np.isfinite(daily)
    if not finite_daily.any():
        return float("nan")
    if agg == "sum":
        return float(np.nansum(daily))
    return float(np.nanmean(daily))


def _rx5day_regional(year: int, months: tuple[int, ...], region: RegionSpec) -> float:
    """Region-mean of the year's max 5-day rainfall total within the window."""
    from datetime import date as _date

    from .historical import get_historical

    start = _date(year, months[0], 1)
    end_month = months[-1]
    end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    end_d = _date(end.year, end.month, end.day)
    da = get_historical("rain", start, end_d)
    da = apply_region(da, region)
    times = pd.DatetimeIndex(da["time"].values)
    keep = np.isin(times.month, months)
    da = da.isel(time=np.flatnonzero(keep))
    # Region mean per day → rolling 5-day sum → max
    with np.errstate(invalid="ignore"):
        daily = np.nanmean(da.values, axis=(1, 2))
    finite = np.isfinite(daily)
    if finite.sum() < 5:
        return float("nan")
    daily = np.nan_to_num(daily, nan=0.0)
    kernel = np.ones(5, dtype=np.float32)
    conv = np.convolve(daily, kernel, mode="valid")
    return float(conv.max())


def _cdd_wmo_regional(year: int, months: tuple[int, ...], region: RegionSpec) -> float:
    """Maximum consecutive-dry-day run within the window on the region-
    mean daily rainfall (dry = < 2.5 mm/day)."""
    from datetime import date as _date

    from ..indices.dry_spell import DRY_THRESHOLD_MM
    from .historical import get_historical

    start = _date(year, months[0], 1)
    end_month = months[-1]
    end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    end_d = _date(end.year, end.month, end.day)
    da = get_historical("rain", start, end_d)
    da = apply_region(da, region)
    times = pd.DatetimeIndex(da["time"].values)
    keep = np.isin(times.month, months)
    da = da.isel(time=np.flatnonzero(keep))
    with np.errstate(invalid="ignore"):
        daily = np.nanmean(da.values, axis=(1, 2))
    dry = (daily < DRY_THRESHOLD_MM) & np.isfinite(daily)
    best = cur = 0
    for x in dry:
        if x:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(best)


def _onset_offset_days_regional(year: int, region: RegionSpec) -> float:
    """Regional onset day-of-year offset vs the region's climatology onset.

    Uses IMD Kerala-style Pai & Rajeevan 2009 criterion:  first day on
    which the 5-day rain sum ≥ 60 mm over the region *and* at least 3
    of the next 5 days remain wet. Simplified regional version — we
    apply the same rule to the region-mean daily rainfall.
    """
    from datetime import date as _date

    from .historical import get_historical

    start = _date(year, 5, 1)                    # May onward
    end = _date(year, 9, 30)
    try:
        da = get_historical("rain", start, end)
    except Exception:
        return float("nan")
    da = apply_region(da, region)
    with np.errstate(invalid="ignore"):
        daily = np.nanmean(da.values, axis=(1, 2))
    daily = np.nan_to_num(daily, nan=0.0)
    T = len(daily)
    if T < 10:
        return float("nan")
    for i in range(T - 5):
        s5 = daily[i:i + 5].sum()
        wet_next = int((daily[i + 5:i + 10] >= 2.5).sum()) if i + 10 <= T else 0
        if s5 >= 60.0 and wet_next >= 3:
            # doy of day i
            times = pd.DatetimeIndex(da["time"].values)
            return float(times[i].dayofyear)
    return float("nan")


# ─── Feature matrix builder ──────────────────────────────────────────
def _feature_cache_path(spec: AnalogSpec, years: tuple[int, int]) -> Path:
    key = f"{spec.signature()}|{years[0]}-{years[1]}"
    d = hashlib.sha256(key.encode()).hexdigest()[:12]
    return CACHE_DIR / "analog_features" / f"features_{d}.pkl"


def _compute_year_features(spec: AnalogSpec, year: int) -> dict[str, float]:
    months = spec.months()
    region = spec.region
    row: dict[str, float] = {}
    if "rain_total_std" in spec.features:
        # Region-mean daily rain summed over the window (raw; standardised later)
        row["rain_total_raw"] = _region_mean_over_window(
            "rain", year, months, region, agg="sum",
        )
    if "rain_rx5day_std" in spec.features:
        row["rain_rx5day_raw"] = _rx5day_regional(year, months, region)
    if "cdd_wmo_std" in spec.features:
        row["cdd_wmo_raw"] = _cdd_wmo_regional(year, months, region)
    if "tmax_mean_anom" in spec.features:
        row["tmax_mean_raw"] = _region_mean_over_window(
            "tmax", year, months, region, agg="mean",
        )
    if "tmin_mean_anom" in spec.features:
        row["tmin_mean_raw"] = _region_mean_over_window(
            "tmin", year, months, region, agg="mean",
        )
    if "onset_offset_days" in spec.features:
        row["onset_doy_raw"] = _onset_offset_days_regional(year, region)
    if "spi3_regional" in spec.features:
        # Regional SPI-3 anchored on the last month of the window,
        # computed from the region-mean rainfall series. We compute a
        # cheap proxy: (rain_total_raw - mean) / std, using train-year
        # stats — same math as SPI on a monthly-aggregated series.
        row["spi3_regional_raw"] = row.get("rain_total_raw", float("nan"))
    return row


def build_feature_matrix(
    spec: AnalogSpec, years: tuple[int, int] | None = None,
    *, use_cache: bool = True,
) -> pd.DataFrame:
    """Compute one row per year in ``years``, standardised against
    ``spec.train_years``.

    Returns
    -------
    pd.DataFrame, index = year, columns = spec.features. Rows with any
    NaN in the raw features are dropped and their years are recorded
    in ``df.attrs["dropped_years"]``.
    """
    y0, y1 = years or (spec.train_years[0], spec.train_years[1] + 12)
    if y0 > y1:
        raise ValueError(f"years range invalid: {y0}..{y1}")

    cache = _feature_cache_path(spec, (y0, y1))
    if use_cache and cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    rows = {}
    dropped = []
    for y in range(y0, y1 + 1):
        r = _compute_year_features(spec, y)
        if any((v is None) or (isinstance(v, float) and not np.isfinite(v))
                for v in r.values()):
            dropped.append(y)
            continue
        rows[y] = r
    if not rows:
        raise ValueError(
            f"build_feature_matrix produced no valid rows for years "
            f"{y0}..{y1}, region={spec.region.kind}:{spec.region.id}. "
            f"Dropped {len(dropped)} due to missing values."
        )

    raw = pd.DataFrame.from_dict(rows, orient="index")
    raw.index.name = "year"

    # Train-only mean/std for standardisation
    tmask = (raw.index >= spec.train_years[0]) & (raw.index <= spec.train_years[1])
    if not tmask.any():
        raise ValueError(
            f"no rows fell in TRAIN_YEARS {spec.train_years}; can't fit "
            "standardisation stats"
        )
    train_stats: dict[str, tuple[float, float]] = {}
    for col in raw.columns:
        vals = raw.loc[tmask, col].to_numpy()
        finite = vals[np.isfinite(vals)]
        m = float(finite.mean())
        s = float(finite.std(ddof=1)) if finite.size >= 2 else 1.0
        s = s if s > 1e-9 else 1.0
        train_stats[col] = (m, s)

    # Build the standardised feature matrix per spec.features
    out = pd.DataFrame(index=raw.index)
    for feat in spec.features:
        if feat == "rain_total_std":
            m, s = train_stats["rain_total_raw"]
            out[feat] = (raw["rain_total_raw"] - m) / s
        elif feat == "rain_rx5day_std":
            m, s = train_stats["rain_rx5day_raw"]
            out[feat] = (raw["rain_rx5day_raw"] - m) / s
        elif feat == "cdd_wmo_std":
            m, s = train_stats["cdd_wmo_raw"]
            out[feat] = (raw["cdd_wmo_raw"] - m) / s
        elif feat == "tmax_mean_anom":
            m, s = train_stats["tmax_mean_raw"]
            out[feat] = (raw["tmax_mean_raw"] - m)          # anomaly, °C — do not divide
        elif feat == "tmin_mean_anom":
            m, s = train_stats["tmin_mean_raw"]
            out[feat] = (raw["tmin_mean_raw"] - m)
        elif feat == "onset_offset_days":
            m, s = train_stats["onset_doy_raw"]
            out[feat] = (raw["onset_doy_raw"] - m)          # days offset — keep in days
        elif feat == "spi3_regional":
            m, s = train_stats["spi3_regional_raw"]
            out[feat] = (raw["spi3_regional_raw"] - m) / s
        else:
            raise ValueError(f"unknown feature {feat!r}")

    out.attrs["spec_signature"] = spec.signature()
    out.attrs["train_stats"] = train_stats
    out.attrs["dropped_years"] = dropped
    out.attrs["spec_version"] = spec.version
    out.attrs["region_signature"] = spec.region.signature()
    out.attrs["window"] = spec.window
    out.attrs["metric"] = spec.metric

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return out


def feature_covariance(matrix: pd.DataFrame, train_years: tuple[int, int]) -> np.ndarray:
    """Return the inverse feature covariance restricted to TRAIN_YEARS.
    Used for Mahalanobis distance. Handles ill-conditioning by a
    small ridge (1e-6 · I) so nearly-collinear features don't blow up."""
    tmask = (matrix.index >= train_years[0]) & (matrix.index <= train_years[1])
    X = matrix.loc[tmask].to_numpy(dtype=np.float64)
    if X.shape[0] < X.shape[1] + 1:
        # Fewer train years than features + 1 → singular; use identity
        return np.eye(X.shape[1])
    cov = np.cov(X, rowvar=False, ddof=1)
    # Ridge for numerical safety
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    return np.linalg.inv(cov)


def target_features_from_row(row: pd.Series, spec: AnalogSpec) -> pd.Series:
    """Given a pre-standardised feature row (Series with spec.features
    keys), return it unchanged. Utility exists for symmetry with the
    pool interface."""
    if not set(spec.features).issubset(row.index):
        missing = set(spec.features) - set(row.index)
        raise KeyError(f"row missing features {sorted(missing)}")
    return row[list(spec.features)].astype(np.float64)
