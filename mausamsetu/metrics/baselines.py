"""
baselines.py
=============
Reference forecast methods to beat.

If our AI model can't beat these dumb baselines, it's useless. If it does
beat them, that gap = the value our AI adds.

BASELINES IMPLEMENTED
---------------------
1. PERSISTENCE   : tomorrow = today (assume no change)
2. CLIMATOLOGY   : tomorrow = 30-year average for that day-of-year
3. EMA_TREND     : linear extrapolation of last 3-day trend

WHY THIS MATTERS
----------------
When judges see "our RMSE = 8.2 mm", the natural question is
"...vs what?"  A single number in isolation is meaningless.
"RMSE 8.2 vs persistence 11.5 vs climatology 9.8" is credible science.
"""
from __future__ import annotations
import numpy as np
import xarray as xr

from mausamsetu import config


# ============================================================================
# 1. PERSISTENCE — the dumbest baseline
# ============================================================================
def persistence_forecast(input_window: np.ndarray, forecast_days: int) -> np.ndarray:
    """
    "Tomorrow = today" repeated for forecast_days.

    Parameters
    ----------
    input_window : (T_in, C, H, W) — the past window (any C ≥ 1)
    forecast_days : how many days ahead to forecast

    Returns
    -------
    (forecast_days, C, H, W) — the last-day frame repeated
    """
    last_day = input_window[-1]                        # (C, H, W)
    return np.stack([last_day] * forecast_days, axis=0)


# ============================================================================
# 2. CLIMATOLOGY — the "long-term average" baseline
# ============================================================================
def climatology_forecast(
    forecast_dates: np.ndarray,               # array of np.datetime64
    climatology: xr.Dataset,                  # variables with dim `dayofyear`
    variables: list[str] = None,
) -> np.ndarray:
    """
    Return the climatological "normal" for each forecast date.

    climatology dataset should have variables like 'rain_clim', 'tmax_clim', 'tmin_clim'
    each with a `dayofyear` dimension of length 365 (or 366) and (lat, lon).
    """
    variables = variables or ["rain_clim", "tmax_clim", "tmin_clim"]
    days_of_year = [((np.datetime64(d).astype("datetime64[D]").astype(int)
                      - np.datetime64(f"{np.datetime64(d, 'Y')}-01-01").astype("datetime64[D]").astype(int)) + 1)
                    for d in forecast_dates]

    frames = []
    for doy in days_of_year:
        # Fetch that doy's climatology
        # (using .sel over the dayofyear index)
        doy_slice = min(doy, 365)   # cap at 365 (skip leap-day)
        frame = np.stack(
            [climatology[v].sel(dayofyear=doy_slice).values for v in variables],
            axis=0,
        )
        frames.append(frame)
    return np.stack(frames, axis=0)   # (forecast_days, C, H, W)


# ============================================================================
# 3. EMA_TREND — extrapolate the last-3-day linear trend
# ============================================================================
def trend_forecast(input_window: np.ndarray, forecast_days: int, lookback: int = 3) -> np.ndarray:
    """
    Fit a linear trend over the last `lookback` days, extrapolate forward.

    Parameters
    ----------
    input_window : (T_in, C, H, W) — the past window (needs T_in ≥ lookback)
    forecast_days : how many days ahead
    lookback : number of past days for the fit
    """
    T_in, C, H, W = input_window.shape
    lookback = min(lookback, T_in)
    recent = input_window[-lookback:]                   # (lookback, C, H, W)
    x = np.arange(lookback).astype(np.float32)          # (lookback,)

    # Per-pixel linear fit: slope + intercept via numpy over the time axis
    x_mean = x.mean()
    y_mean = recent.mean(axis=0, keepdims=True)          # (1, C, H, W)
    x_dev  = x - x_mean                                   # (lookback,)
    y_dev  = recent - y_mean                              # (lookback, C, H, W)
    # slope = Σ(x_dev * y_dev) / Σ(x_dev²)
    num = (x_dev[:, None, None, None] * y_dev).sum(axis=0)
    den = (x_dev ** 2).sum() + 1e-8
    slope     = num / den                                 # (C, H, W)
    intercept = y_mean[0] - slope * x_mean                # (C, H, W)

    # Extrapolate: y(t) = slope * t + intercept for t = T_in, T_in+1, ...
    forecast = []
    for t in range(T_in, T_in + forecast_days):
        y_t = slope * t + intercept
        forecast.append(y_t)
    return np.stack(forecast, axis=0)                     # (forecast_days, C, H, W)


# ============================================================================
# EVALUATE ALL BASELINES ON THE TEST SET
# ============================================================================
def evaluate_baselines(dataset) -> dict:
    """
    Compute RMSE, MAE, POD/FAR/CSI for each baseline over a whole dataset.

    Parameters
    ----------
    dataset : an instance of CauveryWindowDataset (from preprocess.dataset)

    Returns
    -------
    dict keyed by baseline name, values = dict of metrics
    """
    from mausamsetu.metrics.metrics import mae, rmse, pod, far, csi

    preds_pers = []
    preds_trend = []
    trues = []

    for i in range(len(dataset)):
        x, y = dataset[i]                    # x: (T_in, C_in, H, W), y: (T_out, 3, H, W)
        x_np = x.numpy()                     # PyTorch → numpy
        y_np = y.numpy()

        # persistence: repeat last input frame's first 3 channels (matches output shape)
        # Input channel order: [rain, tmax, tmin, insat_lst, insat_rain]
        # Output channel order: [rain, tmax, tmin]
        input_last = x_np[:, :3]             # take rain/tmax/tmin only (T_in, 3, H, W)
        preds_pers.append(persistence_forecast(input_last, y_np.shape[0]))
        preds_trend.append(trend_forecast(input_last, y_np.shape[0]))
        trues.append(y_np)

    preds_pers = np.concatenate(preds_pers, axis=0)
    preds_trend = np.concatenate(preds_trend, axis=0)
    trues = np.concatenate(trues, axis=0)

    results = {}
    for name, pred in [("persistence", preds_pers), ("trend", preds_trend)]:
        # Overall metrics (all 3 vars combined)
        results[name] = {
            "MAE":  mae(pred, trues),
            "RMSE": rmse(pred, trues),
        }
        # Per-variable metrics (rain only for POD/FAR/CSI in normalized-anomaly space)
        rain_pred = pred[:, 0]
        rain_true = trues[:, 0]
        # threshold ~0 in anomaly space ~ "any positive anomaly"
        results[name]["POD@0"] = pod(rain_pred, rain_true, threshold=0.0)
        results[name]["FAR@0"] = far(rain_pred, rain_true, threshold=0.0)
        results[name]["CSI@0"] = csi(rain_pred, rain_true, threshold=0.0)

    return results


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    from mausamsetu.preprocess.dataset import CauveryWindowDataset
    ds = CauveryWindowDataset(split="test")
    print(f"Evaluating baselines on test set ({len(ds)} windows)...")
    results = evaluate_baselines(ds)
    print("\n=== BASELINE METRICS ===")
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
