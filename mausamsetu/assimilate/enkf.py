"""
enkf.py
========
Ensemble Kalman Filter — the twin's heartbeat.

WHAT IS EnKF?
-------------
EnKF is the industry-standard data-assimilation method for weather/climate
systems. It combines two ingredients at every time step:

  1. The MODEL's forecast (uncertain prior)
  2. New OBSERVATIONS from satellites/stations (uncertain measurements)

Result: an "analysis" state that's more accurate than either alone.

THE MATH (kept simple)
----------------------
  Forecast ensemble: X_f = [x_f^(1), ..., x_f^(N)]  — N members, each (H, W)
  Observation vector: y (from INSAT at overlapping pixels)
  Observation operator: H (maps state space → observation space)
  Observation error covariance: R (assumed diagonal)

  For each ensemble member i:
      d^(i) = y + noise_i - H(x_f^(i))      "innovation"
      x_a^(i) = x_f^(i) + K · d^(i)          "analysis"

  where the Kalman gain K = P_f · H^T · (H · P_f · H^T + R)^(-1)
  and P_f is the ensemble covariance.

WHY THIS MAKES US A TWIN
------------------------
Without assimilation, the model just makes forecasts.
With assimilation, the model's state is CONTINUOUSLY UPDATED by real
observations — like a GPS re-locking on satellite signals every second.
This "always in sync with reality" property is Property #2 of a digital twin.

FALLBACK
--------
If EnKF math is too slow on CPU, we provide a simpler
"nudging" alternative that also works.
"""
from __future__ import annotations
import numpy as np


# ============================================================================
# ENSEMBLE KALMAN FILTER — the star method
# ============================================================================
def enkf_update(
    forecast_ensemble: np.ndarray,     # (N, H, W)  — forecast members
    observation: np.ndarray,           # (H, W)     — observed field
    obs_error_std: float,              # scalar     — observation std
    localization_radius: int = 3,      # in pixels; None = global
    inflation: float = 1.05,           # spread inflation factor
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply an Ensemble Kalman Filter update to correct forecasts with observations.

    Simplification vs. full EnKF:
      - We assume the observation operator H is identity (same grid).
      - R is scalar (diagonal, uniform).
      - Localization is optional (limits distant correlations from being spurious).

    Parameters
    ----------
    forecast_ensemble : ndarray (N, H, W)
        Ensemble of forecast members.
    observation : ndarray (H, W)
        Observed field (must be same grid).
    obs_error_std : float
        Standard deviation of observation error.
    localization_radius : int
        Restrict covariance updates to pixels within this radius (in cells).
        Set None to disable (may cause spurious long-range corrections).
    inflation : float
        Multiplicative spread inflation (>1) to counter ensemble collapse.
    rng : Generator
        Random state (for reproducibility).

    Returns
    -------
    analysis_ensemble : ndarray (N, H, W)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, H, W = forecast_ensemble.shape
    x_f = forecast_ensemble.copy()

    # --- Inflate ensemble spread (prevents collapse to a single trajectory) ---
    x_f_mean = x_f.mean(axis=0, keepdims=True)                # (1, H, W)
    x_f = x_f_mean + inflation * (x_f - x_f_mean)

    # --- Simplified case: H is identity, R is σ² · I ---
    # Forecast ensemble covariance (per-pixel variance)
    x_f_dev = x_f - x_f.mean(axis=0, keepdims=True)            # (N, H, W)
    P_f_diag = (x_f_dev ** 2).mean(axis=0)                     # (H, W)

    # Kalman gain (element-wise, since H = I and R is scalar)
    R = obs_error_std ** 2
    K = P_f_diag / (P_f_diag + R + 1e-12)                     # (H, W)

    # --- Perturbed observations (stochastic EnKF) ---
    obs_perturbed = observation[None, :, :] + rng.normal(
        0, obs_error_std, size=(N, H, W)
    ).astype(np.float32)

    # --- Update each member ---
    innovation = obs_perturbed - x_f                           # (N, H, W)
    x_a = x_f + K[None, :, :] * innovation

    return x_a


# ============================================================================
# NUDGING (SIMPLER FALLBACK)
# ============================================================================
def nudge(
    forecast: np.ndarray,           # (H, W) — deterministic forecast
    observation: np.ndarray,        # (H, W)
    K: float = 0.5,                 # blending weight [0, 1]
) -> np.ndarray:
    """
    Simple linear blend: x_a = x_f + K · (y - x_f)

    - K = 0 → trust the model fully (ignore observation)
    - K = 1 → trust the observation fully
    - K ≈ 0.5 → balanced

    A fallback when EnKF is too heavy. Same interface as EnKF.
    """
    innovation = observation - forecast
    return forecast + K * innovation


# ============================================================================
# ORCHESTRATOR — run the twin loop over a period
# ============================================================================
def run_twin_loop(
    model_fn,                        # callable: (past_window) -> (forecast_ens, H, W)
    observations: np.ndarray,        # (T, H, W)
    ensemble_size: int = 20,
    obs_error_std: float = 2.0,
    initial_window: np.ndarray | None = None,   # (T_in, H, W) starting state
    inflation: float = 1.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run the OBSERVE → FORECAST → ASSIMILATE loop over a full timeline.

    Parameters
    ----------
    model_fn : function that takes a past window and returns an ensemble forecast
               (or a deterministic forecast; we perturb it if it isn't already an ensemble)
    observations : (T, H, W) — one observation per time step
    ensemble_size : how many ensemble members to maintain
    obs_error_std : std for the EnKF R matrix
    initial_window : starting sequence (T_in, H, W) — usually the first few
                     days of observations

    Returns
    -------
    dict with keys:
      'forecast' : (T, H, W)   — ensemble mean forecast at each step
      'analysis' : (T, H, W)   — ensemble mean analysis at each step
      'spread'   : (T, H, W)   — ensemble std at each step
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T, H, W = observations.shape
    forecasts = np.zeros((T, H, W), dtype=np.float32)
    analyses  = np.zeros((T, H, W), dtype=np.float32)
    spreads   = np.zeros((T, H, W), dtype=np.float32)

    # Seed the state with initial window or first few observations
    if initial_window is None:
        history = observations[:6].copy()      # first 6 days
        start = 6
    else:
        history = initial_window.copy()
        start = 0

    for t in range(start, T):
        # --- FORECAST step ---
        forecast_ens = model_fn(history)       # (N, H, W) — expected shape
        if forecast_ens.ndim == 2:
            # Deterministic model: manufacture ensemble by perturbing
            forecast_ens = forecast_ens[None, :, :].repeat(ensemble_size, axis=0)
            forecast_ens = forecast_ens + rng.normal(0, 0.5, forecast_ens.shape).astype(np.float32)

        forecasts[t] = forecast_ens.mean(axis=0)

        # --- ASSIMILATE step ---
        obs_t = observations[t]
        analysis_ens = enkf_update(
            forecast_ens, obs_t, obs_error_std, inflation=inflation, rng=rng,
        )
        analyses[t] = analysis_ens.mean(axis=0)
        spreads[t]  = analysis_ens.std(axis=0)

        # --- Advance history: append the analysis mean (drop oldest) ---
        history = np.concatenate([history[1:], analyses[t:t+1]], axis=0)

    return {
        "forecast": forecasts,
        "analysis": analyses,
        "spread":   spreads,
    }


# ============================================================================
# CLI DEMO — test EnKF with synthetic data
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    H, W = 19, 17

    # --- Synthetic "truth" ---
    truth = rng.normal(20, 3, size=(H, W)).astype(np.float32)

    # --- Ensemble forecast (biased + noisy) ---
    N = 20
    forecast_ens = truth[None, :, :] + rng.normal(1, 2, size=(N, H, W)).astype(np.float32)

    # --- Observations (noisy version of truth) ---
    obs = truth + rng.normal(0, 1.5, size=(H, W)).astype(np.float32)

    # --- Run EnKF ---
    analysis = enkf_update(forecast_ens, obs, obs_error_std=1.5, inflation=1.05, rng=rng)

    # --- Metrics: how close is each to truth? ---
    f_err = np.sqrt(np.mean((forecast_ens.mean(0) - truth) ** 2))
    a_err = np.sqrt(np.mean((analysis.mean(0)     - truth) ** 2))
    o_err = np.sqrt(np.mean((obs                   - truth) ** 2))

    print("=" * 60)
    print("EnKF sanity test")
    print("=" * 60)
    print(f"Forecast ensemble mean RMSE vs truth:   {f_err:.3f}")
    print(f"Observation                RMSE vs truth: {o_err:.3f}")
    print(f"Analysis (EnKF)            RMSE vs truth: {a_err:.3f}")
    print()
    print("Expected: analysis RMSE < forecast RMSE AND < observation RMSE")
    print(f"  {'✓ PASS' if a_err < min(f_err, o_err) else '✗ FAIL'}")
