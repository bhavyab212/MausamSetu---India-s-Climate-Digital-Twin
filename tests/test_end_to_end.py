"""
test_end_to_end.py
===================
End-to-end integration test — verifies the entire MausamSetu pipeline works.

Run with:
    python -m tests.test_end_to_end
or:
    pytest tests/test_end_to_end.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mausamsetu import config


def test_1_config():
    """Config loads and paths are valid."""
    assert config.PILOT_NAME == "Cauvery Basin"
    assert 10.0 == config.PILOT_LAT_START
    assert config.MASTER_RES == 0.25
    print("✓ test_1_config passed")


def test_2_synthetic_data():
    """Synthetic data can be generated."""
    from mausamsetu.data.synthetic import generate_synthetic_cauvery, cauvery_coords
    lat, lon = cauvery_coords()
    assert len(lat) == 19 and len(lon) == 17, f"Expected 19x17, got {len(lat)}x{len(lon)}"
    assert config.CAUVERY_NC.exists() or (config.SYNTHETIC_DIR / "cauvery_synthetic.nc").exists()
    print("✓ test_2_synthetic_data passed")


def test_3_processed_dataset():
    """Cauvery.nc exists with all required variables."""
    assert config.CAUVERY_NC.exists(), f"{config.CAUVERY_NC} not found. Run build_dataset first."
    ds = xr.open_dataset(config.CAUVERY_NC)
    required = ["rain", "tmax", "tmin", "insat_lst", "insat_rain",
                "rain_anom", "tmax_anom", "tmin_anom"]
    for v in required:
        assert v in ds.data_vars, f"Missing variable: {v}"
    ds.close()
    print(f"✓ test_3_processed_dataset passed — {len(required)} vars found")


def test_4_pytorch_dataset():
    """PyTorch dataset yields (input, target) with correct shapes."""
    from mausamsetu.preprocess.dataset import CauveryWindowDataset
    ds = CauveryWindowDataset(split="train")
    x, y = ds[0]
    assert x.shape == (6, 5, 19, 17), f"Bad x shape: {x.shape}"
    assert y.shape == (7, 3, 19, 17), f"Bad y shape: {y.shape}"
    assert x.dtype == torch.float32
    print(f"✓ test_4_pytorch_dataset passed — {len(ds)} training windows")


def test_5_forecaster_forward():
    """Forecaster model runs a forward pass."""
    from mausamsetu.model.forecaster import MausamSetuForecaster
    model = MausamSetuForecaster()
    x = torch.randn(2, 6, 5, 19, 17)
    y = model(x)
    assert y.shape == (2, 7, 3, 19, 17), f"Bad output: {y.shape}"
    print(f"✓ test_5_forecaster_forward passed — {model.count_parameters():,} params")


def test_6_loss():
    """Physics-informed loss computes without error."""
    from mausamsetu.model.loss import PhysicsInformedLoss
    loss_fn = PhysicsInformedLoss()
    y_pred = torch.randn(2, 7, 3, 19, 17)
    y_true = torch.randn(2, 7, 3, 19, 17)
    rain_prob = torch.sigmoid(torch.randn(2, 7, 1, 19, 17))
    rain_bin = (y_true[:, :, 0:1] > 0).float()
    losses = loss_fn(y_pred, y_true, rain_prob=rain_prob, rain_true_binary=rain_bin)
    assert "total" in losses and losses["total"].item() >= 0
    print(f"✓ test_6_loss passed — total loss = {losses['total'].item():.4f}")


def test_7_baselines():
    """Baselines produce meaningful metrics."""
    from mausamsetu.metrics.baselines import evaluate_baselines
    from mausamsetu.preprocess.dataset import CauveryWindowDataset
    ds = CauveryWindowDataset(split="val")  # smaller = faster
    results = evaluate_baselines(ds)
    assert "persistence" in results and "trend" in results
    print(f"✓ test_7_baselines passed — persistence RMSE = {results['persistence']['RMSE']:.4f}")


def test_8_enkf():
    """EnKF assimilation produces analysis with error < forecast error."""
    from mausamsetu.assimilate.enkf import enkf_update
    rng = np.random.default_rng(0)
    truth = rng.normal(20, 3, size=(19, 17)).astype(np.float32)
    forecast_ens = truth[None] + rng.normal(2, 3, size=(20, 19, 17)).astype(np.float32)
    obs = truth + rng.normal(0, 1.5, size=(19, 17)).astype(np.float32)
    analysis = enkf_update(forecast_ens, obs, obs_error_std=1.5, rng=rng)
    err_fc = np.sqrt(np.mean((forecast_ens.mean(0) - truth) ** 2))
    err_an = np.sqrt(np.mean((analysis.mean(0) - truth) ** 2))
    assert err_an < err_fc, f"EnKF didn't improve: {err_fc:.2f} → {err_an:.2f}"
    print(f"✓ test_8_enkf passed — RMSE {err_fc:.2f} → {err_an:.2f}")


def test_9_scenarios():
    """Storyline engine produces perturbed climate."""
    from mausamsetu.storyline.scenario import Scenario, apply_scenario
    ds = xr.open_dataset(config.CAUVERY_NC)
    baseline = ds[["rain", "tmax", "tmin"]].isel(time=slice(0, 30))
    sc = Scenario(delta_temp=+2.0, delta_rain_pct=-15)
    scenario_ds = apply_scenario(baseline, sc)
    diff_temp = float((scenario_ds.tmax - baseline.tmax).mean())
    assert 1.9 < diff_temp < 2.1, f"Bad ΔT: {diff_temp}"
    print(f"✓ test_9_scenarios passed — applied ΔT={diff_temp:.2f}°C")


def test_10_impacts():
    """Impact modules compute basin totals."""
    from mausamsetu.impacts.hydrology import basin_daily_inflow
    from mausamsetu.impacts.heat import heat_stress_days
    from mausamsetu.impacts.rupee_risk import total_rupee_risk
    from mausamsetu.storyline.scenario import Scenario, apply_scenario

    ds = xr.open_dataset(config.CAUVERY_NC)
    baseline = ds[["rain", "tmax", "tmin"]].sel(time=slice("2023-06-01", "2023-09-30"))

    inflow = basin_daily_inflow(baseline.rain)
    assert inflow.sum().item() > 0, "Inflow should be positive"

    sc = Scenario(delta_temp=+2.5, delta_rain_pct=-20)
    scenario = apply_scenario(baseline, sc)
    result = total_rupee_risk(baseline, scenario)
    assert result["total_crore"] > 0, "₹ risk should be positive under bad scenario"
    print(f"✓ test_10_impacts passed — ₹{result['total_crore']:,.0f} crore under bad scenario")


def main():
    print("=" * 70)
    print("MAUSAMSETU — END-TO-END INTEGRATION TEST")
    print("=" * 70)
    print()
    tests = [
        test_1_config,
        test_2_synthetic_data,
        test_3_processed_dataset,
        test_4_pytorch_dataset,
        test_5_forecaster_forward,
        test_6_loss,
        test_7_baselines,
        test_8_enkf,
        test_9_scenarios,
        test_10_impacts,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except Exception as e:
            print(f"✗ {t.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    print()
    print("=" * 70)
    print(f"RESULT: {n_pass}/{len(tests)} tests passed")
    print("=" * 70)
    return n_pass == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
