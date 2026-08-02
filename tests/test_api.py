"""End-to-end contract tests for the MausamSetu FastAPI service.

These tests use the real ``cauvery.nc`` datacube and ``forecaster_best.pt``
checkpoint — no mocks or stubs. The heavy resources are loaded once for the
module via a shared TestClient fixture.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mausamsetu.dashboard.api.main import app
from mausamsetu.dashboard.api.serializers import mask_missing, to_jsonable


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as instance:
        yield instance


# ---------------------------------------------------------------------------
# Serializer contract
# ---------------------------------------------------------------------------


def test_mask_missing_replaces_project_sentinels():
    raw = np.array([1.0, -999.0, 99.9, np.nan, 5.0], dtype=np.float64)
    cleaned = mask_missing(np.ma.array(raw, mask=[False, False, False, False, True]))
    assert cleaned[0] == 1.0
    assert math.isnan(cleaned[1])
    assert math.isnan(cleaned[2])
    assert math.isnan(cleaned[3])
    assert math.isnan(cleaned[4])


def test_to_jsonable_converts_infinities_and_nan():
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(np.array([1.0, np.nan])) == [1.0, None]


# ---------------------------------------------------------------------------
# Health + OpenAPI inventory
# ---------------------------------------------------------------------------


def test_health_reports_real_datacube_state(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["datacube_days"] == 1461
    assert payload["ist_time"].endswith("+05:30")


def test_openapi_lists_all_expected_paths(client):
    doc = client.get("/openapi.json").json()
    expected = {
        "/api/v1/alerts",
        "/api/v1/data/dates",
        "/api/v1/data/grid/{grid_date}",
        "/api/v1/data/timeseries",
        "/api/v1/data/variables",
        "/api/v1/forecast/assimilation/{assim_date}",
        "/api/v1/forecast/{forecast_date}",
        "/api/v1/health",
        "/api/v1/impacts/heat",
        "/api/v1/impacts/hydrology",
        "/api/v1/impacts/rupee",
        "/api/v1/reports/templates",
        "/api/v1/scenarios/presets",
        "/api/v1/scenarios/run",
        "/api/v1/settings/thresholds",
        "/api/v1/state/current",
        "/api/v1/validation/metrics",
        "/api/v1/validation/metrics/date/{query_date}",
    }
    assert expected.issubset(doc["paths"].keys())


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------


def test_state_current_returns_ist_and_completeness(client):
    response = client.get("/api/v1/state/current", params={"date": "2023-07-15"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["date_ist"].endswith("+05:30")
    assert payload["completeness"]["basin_cells"] > 0
    assert 0.0 <= payload["completeness"]["completeness_pct"] <= 100.0
    labels = {kpi["label"]: kpi["unit"] for kpi in payload["kpis"]}
    assert labels["Basin rainfall"] == "mm/day"
    assert labels["Basin Tmax"] == "degC"


def test_state_current_rejects_unknown_date(client):
    response = client.get("/api/v1/state/current", params={"date": "2100-01-01"})
    assert response.status_code == 400
    assert "not in datacube" in response.json()["detail"]


def test_data_dates_has_expected_count(client):
    response = client.get("/api/v1/data/dates")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1461
    assert payload["dates"][0] == "2020-01-01"
    assert payload["dates"][-1] == "2023-12-31"


def test_data_grid_returns_finite_rain_values(client):
    response = client.get("/api/v1/data/grid/2023-07-15", params={"var": "rain"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["unit"] == "mm/day"
    assert payload["variable"] == "rain"
    assert len(payload["lat"]) == 19
    assert len(payload["lon"]) == 17
    # All valid cells should be finite floats
    for row in payload["values"]:
        for cell in row:
            assert cell is None or isinstance(cell, float)


def test_data_grid_rejects_unknown_variable(client):
    response = client.get("/api/v1/data/grid/2023-07-15", params={"var": "does_not_exist"})
    assert response.status_code == 400


def test_data_timeseries_units_match_field_names(client):
    response = client.get(
        "/api/v1/data/timeseries",
        params={"lat": 12.5, "lon": 77.5, "days": 7, "end_date": "2023-07-15"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["dates"]) == 7
    assert len(payload["rain_mm_per_day"]) == 7
    assert len(payload["tmax_c"]) == 7


# ---------------------------------------------------------------------------
# Forecast + assimilation
# ---------------------------------------------------------------------------


def test_forecast_returns_p10_p50_p90_in_physical_units(client):
    response = client.get("/api/v1/forecast/2023-07-15", params={"horizon": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["horizon_days"] == 3
    assert payload["rain"]["unit"] == "mm/day"
    assert payload["tmax"]["unit"] == "degC"
    p50 = payload["rain"]["p50"]
    assert len(p50) == 3 and len(p50[0]) == 19 and len(p50[0][0]) == 17
    # Physics clamp: rain ≥ 0 everywhere
    for t in p50:
        for row in t:
            for value in row:
                assert value is None or value >= 0.0


def test_assimilation_rmse_reduction_is_nonnegative(client):
    response = client.get(
        "/api/v1/forecast/assimilation/2023-07-15",
        params={"var": "rain"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["unit"] == "mm/day"
    assert payload["rmse_after"] <= payload["rmse_before"] + 1e-6


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_scenarios_presets_come_from_config(client):
    response = client.get("/api/v1/scenarios/presets")
    assert response.status_code == 200
    labels = [preset["label"] for preset in response.json()["presets"]]
    assert any(label.startswith("SSP1-2.6") for label in labels)
    assert any(label.startswith("SSP5-8.5") for label in labels)


def test_scenario_historical_mode(client):
    response = client.post("/api/v1/scenarios/run", json={
        "baseline_mode": "historical",
        "start_date": "2023-06-01",
        "days": 14,
        "delta_temp_c": 1.0,
        "delta_rain_pct": -10.0,
        "label": "historical-test",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_mode"] == "historical"
    assert len(payload["dates"]) == 14
    assert payload["baseline"]["rain"]["unit"] == "mm/day"
    assert payload["impacts"]["rupee"]["unit"] == "₹ crore"


def test_scenario_forecast_mode(client):
    response = client.post("/api/v1/scenarios/run", json={
        "baseline_mode": "forecast",
        "start_date": "2023-07-15",
        "horizon": 5,
        "delta_temp_c": 2.0,
        "delta_rain_pct": 5.0,
        "label": "forecast-test",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_mode"] == "forecast"
    assert len(payload["dates"]) == 5


def test_scenario_historical_requires_days(client):
    response = client.post("/api/v1/scenarios/run", json={
        "baseline_mode": "historical",
        "start_date": "2023-06-01",
        "delta_temp_c": 0.0,
        "delta_rain_pct": 0.0,
    })
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Impacts, alerts, settings
# ---------------------------------------------------------------------------


def test_impacts_hydrology_returns_inflow_series(client):
    response = client.get("/api/v1/impacts/hydrology", params={
        "start_date": "2023-06-01", "days": 15,
    })
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["dates"]) == 15
    assert len(payload["inflow_m3_per_day"]) == 15


def test_impacts_heat_uses_config_thresholds(client):
    response = client.get("/api/v1/impacts/heat", params={
        "start_date": "2023-04-01", "days": 30,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["heat_stress_threshold_c"] == 40.0
    assert payload["extreme_heat_threshold_c"] == 45.0


def test_impacts_rupee_bounds_are_finite(client):
    response = client.get("/api/v1/impacts/rupee", params={
        "start_date": "2023-06-01",
        "days": 15,
        "delta_temp_c": 2.0,
        "delta_rain_pct": -20.0,
    })
    assert response.status_code == 200
    payload = response.json()
    assert math.isfinite(payload["total_crore"])
    assert payload["unit"] == "₹ crore"


def test_alerts_expose_threshold_provenance(client):
    response = client.get("/api/v1/alerts", params={"date": "2023-07-15"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluated_date_ist"].endswith("+05:30")
    sources = payload["thresholds"]["sources"]
    assert sources["rain_warning_mm_per_day"] == "train_p95_2020_2021"
    assert sources["heat_critical_c"] == "config.EXTREME_HEAT_TEMP"


def test_alerts_rejects_unknown_date(client):
    response = client.get("/api/v1/alerts", params={"date": "1999-01-01"})
    assert response.status_code == 400


def test_settings_thresholds_are_positive(client):
    response = client.get("/api/v1/settings/thresholds")
    assert response.status_code == 200
    thresholds = response.json()["thresholds"]
    assert thresholds["rain_warning_mm_per_day"] < thresholds["rain_critical_mm_per_day"]
    assert thresholds["heat_warning_c"] == 40.0
    assert thresholds["heat_critical_c"] == 45.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_metrics_bundles_all_three_baselines(client):
    response = client.get("/api/v1/validation/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["variable"] == "rain"
    assert payload["unit"] == "mm/day"
    assert payload["years"] == [2023]
    for key in ("ours", "persistence", "climatology"):
        assert math.isfinite(payload[key]["rmse"])


def test_validation_metrics_lead_zero_reasonable(client):
    response = client.get("/api/v1/validation/metrics", params={"lead_day": 0})
    assert response.status_code == 200


def test_validation_metrics_date_matches_bundle_units(client):
    response = client.get(
        "/api/v1/validation/metrics/date/2023-07-15",
        params={"variable": "rain"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["unit"] == "mm/day"
    assert payload["n_forecast_days"] == 7


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_reports_templates_have_ids(client):
    response = client.get("/api/v1/reports/templates")
    assert response.status_code == 200
    templates = response.json()["templates"]
    ids = {template["id"] for template in templates}
    assert {"weekly_basin_pulse", "scenario_analysis", "impact_assessment"} <= ids


# ---------------------------------------------------------------------------
# Smoke artifacts
# ---------------------------------------------------------------------------


def test_smoke_artifacts_exist():
    directory = Path("L:/MausamSetu/docs/api-smoke")
    for name in (
        "health",
        "state_current",
        "data_dates_head",
        "data_grid_2023_07_15",
        "forecast_2023_07_15",
        "assimilation_2023_07_15",
        "scenarios_run_historical",
        "scenarios_run_forecast",
        "impacts_hydrology",
        "impacts_heat",
        "impacts_rupee",
        "alerts",
        "settings_thresholds",
        "reports_templates",
        "validation_metrics",
        "validation_metrics_date",
    ):
        assert (directory / f"{name}.json").is_file(), f"Missing smoke artifact for {name}"
