"""Exercise every /api/v1 endpoint with real data and save the JSON responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from mausamsetu.dashboard.api.main import app  # noqa: E402


OUTPUT_DIR = ROOT / "docs" / "api-smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save(name: str, response) -> None:
    """Persist a JSON response to docs/api-smoke/<name>.json and print a summary."""
    status = response.status_code
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    path = OUTPUT_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump({"status": status, "response": payload}, fp, indent=2, ensure_ascii=False)
    print(f"{status} {name} -> {path.relative_to(ROOT)}")


def summarize(name: str, response) -> None:
    """Save + trim big grids to a preview so smoke files stay reviewable."""
    if response.headers.get("content-type", "").startswith("application/json"):
        payload = response.json()
        for key in ("values", "observed", "model_mean", "corrected"):
            if isinstance(payload, dict) and isinstance(payload.get(key), list):
                payload[key] = f"<omitted {len(payload[key])}x{len(payload[key][0]) if payload[key] else 0} grid>"
        if isinstance(payload, dict):
            for var_key in ("rain", "tmax", "tmin"):
                block = payload.get(var_key)
                if isinstance(block, dict):
                    for stat_key in ("p10", "p50", "p90"):
                        stat = block.get(stat_key)
                        if isinstance(stat, list):
                            block[stat_key] = f"<omitted {len(stat)}-day grid>"
        wrapped = {"status": response.status_code, "response": payload}
        path = OUTPUT_DIR / f"{name}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(wrapped, fp, indent=2, ensure_ascii=False)
        print(f"{response.status_code} {name} -> {path.relative_to(ROOT)}")
    else:
        save(name, response)


def main() -> None:
    with TestClient(app) as client:
        save("health", client.get("/api/v1/health"))
        save("state_current", client.get("/api/v1/state/current"))
        save("state_current_date", client.get("/api/v1/state/current", params={"date": "2023-07-15"}))

        save("data_variables", client.get("/api/v1/data/variables"))
        dates_response = client.get("/api/v1/data/dates")
        payload = dates_response.json()
        preview = {
            "status": dates_response.status_code,
            "response": {
                "count": payload["count"],
                "first": payload["dates"][:3],
                "last": payload["dates"][-3:],
            },
        }
        with (OUTPUT_DIR / "data_dates_head.json").open("w", encoding="utf-8") as fp:
            json.dump(preview, fp, indent=2, ensure_ascii=False)
        print(f"{dates_response.status_code} data_dates_head -> docs/api-smoke/data_dates_head.json")
        summarize("data_grid_2023_07_15", client.get("/api/v1/data/grid/2023-07-15", params={"var": "rain"}))
        save("data_timeseries", client.get(
            "/api/v1/data/timeseries",
            params={"lat": 12.5, "lon": 77.5, "days": 14, "end_date": "2023-07-15"},
        ))

        summarize("forecast_2023_07_15", client.get("/api/v1/forecast/2023-07-15", params={"horizon": 7}))
        summarize("assimilation_2023_07_15", client.get("/api/v1/forecast/assimilation/2023-07-15", params={"var": "rain"}))

        save("scenarios_presets", client.get("/api/v1/scenarios/presets"))
        save("scenarios_run_historical", client.post("/api/v1/scenarios/run", json={
            "baseline_mode": "historical",
            "start_date": "2023-06-01",
            "days": 30,
            "delta_temp_c": 2.0,
            "delta_rain_pct": -20.0,
            "label": "hist-warm-dry",
        }))
        save("scenarios_run_forecast", client.post("/api/v1/scenarios/run", json={
            "baseline_mode": "forecast",
            "start_date": "2023-07-15",
            "horizon": 7,
            "delta_temp_c": 1.5,
            "delta_rain_pct": 10.0,
            "label": "fcast-warm-wet",
        }))

        save("impacts_hydrology", client.get("/api/v1/impacts/hydrology", params={
            "start_date": "2023-06-01", "days": 30,
        }))
        save("impacts_heat", client.get("/api/v1/impacts/heat", params={
            "start_date": "2023-04-01", "days": 60,
        }))
        save("impacts_rupee", client.get("/api/v1/impacts/rupee", params={
            "start_date": "2023-06-01",
            "days": 30,
            "delta_temp_c": 2.0,
            "delta_rain_pct": -20.0,
        }))

        save("alerts", client.get("/api/v1/alerts", params={"date": "2023-07-15"}))
        save("settings_thresholds", client.get("/api/v1/settings/thresholds"))
        save("reports_templates", client.get("/api/v1/reports/templates"))

        print("Warming validation bundle (this may take a few minutes)...")
        save("validation_metrics", client.get("/api/v1/validation/metrics"))
        save("validation_metrics_lead0", client.get("/api/v1/validation/metrics", params={"lead_day": 0}))
        save("validation_metrics_date", client.get(
            "/api/v1/validation/metrics/date/2023-07-15",
            params={"variable": "rain"},
        ))


if __name__ == "__main__":
    main()
