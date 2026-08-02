"""Report template metadata routes."""

from __future__ import annotations

from fastapi import APIRouter

from mausamsetu.dashboard.api.schemas import (
    ReportTemplate,
    ReportTemplatesResponse,
)

router = APIRouter(tags=["reports"])

TEMPLATES = [
    ReportTemplate(
        id="weekly_basin_pulse",
        name="Weekly Basin Pulse",
        description="Rolling seven-day summary of rainfall, temperature, forecast, and alerts.",
        required_fields=[
            "state.current",
            "forecast.rain.p50",
            "validation.metrics",
            "alerts",
        ],
    ),
    ReportTemplate(
        id="scenario_analysis",
        name="Scenario Analysis",
        description="Baseline vs IPCC SSP storyline comparison with hydrology and heat impacts.",
        required_fields=[
            "scenarios.presets",
            "scenarios.run",
            "impacts.hydrology",
            "impacts.heat",
        ],
    ),
    ReportTemplate(
        id="impact_assessment",
        name="Impact Assessment",
        description="Rupee-denominated basin-level impact aggregation for a delta pair.",
        required_fields=[
            "impacts.rupee",
            "impacts.heat",
            "impacts.hydrology",
        ],
    ),
]


@router.get("/reports/templates", response_model=ReportTemplatesResponse)
def report_templates() -> ReportTemplatesResponse:
    return ReportTemplatesResponse(templates=TEMPLATES)
