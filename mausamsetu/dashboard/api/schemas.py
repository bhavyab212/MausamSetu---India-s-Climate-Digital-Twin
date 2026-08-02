"""Pydantic v2 request and response contracts for the dashboard API.

Every numeric field carries a unit either in the field name (``rainfall_mm_per_day``)
or in a sibling ``unit`` field. Missing values are serialized as ``None``.
Timestamps use ISO 8601 with the IST offset (``+05:30``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared configuration + reusable aliases
# ---------------------------------------------------------------------------


class ApiModel(BaseModel):
    """Base configuration for strict, well-typed API contracts."""

    model_config = ConfigDict(extra="forbid")


Value = Annotated[float | None, Field(description="Numeric value or ``null`` for missing/masked cells.")]
Grid2D = list[list[Value]]
GridT2D = list[list[list[Value]]]  # (time, lat, lon)


# ---------------------------------------------------------------------------
# Health / state
# ---------------------------------------------------------------------------


class HealthResponse(ApiModel):
    """Runtime readiness of the trained model and the in-memory datacube."""

    status: str = Field(description="``ok`` when the model and datacube are loaded.")
    model_loaded: bool = Field(description="Whether the trained forecaster is loaded.")
    datacube_days: int = Field(ge=0, description="Number of daily timesteps, in days.")
    ist_time: datetime = Field(description="Current ISO 8601 timestamp with the IST offset.")


class CompletenessStats(ApiModel):
    valid_cells: int = Field(ge=0, description="Finite, non-sentinel cells inside the basin mask.")
    basin_cells: int = Field(ge=0, description="Total basin-mask cells (denominator).")
    completeness_pct: float = Field(ge=0.0, le=100.0, description="Percent complete.")


class CurrentStateKPI(ApiModel):
    label: str
    value: float | None
    unit: str


class CurrentStateResponse(ApiModel):
    date_ist: datetime
    basin_rainfall_mm_per_day: float | None
    basin_tmax_c: float | None
    basin_tmin_c: float | None
    completeness: CompletenessStats
    kpis: list[CurrentStateKPI]


# ---------------------------------------------------------------------------
# Data (dates, variables, grid, timeseries)
# ---------------------------------------------------------------------------


class DatesResponse(ApiModel):
    dates: list[date]
    count: int = Field(ge=0)


class VariableInfo(ApiModel):
    name: str
    unit: str
    description: str
    dtype: str
    dims: list[str]


class VariablesResponse(ApiModel):
    variables: list[VariableInfo]


class GridResponse(ApiModel):
    date_ist: datetime
    variable: str
    unit: str
    lat: list[float]
    lon: list[float]
    values: Grid2D
    completeness: CompletenessStats


class TimeseriesResponse(ApiModel):
    lat_deg_north: float
    lon_deg_east: float
    dates: list[date]
    rain_mm_per_day: list[Value]
    tmax_c: list[Value]
    tmin_c: list[Value]
    insat_lst_c: list[Value]
    insat_rain_mm_per_day: list[Value]


# ---------------------------------------------------------------------------
# Forecast + assimilation
# ---------------------------------------------------------------------------


class ForecastVariableBlock(ApiModel):
    unit: str
    p10: GridT2D
    p50: GridT2D
    p90: GridT2D


class ForecastResponse(ApiModel):
    forecast_start_ist: datetime
    horizon_days: int
    lat: list[float]
    lon: list[float]
    dates: list[date]
    rain: ForecastVariableBlock
    tmax: ForecastVariableBlock
    tmin: ForecastVariableBlock


class AssimilationResponse(ApiModel):
    date_ist: datetime
    variable: str
    unit: str
    lat: list[float]
    lon: list[float]
    observed: Grid2D
    model_mean: Grid2D
    corrected: Grid2D
    rmse_before: float
    rmse_after: float
    reduction_pct: float


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

ScenarioMode = Literal["historical", "forecast"]


class ScenarioPreset(ApiModel):
    label: str
    delta_temp_c: float
    delta_rain_pct: float


class ScenarioPresetsResponse(ApiModel):
    presets: list[ScenarioPreset]


class ScenarioRunRequest(ApiModel):
    baseline_mode: ScenarioMode
    start_date: date
    days: int | None = Field(default=None, ge=1, le=366)
    horizon: int | None = Field(default=None, ge=1, le=7)
    delta_temp_c: float = Field(ge=-5.0, le=8.0)
    delta_rain_pct: float = Field(ge=-80.0, le=80.0)
    seasonal_months: list[int] | None = Field(default=None)
    label: str = Field(default="custom", min_length=1, max_length=64)


class ScenarioImpactHydrology(ApiModel):
    baseline_total_m3: float
    scenario_total_m3: float
    delta_m3: float
    delta_pct: float


class ScenarioImpactHeat(ApiModel):
    baseline_hot_pixel_days: int
    scenario_hot_pixel_days: int
    delta_hot_pixel_days: int
    baseline_extreme_pixel_days: int
    scenario_extreme_pixel_days: int
    delta_extreme_pixel_days: int


class ScenarioImpactRupee(ApiModel):
    total_crore: float
    rainfall_component_crore: float
    heat_component_crore: float
    unit: str = "₹ crore"


class ScenarioImpacts(ApiModel):
    hydrology: ScenarioImpactHydrology
    heat: ScenarioImpactHeat
    rupee: ScenarioImpactRupee


class ScenarioSeriesBlock(ApiModel):
    unit: str
    basin_mean: list[Value]


class ScenarioRunResponse(ApiModel):
    baseline_mode: ScenarioMode
    start_date_ist: datetime
    dates: list[date]
    label: str
    delta_temp_c: float
    delta_rain_pct: float
    baseline: dict[str, ScenarioSeriesBlock]
    scenario: dict[str, ScenarioSeriesBlock]
    delta: dict[str, ScenarioSeriesBlock]
    impacts: ScenarioImpacts


# ---------------------------------------------------------------------------
# Impacts
# ---------------------------------------------------------------------------


class HydrologyResponse(ApiModel):
    start_date_ist: datetime
    dates: list[date]
    inflow_m3_per_day: list[Value]
    total_m3: float


class HeatResponse(ApiModel):
    start_date_ist: datetime
    dates: list[date]
    heat_stress_days: int
    extreme_heat_days: int
    heat_stress_threshold_c: float
    extreme_heat_threshold_c: float


class RupeeResponse(ApiModel):
    start_date_ist: datetime
    dates: list[date]
    total_crore: float
    rainfall_component_crore: float
    heat_component_crore: float
    unit: str = "₹ crore"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class MetricBundle(ApiModel):
    mae: float | None
    rmse: float | None
    bias: float | None
    pod_1mm: float | None = Field(default=None, alias="pod@1mm")
    far_1mm: float | None = Field(default=None, alias="far@1mm")
    csi_1mm: float | None = Field(default=None, alias="csi@1mm")
    hss_1mm: float | None = Field(default=None, alias="hss@1mm")
    pod_10mm: float | None = Field(default=None, alias="pod@10mm")
    far_10mm: float | None = Field(default=None, alias="far@10mm")
    csi_10mm: float | None = Field(default=None, alias="csi@10mm")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ValidationMetricsResponse(ApiModel):
    variable: str
    unit: str
    split: str = "test"
    years: list[int]
    lead_day: int | None
    n_windows: int
    ours: MetricBundle
    persistence: MetricBundle
    climatology: MetricBundle


class ValidationDateMetricsResponse(ApiModel):
    variable: str
    unit: str
    date_ist: datetime
    n_forecast_days: int
    ours: MetricBundle
    persistence: MetricBundle
    climatology: MetricBundle


# ---------------------------------------------------------------------------
# Alerts + settings
# ---------------------------------------------------------------------------

AlertSeverity = Literal["info", "warning", "critical"]
AlertType = Literal["rainfall", "heat", "extreme_heat"]


class AlertItem(ApiModel):
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    observed_value: float
    threshold: float
    unit: str
    evaluated_date: date
    data_source: str
    threshold_source: str


class AlertsResponse(ApiModel):
    evaluated_date_ist: datetime
    alerts: list[AlertItem]
    thresholds: "ThresholdConfig"


class ThresholdConfig(ApiModel):
    rain_warning_mm_per_day: float
    rain_critical_mm_per_day: float
    heat_warning_c: float
    heat_critical_c: float
    sources: dict[str, str]


class ThresholdsResponse(ApiModel):
    thresholds: ThresholdConfig


AlertsResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class ReportTemplate(ApiModel):
    id: str
    name: str
    description: str
    required_fields: list[str]


class ReportTemplatesResponse(ApiModel):
    templates: list[ReportTemplate]
