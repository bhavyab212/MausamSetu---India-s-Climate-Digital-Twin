import { MOCK_IST_TIMESTAMP, addDays, type PipelineStatus, type SemanticTone } from "./_shared";

export interface MockScenarioRunRequest {
  baseline_mode: "historical" | "forecast";
  start_date: string;
  days?: number;
  horizon?: number;
  delta_temp_c: number;
  delta_rain_pct: number;
  seasonal_months?: number[];
  label: string;
}

interface MockSeriesBlock { unit: string; basin_mean: Array<number | null> }
export interface MockScenarioRunResponse {
  baseline_mode: "historical" | "forecast";
  start_date_ist: string;
  dates: string[];
  label: string;
  delta_temp_c: number;
  delta_rain_pct: number;
  baseline: Record<string, MockSeriesBlock>;
  scenario: Record<string, MockSeriesBlock>;
  delta: Record<string, MockSeriesBlock>;
  impacts: {
    hydrology: { baseline_total_m3: number; scenario_total_m3: number; delta_m3: number; delta_pct: number };
    heat: { baseline_hot_pixel_days: number; scenario_hot_pixel_days: number; delta_hot_pixel_days: number; baseline_extreme_pixel_days: number; scenario_extreme_pixel_days: number; delta_extreme_pixel_days: number };
    rupee: { total_crore: number; rainfall_component_crore: number; heat_component_crore: number; unit: string };
  };
}

export const scenarioPresetsResponse = {
  presets: [
    { label: "SSP1-2.6", delta_temp_c: 1.5, delta_rain_pct: 3 },
    { label: "SSP2-4.5", delta_temp_c: 2, delta_rain_pct: -5 },
    { label: "SSP3-7.0", delta_temp_c: 3.2, delta_rain_pct: -12 },
    { label: "SSP5-8.5", delta_temp_c: 4.4, delta_rain_pct: -20 },
  ],
};

export const scenarioRunRequest: MockScenarioRunRequest = {
  baseline_mode: "forecast",
  start_date: "2026-07-15",
  horizon: 7,
  delta_temp_c: 2,
  delta_rain_pct: -20,
  seasonal_months: [6, 7, 8, 9],
  label: "CAV-2026-047",
};

const scenarioDates = Array.from({ length: 7 }, (_, day) => addDays("2026-07-15", day));
const baselineRain = [32, 34, 44, 48, 41, 28, 22];
const scenarioRain = baselineRain.map((value) => Number((value * 0.8).toFixed(1)));
const baselineTmax = [30.1, 30.8, 31.6, 32.4, 33.1, 32.6, 31.9];
const scenarioTmax = baselineTmax.map((value) => Number((value + 2).toFixed(1)));
const baselineTmin = [22.1, 22.4, 22.8, 23.2, 23.7, 23.4, 22.9];
const scenarioTmin = baselineTmin.map((value) => Number((value + 2).toFixed(1)));

export const scenarioRunResponse: MockScenarioRunResponse = {
  baseline_mode: "forecast",
  start_date_ist: MOCK_IST_TIMESTAMP,
  dates: scenarioDates,
  label: "CAV-2026-047",
  delta_temp_c: 2,
  delta_rain_pct: -20,
  baseline: {
    rain: { unit: "mm/day", basin_mean: baselineRain },
    tmax: { unit: "degC", basin_mean: baselineTmax },
    tmin: { unit: "degC", basin_mean: baselineTmin },
  },
  scenario: {
    rain: { unit: "mm/day", basin_mean: scenarioRain },
    tmax: { unit: "degC", basin_mean: scenarioTmax },
    tmin: { unit: "degC", basin_mean: scenarioTmin },
  },
  delta: {
    rain: { unit: "mm/day", basin_mean: scenarioRain.map((value, index) => Number((value - baselineRain[index]).toFixed(1))) },
    tmax: { unit: "degC", basin_mean: Array(7).fill(2) },
    tmin: { unit: "degC", basin_mean: Array(7).fill(2) },
  },
  impacts: {
    hydrology: { baseline_total_m3: 210_000_000, scenario_total_m3: 134_400_000, delta_m3: -75_600_000, delta_pct: -36 },
    heat: { baseline_hot_pixel_days: 3, scenario_hot_pixel_days: 11, delta_hot_pixel_days: 8, baseline_extreme_pixel_days: 0, scenario_extreme_pixel_days: 2, delta_extreme_pixel_days: 2 },
    rupee: { total_crore: 462, rainfall_component_crore: 318, heat_component_crore: 144, unit: "₹ crore" },
  },
};

export const scenarioControls = {
  scenario_id: "CAV-2026-047",
  temperature_anomaly_c: 2,
  rainfall_change_pct: -20,
  monsoon_onset_shift_days: 5,
  region: "Cauvery Basin",
  period: "JJAS 2026",
  ensemble_members: 32,
  mc_dropout_enabled: true,
  physics_constraints_enabled: true,
  estimated_runtime_seconds: 155,
} as const;

export const screenshotScenarioPresets = [
  { id: "drought-2016", label: "2016 Drought", tone: "warning" },
  { id: "flood-2019", label: "2019 Flood", tone: "info" },
  { id: "rcp-45", label: "RCP 4.5", tone: "critical" },
] satisfies Array<{ id: string; label: string; tone: SemanticTone }>;

export const runoffResponse = Array.from({ length: 30 }, (_, day) => ({
  day: day + 1,
  baseline_pct_of_normal: Number((100 + Math.sin(day / 3) * 4).toFixed(1)),
  scenario_pct_of_normal: Number((96 - day * 1.52 + Math.sin(day / 4) * 3).toFixed(1)),
}));

export const scenarioAssumptions = [
  "Perturbations applied uniformly across basin",
  "Model coupled response (SCS-CN hydro)",
  "32-member Monte Carlo ensemble",
  "Physics constraints active",
  "Baseline: JJAS 2026 median forecast",
] as const;

export const scenarioSectorImpacts = [
  { label: "Basin runoff", delta: -36, unit: "%", before: "520 MCft", after: "374 MCft", status: "Critical", tone: "critical", series: [520, 498, 471, 438, 412, 392, 374] },
  { label: "Crop stress days", delta: 14, unit: "days", before: "8 days", after: "22 days", status: "Severe", tone: "critical", series: [8, 9, 11, 14, 17, 20, 22] },
  { label: "Heatwave days", delta: 8, unit: "days", before: "3 days", after: "11 days", status: "Elevated", tone: "warning", series: [3, 4, 5, 7, 8, 10, 11] },
  { label: "Flash flood probability", delta: -42, unit: "%", before: "High", after: "Moderate", status: "Improved", tone: "positive", series: [82, 75, 69, 62, 55, 49, 40] },
] satisfies Array<{ label: string; delta: number; unit: string; before: string; after: string; status: string; tone: SemanticTone; series: number[] }>;

export const scenarioKeyFinding = "A 20% rainfall deficit combined with +2 °C warming produces a disproportionate 36% drop in basin runoff — soil moisture crosses the drying threshold.";

export const scenarioPipeline: readonly { label: string; status: PipelineStatus }[] = [
  { label: "CONFIGURE", status: "completed" }, { label: "PERTURB", status: "completed" },
  { label: "PROPAGATE", status: "active" }, { label: "IMPACT", status: "pending" },
  { label: "RENDER", status: "pending" }, { label: "COMPARE", status: "pending" },
];
