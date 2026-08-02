import {
  CAUVERY_BASIN_CELLS,
  CAUVERY_LAT,
  CAUVERY_LON,
  MOCK_IST_TIMESTAMP,
  buildCauveryGrid,
  type PipelineStatus,
  type SemanticTone,
} from "./_shared";

export interface MockCurrentState {
  date_ist: string;
  basin_rainfall_mm_per_day: number | null;
  basin_tmax_c: number | null;
  basin_tmin_c: number | null;
  completeness: { valid_cells: number; basin_cells: number; completeness_pct: number };
  kpis: Array<{ label: string; value: number | null; unit: string }>;
}

export interface MockGrid {
  date_ist: string;
  variable: string;
  unit: string;
  lat: number[];
  lon: number[];
  values: Array<Array<number | null>>;
  completeness: MockCurrentState["completeness"];
}

export const overviewCurrentState: MockCurrentState = {
  date_ist: MOCK_IST_TIMESTAMP,
  basin_rainfall_mm_per_day: 32.4,
  basin_tmax_c: 31.8,
  basin_tmin_c: 23.1,
  completeness: { valid_cells: 311, basin_cells: CAUVERY_BASIN_CELLS, completeness_pct: 100 },
  kpis: [
    { label: "Basin rainfall", value: 32.4, unit: "mm/day" },
    { label: "Basin Tmax", value: 31.8, unit: "degC" },
    { label: "Basin Tmin", value: 23.1, unit: "degC" },
    { label: "Data completeness", value: 100, unit: "%" },
  ],
};

export const overviewRainfallGrid: MockGrid = {
  date_ist: MOCK_IST_TIMESTAMP,
  variable: "rain",
  unit: "mm/day",
  lat: CAUVERY_LAT,
  lon: CAUVERY_LON,
  values: buildCauveryGrid(27, 13, 2, 82),
  completeness: overviewCurrentState.completeness,
};

export interface OverviewKpiView {
  id: string;
  label: string;
  value: number;
  unit: string;
  detail: string;
  tone: SemanticTone;
  visual: "sparkline" | "gauge";
  series?: number[];
}

export const overviewKpis: readonly OverviewKpiView[] = [
  { id: "rainfall", label: "BASIN RAINFALL 24H", value: 32.4, unit: "mm", detail: "+78% vs normal", tone: "info", visual: "sparkline", series: [14, 17, 15, 22, 26, 24, 32.4] },
  { id: "tmax", label: "PEAK TMAX", value: 31.8, unit: "°C", detail: "@ Tiruchirappalli", tone: "warning", visual: "sparkline", series: [29.2, 30.1, 29.7, 30.8, 31.2, 30.9, 31.8] },
  { id: "reservoir", label: "RESERVOIR (METTUR)", value: 87, unit: "%", detail: "+18%", tone: "positive", visual: "gauge" },
  { id: "peak", label: "FORECAST PEAK", value: 48, unit: "mm", detail: "Jul 19", tone: "positive", visual: "sparkline", series: [32, 34, 44, 48, 41, 28, 22] },
  { id: "skill", label: "MODEL SKILL", value: 0.52, unit: "CSI", detail: "+29% vs baseline", tone: "positive", visual: "sparkline", series: [0.41, 0.43, 0.44, 0.47, 0.48, 0.5, 0.52] },
];

export const overviewForecast = [
  { date: "2026-07-16", rainfall_p10_mm: 22, rainfall_p50_mm: 32, rainfall_p90_mm: 40, climatology_mm: 18 },
  { date: "2026-07-17", rainfall_p10_mm: 24, rainfall_p50_mm: 34, rainfall_p90_mm: 43, climatology_mm: 18 },
  { date: "2026-07-18", rainfall_p10_mm: 31, rainfall_p50_mm: 44, rainfall_p90_mm: 55, climatology_mm: 18 },
  { date: "2026-07-19", rainfall_p10_mm: 34, rainfall_p50_mm: 48, rainfall_p90_mm: 60, climatology_mm: 18 },
  { date: "2026-07-20", rainfall_p10_mm: 29, rainfall_p50_mm: 41, rainfall_p90_mm: 52, climatology_mm: 18 },
  { date: "2026-07-21", rainfall_p10_mm: 19, rainfall_p50_mm: 28, rainfall_p90_mm: 37, climatology_mm: 18 },
  { date: "2026-07-22", rainfall_p10_mm: 14, rainfall_p50_mm: 22, rainfall_p90_mm: 31, climatology_mm: 18 },
] as const;

export const overviewReservoirs = [
  { id: "kabini", name: "Kabini Reservoir", storage_pct: 87, lat_deg_north: 11.95, lon_deg_east: 76.35 },
  { id: "krs", name: "KRS Reservoir", storage_pct: 82, lat_deg_north: 12.42, lon_deg_east: 76.57 },
  { id: "mettur", name: "Mettur Reservoir", storage_pct: 87, lat_deg_north: 11.8, lon_deg_east: 77.8 },
] as const;

export const overviewSectorStatus: readonly { sector: string; status: string; tone: SemanticTone }[] = [
  { sector: "Water security", status: "Stable", tone: "positive" },
  { sector: "Agriculture", status: "Low risk", tone: "positive" },
  { sector: "Heat stress", status: "Moderate", tone: "warning" },
  { sector: "Flood risk", status: "Elevated", tone: "critical" },
];

export const overviewModelMetrics = [
  { label: "RMSE", value: 8.2, unit: "mm" }, { label: "POD", value: 0.71, unit: "score" },
  { label: "FAR", value: 0.31, unit: "score" }, { label: "CSI", value: 0.52, unit: "score" },
  { label: "ACC", value: 0.68, unit: "score" }, { label: "FSS", value: 0.61, unit: "score" },
] as const;

export const overviewSystemHealth = [
  { source: "INSAT-3DR", status: "Online", detail: "Live", tone: "positive" },
  { source: "IMD grid feed", status: "Synced", detail: "09:34 IST", tone: "positive" },
  { source: "MOSDAC", status: "Active", detail: "Live", tone: "positive" },
  { source: "Assimilation", status: "Complete", detail: "Cycle 46", tone: "positive" },
  { source: "Next cycle", status: "25 min 48 sec", detail: "Countdown", tone: "info" },
  { source: "Latency", status: "42 ms", detail: "API", tone: "positive" },
] as const;

export const overviewRecentAlerts = [
  { id: "heavy-rain", title: "Heavy rain", detail: "Kodagu · >75 mm expected", time_ist: "09:34 IST", tone: "critical" },
  { id: "soil", title: "Soil saturation", detail: "Kabini · 87%", time_ist: "09:12 IST", tone: "warning" },
  { id: "reservoir", title: "Reservoir advisory", detail: "Mettur · +40% inflow", time_ist: "08:47 IST", tone: "info" },
] as const;

export const overviewPipeline: readonly { label: string; status: PipelineStatus }[] = [
  { label: "INGEST", status: "completed" }, { label: "REGRID", status: "completed" },
  { label: "ASSIMILATE", status: "active" }, { label: "FORECAST", status: "pending" },
  { label: "IMPACT", status: "pending" }, { label: "RENDER", status: "pending" },
];
