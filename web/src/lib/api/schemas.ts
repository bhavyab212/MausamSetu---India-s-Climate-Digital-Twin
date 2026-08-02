import { z } from "zod"

const nullableNumber = z.number().nullable()
const grid2d = z.array(z.array(nullableNumber))
const grid3d = z.array(grid2d)

export const healthSchema = z.object({
  status: z.string(),
  model_loaded: z.boolean(),
  datacube_days: z.number().int().nonnegative(),
  ist_time: z.string(),
})
export type Health = z.infer<typeof healthSchema>

export const completenessSchema = z.object({
  valid_cells: z.number().int().nonnegative(),
  basin_cells: z.number().int().nonnegative(),
  completeness_pct: z.number(),
})
export type Completeness = z.infer<typeof completenessSchema>

export const currentStateSchema = z.object({
  date_ist: z.string(),
  basin_rainfall_mm_per_day: nullableNumber,
  basin_tmax_c: nullableNumber,
  basin_tmin_c: nullableNumber,
  completeness: completenessSchema,
  kpis: z.array(
    z.object({
      label: z.string(),
      value: nullableNumber,
      unit: z.string(),
    }),
  ),
})
export type CurrentState = z.infer<typeof currentStateSchema>

export const datesSchema = z.object({
  dates: z.array(z.string()),
  count: z.number().int().nonnegative(),
})
export type Dates = z.infer<typeof datesSchema>

export const variablesSchema = z.object({
  variables: z.array(
    z.object({
      name: z.string(),
      unit: z.string(),
      description: z.string(),
      dtype: z.string(),
      dims: z.array(z.string()),
    }),
  ),
})
export type Variables = z.infer<typeof variablesSchema>

export const gridSchema = z.object({
  date_ist: z.string(),
  variable: z.string(),
  unit: z.string(),
  lat: z.array(z.number()),
  lon: z.array(z.number()),
  values: grid2d,
  completeness: completenessSchema,
})
export type Grid = z.infer<typeof gridSchema>

export const timeseriesSchema = z.object({
  lat_deg_north: z.number(),
  lon_deg_east: z.number(),
  dates: z.array(z.string()),
  rain_mm_per_day: z.array(nullableNumber),
  tmax_c: z.array(nullableNumber),
  tmin_c: z.array(nullableNumber),
  insat_lst_c: z.array(nullableNumber),
  insat_rain_mm_per_day: z.array(nullableNumber),
})
export type Timeseries = z.infer<typeof timeseriesSchema>

const forecastBlock = z.object({
  unit: z.string(),
  p10: grid3d,
  p50: grid3d,
  p90: grid3d,
})

export const forecastSchema = z.object({
  forecast_start_ist: z.string(),
  horizon_days: z.number().int().positive(),
  lat: z.array(z.number()),
  lon: z.array(z.number()),
  dates: z.array(z.string()),
  rain: forecastBlock,
  tmax: forecastBlock,
  tmin: forecastBlock,
})
export type Forecast = z.infer<typeof forecastSchema>

export const assimilationSchema = z.object({
  date_ist: z.string(),
  variable: z.string(),
  unit: z.string(),
  lat: z.array(z.number()),
  lon: z.array(z.number()),
  observed: grid2d,
  model_mean: grid2d,
  corrected: grid2d,
  rmse_before: z.number(),
  rmse_after: z.number(),
  reduction_pct: z.number(),
})
export type Assimilation = z.infer<typeof assimilationSchema>

export const scenarioPresetsSchema = z.object({
  presets: z.array(
    z.object({
      label: z.string(),
      delta_temp_c: z.number(),
      delta_rain_pct: z.number(),
    }),
  ),
})
export type ScenarioPresets = z.infer<typeof scenarioPresetsSchema>

export const scenarioModeSchema = z.enum(["historical", "forecast"])
export type ScenarioMode = z.infer<typeof scenarioModeSchema>

export const scenarioRunRequestSchema = z.object({
  baseline_mode: scenarioModeSchema,
  start_date: z.string(),
  days: z.number().int().min(1).max(366).optional(),
  horizon: z.number().int().min(1).max(7).optional(),
  delta_temp_c: z.number().min(-5).max(8),
  delta_rain_pct: z.number().min(-80).max(80),
  seasonal_months: z.array(z.number().int().min(1).max(12)).optional(),
  label: z.string().min(1).max(64).default("custom"),
})
export type ScenarioRunRequest = z.infer<typeof scenarioRunRequestSchema>

const seriesBlock = z.object({
  unit: z.string(),
  basin_mean: z.array(nullableNumber),
})

export const scenarioRunResponseSchema = z.object({
  baseline_mode: scenarioModeSchema,
  start_date_ist: z.string(),
  dates: z.array(z.string()),
  label: z.string(),
  delta_temp_c: z.number(),
  delta_rain_pct: z.number(),
  baseline: z.record(seriesBlock),
  scenario: z.record(seriesBlock),
  delta: z.record(seriesBlock),
  impacts: z.object({
    hydrology: z.object({
      baseline_total_m3: z.number(),
      scenario_total_m3: z.number(),
      delta_m3: z.number(),
      delta_pct: z.number(),
    }),
    heat: z.object({
      baseline_hot_pixel_days: z.number().int(),
      scenario_hot_pixel_days: z.number().int(),
      delta_hot_pixel_days: z.number().int(),
      baseline_extreme_pixel_days: z.number().int(),
      scenario_extreme_pixel_days: z.number().int(),
      delta_extreme_pixel_days: z.number().int(),
    }),
    rupee: z.object({
      total_crore: z.number(),
      rainfall_component_crore: z.number(),
      heat_component_crore: z.number(),
      unit: z.string(),
    }),
  }),
})
export type ScenarioRunResponse = z.infer<typeof scenarioRunResponseSchema>

export const hydrologySchema = z.object({
  start_date_ist: z.string(),
  dates: z.array(z.string()),
  inflow_m3_per_day: z.array(nullableNumber),
  total_m3: z.number(),
})
export type Hydrology = z.infer<typeof hydrologySchema>

export const heatSchema = z.object({
  start_date_ist: z.string(),
  dates: z.array(z.string()),
  heat_stress_days: z.number().int(),
  extreme_heat_days: z.number().int(),
  heat_stress_threshold_c: z.number(),
  extreme_heat_threshold_c: z.number(),
})
export type Heat = z.infer<typeof heatSchema>

export const rupeeSchema = z.object({
  start_date_ist: z.string(),
  dates: z.array(z.string()),
  total_crore: z.number(),
  rainfall_component_crore: z.number(),
  heat_component_crore: z.number(),
  unit: z.string(),
})
export type Rupee = z.infer<typeof rupeeSchema>

const metricBundleSchema = z.object({
  mae: nullableNumber,
  rmse: nullableNumber,
  bias: nullableNumber,
  "pod@1mm": nullableNumber.optional(),
  "far@1mm": nullableNumber.optional(),
  "csi@1mm": nullableNumber.optional(),
  "hss@1mm": nullableNumber.optional(),
  "pod@10mm": nullableNumber.optional(),
  "far@10mm": nullableNumber.optional(),
  "csi@10mm": nullableNumber.optional(),
})
export type MetricBundle = z.infer<typeof metricBundleSchema>

export const validationMetricsSchema = z.object({
  variable: z.string(),
  unit: z.string(),
  split: z.string(),
  years: z.array(z.number().int()),
  lead_day: z.number().int().nullable(),
  n_windows: z.number().int().nonnegative(),
  ours: metricBundleSchema,
  persistence: metricBundleSchema,
  climatology: metricBundleSchema,
})
export type ValidationMetrics = z.infer<typeof validationMetricsSchema>

export const validationDateMetricsSchema = z.object({
  variable: z.string(),
  unit: z.string(),
  date_ist: z.string(),
  n_forecast_days: z.number().int().positive(),
  ours: metricBundleSchema,
  persistence: metricBundleSchema,
  climatology: metricBundleSchema,
})
export type ValidationDateMetrics = z.infer<typeof validationDateMetricsSchema>

export const alertSeveritySchema = z.enum(["info", "warning", "critical"])
export const alertTypeSchema = z.enum(["rainfall", "heat", "extreme_heat"])

export const alertItemSchema = z.object({
  id: z.string(),
  type: alertTypeSchema,
  severity: alertSeveritySchema,
  title: z.string(),
  description: z.string(),
  observed_value: z.number(),
  threshold: z.number(),
  unit: z.string(),
  evaluated_date: z.string(),
  data_source: z.string(),
  threshold_source: z.string(),
})
export type AlertItem = z.infer<typeof alertItemSchema>

export const thresholdConfigSchema = z.object({
  rain_warning_mm_per_day: z.number(),
  rain_critical_mm_per_day: z.number(),
  heat_warning_c: z.number(),
  heat_critical_c: z.number(),
  sources: z.record(z.string()),
})
export type ThresholdConfig = z.infer<typeof thresholdConfigSchema>

export const alertsSchema = z.object({
  evaluated_date_ist: z.string(),
  alerts: z.array(alertItemSchema),
  thresholds: thresholdConfigSchema,
})
export type Alerts = z.infer<typeof alertsSchema>

export const thresholdsResponseSchema = z.object({
  thresholds: thresholdConfigSchema,
})
export type ThresholdsResponse = z.infer<typeof thresholdsResponseSchema>

export const reportTemplatesSchema = z.object({
  templates: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      description: z.string(),
      required_fields: z.array(z.string()),
    }),
  ),
})
export type ReportTemplates = z.infer<typeof reportTemplatesSchema>
