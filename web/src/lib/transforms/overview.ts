import type { CurrentState, Forecast, Grid, Health } from "../api/schemas"
import { aggregateForecastGrid } from "./forecast"

export type SemanticTone = "info" | "positive" | "warning" | "critical" | "neutral"

export interface Availability {
  available: boolean
  unavailableLabel: string | null
}

export interface OverviewKpiView extends Availability {
  id: string
  label: string
  value: number | null
  unit: string
  displayValue: string
  detail: string
  tone: SemanticTone
  visual: "sparkline" | "gauge"
  series: number[]
}

export interface OverviewForecastPoint extends Availability {
  date: string
  rainfall_p10_mm: number | null
  rainfall_p50_mm: number | null
  rainfall_p90_mm: number | null
  climatology_mm: number | null
}

export interface OverviewView {
  updatedIst: string | null
  current: CurrentState | null
  rainfallGrid: Grid | null
  kpis: OverviewKpiView[]
  forecast: OverviewForecastPoint[]
  reservoir: Availability
  advisories: Availability
  systemHealth: Health | null
}

function availableNumber(value: number | null, unit: string): Pick<OverviewKpiView, "available" | "unavailableLabel" | "displayValue"> {
  return value === null
    ? { available: false, unavailableLabel: "Unavailable", displayValue: "Unavailable" }
    : { available: true, unavailableLabel: null, displayValue: `${value} ${unit}`.trim() }
}

function findKpi(state: CurrentState | null, label: string, fallback: number | null, unit: string): { value: number | null; unit: string } {
  const match = state?.kpis.find((item) => item.label.toLowerCase() === label.toLowerCase())
  return { value: match?.value ?? fallback, unit: match?.unit ?? unit }
}

export function transformOverview(
  current: CurrentState | null | undefined,
  rainfallGrid: Grid | null | undefined,
  forecast: Forecast | null | undefined,
  health?: Health | null,
): OverviewView {
  const state = current ?? null
  const rainfall = findKpi(state, "Basin rainfall", state?.basin_rainfall_mm_per_day ?? null, "mm/day")
  const tmax = findKpi(state, "Basin Tmax", state?.basin_tmax_c ?? null, "degC")
  const completeness = state?.completeness.completeness_pct ?? null
  const peak = forecast ? aggregateForecastGrid(forecast.rain.p50) : []
  const forecastPoints: OverviewForecastPoint[] = (forecast?.dates ?? []).map((date, index) => {
    const p10 = forecast ? aggregateForecastGrid(forecast.rain.p10)[index] ?? null : null
    const p50 = peak[index] ?? null
    const p90 = forecast ? aggregateForecastGrid(forecast.rain.p90)[index] ?? null : null
    const available = p10 !== null || p50 !== null || p90 !== null
    return {
      date,
      rainfall_p10_mm: p10,
      rainfall_p50_mm: p50,
      rainfall_p90_mm: p90,
      climatology_mm: null,
      available,
      unavailableLabel: available ? null : "Unavailable",
    }
  })

  const kpis: OverviewKpiView[] = [
    { id: "rainfall", label: "BASIN RAINFALL 24H", value: rainfall.value, unit: rainfall.unit, detail: state ? "Current basin state" : "Unavailable", tone: "info", visual: "sparkline", series: rainfall.value === null ? [] : [rainfall.value], ...availableNumber(rainfall.value, rainfall.unit) },
    { id: "tmax", label: "PEAK TMAX", value: tmax.value, unit: tmax.unit, detail: state ? "Current basin state" : "Unavailable", tone: "warning", visual: "sparkline", series: tmax.value === null ? [] : [tmax.value], ...availableNumber(tmax.value, tmax.unit) },
    { id: "reservoir", label: "RESERVOIR (METTUR)", value: null, unit: "%", detail: "Reservoir data unavailable from API", tone: "neutral", visual: "gauge", series: [], available: false, unavailableLabel: "Unavailable", displayValue: "Unavailable" },
    { id: "peak", label: "FORECAST PEAK", value: forecastPoints.reduce<number | null>((max, point) => point.rainfall_p50_mm !== null && (max === null || point.rainfall_p50_mm > max) ? point.rainfall_p50_mm : max, null), unit: forecast?.rain.unit ?? "mm/day", detail: forecast ? "Basin mean p50" : "Forecast unavailable", tone: "positive", visual: "sparkline", series: peak.filter((value): value is number => value !== null), ...availableNumber(forecastPoints.reduce<number | null>((max, point) => point.rainfall_p50_mm !== null && (max === null || point.rainfall_p50_mm > max) ? point.rainfall_p50_mm : max, null), forecast?.rain.unit ?? "mm/day") },
    { id: "skill", label: "MODEL SKILL", value: null, unit: "CSI", detail: "Validation metrics unavailable", tone: "neutral", visual: "sparkline", series: [], available: false, unavailableLabel: "Unavailable", displayValue: "Unavailable" },
  ]

  if (completeness !== null) {
    kpis.push({ id: "completeness", label: "DATA COMPLETENESS", value: completeness, unit: "%", detail: "Current basin state", tone: completeness >= 95 ? "positive" : "warning", visual: "gauge", series: [completeness], ...availableNumber(completeness, "%") })
  }

  return {
    updatedIst: state?.date_ist ?? forecast?.forecast_start_ist ?? health?.ist_time ?? null,
    current: state,
    rainfallGrid: rainfallGrid ?? null,
    kpis,
    forecast: forecastPoints,
    reservoir: { available: false, unavailableLabel: "Unavailable from API" },
    advisories: { available: false, unavailableLabel: "Unavailable from API" },
    systemHealth: health ?? null,
  }
}
