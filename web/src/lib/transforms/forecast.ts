import type { Forecast } from "../api/schemas"

export type NullableGrid = Array<Array<number | null>>
export type ForecastCube = NullableGrid[]

export interface ForecastPoint {
  date: string
  observed: number | null
  p10: number | null
  p50: number | null
  p90: number | null
  climatology: number | null
  available: boolean
  unavailableLabel: string | null
}

export interface SpatialForecastFrame {
  id: string
  date: string
  basin_mean_rainfall_mm: number | null
  values_mm_per_day: NullableGrid
  selected: boolean
  available: boolean
  unavailableLabel: string | null
}

export interface ForecastSummary {
  horizon_days: number
  ensemble_members: number | null
  peak_intensity_mm: number | null
  peak_date: string | null
  confidence: string | null
  model: string | null
  updated_ist: string | null
  available: boolean
  unavailableLabel: string | null
}

export interface ForecastView {
  response: Forecast | null
  summary: ForecastSummary
  basinSeries: ForecastPoint[]
  spatialFrames: SpatialForecastFrame[]
  ensembleMembers: Array<{ id: string; rainfall_mm: Array<number | null> }>
  ensembleAvailable: boolean
  ensembleUnavailableLabel: string | null
}

export function aggregateForecastGrid(cube: ForecastCube | null | undefined): Array<number | null> {
  if (!cube) return []
  return cube.map((grid) => {
    let total = 0
    let count = 0
    for (const row of grid ?? []) {
      for (const value of row ?? []) {
        if (typeof value === "number" && Number.isFinite(value)) {
          total += value
          count += 1
        }
      }
    }
    return count > 0 ? total / count : null
  })
}

export function getForecastCellValue(
  cube: ForecastCube | null | undefined,
  dayIndex: number,
  rowIndex: number,
  columnIndex: number,
): number | null {
  const value = cube?.[dayIndex]?.[rowIndex]?.[columnIndex]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function peakOf(values: Array<number | null>, dates: string[]): { value: number | null; date: string | null } {
  let value: number | null = null
  let date: string | null = null
  values.forEach((candidate, index) => {
    if (candidate !== null && (value === null || candidate > value)) {
      value = candidate
      date = dates[index] ?? null
    }
  })
  return { value, date }
}

export function transformForecast(response: Forecast | null | undefined): ForecastView {
  const forecast = response ?? null
  if (!forecast) {
    return {
      response: null,
      summary: { horizon_days: 0, ensemble_members: null, peak_intensity_mm: null, peak_date: null, confidence: null, model: null, updated_ist: null, available: false, unavailableLabel: "Forecast unavailable" },
      basinSeries: [], spatialFrames: [], ensembleMembers: [], ensembleAvailable: false, ensembleUnavailableLabel: "Ensemble members unavailable from API",
    }
  }

  const p10 = aggregateForecastGrid(forecast.rain.p10)
  const p50 = aggregateForecastGrid(forecast.rain.p50)
  const p90 = aggregateForecastGrid(forecast.rain.p90)
  const peak = peakOf(p50, forecast.dates)
  const basinSeries = forecast.dates.map((date, index): ForecastPoint => {
    const values = [p10[index] ?? null, p50[index] ?? null, p90[index] ?? null]
    const available = values.some((value) => value !== null)
    return { date, observed: null, p10: values[0], p50: values[1], p90: values[2], climatology: null, available, unavailableLabel: available ? null : "Unavailable" }
  })
  const spatialFrames = forecast.dates.map((date, index): SpatialForecastFrame => {
    const grid = forecast.rain.p50[index] ?? []
    const mean = p50[index] ?? null
    return { id: `day-${index + 1}`, date, basin_mean_rainfall_mm: mean, values_mm_per_day: grid, selected: index === 0, available: mean !== null, unavailableLabel: mean === null ? "Unavailable" : null }
  })
  return {
    response: forecast,
    summary: { horizon_days: forecast.horizon_days, ensemble_members: null, peak_intensity_mm: peak.value, peak_date: peak.date, confidence: null, model: null, updated_ist: forecast.forecast_start_ist, available: true, unavailableLabel: null },
    basinSeries,
    spatialFrames,
    ensembleMembers: [],
    ensembleAvailable: false,
    ensembleUnavailableLabel: "Ensemble members unavailable from API",
  }
}
