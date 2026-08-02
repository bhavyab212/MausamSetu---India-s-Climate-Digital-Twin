import type { Assimilation, Grid, Timeseries } from "../api/schemas"
import type { Availability, SemanticTone } from "./overview"
import type { NullableGrid } from "./forecast"

export interface MapLayer extends Availability {
  id: string
  label: string
  enabled: boolean
  selected: boolean
}

export interface SelectedMapMetric extends Availability {
  label: string
  value: number | null
  unit: string
  tone: SemanticTone
}

export interface SelectedMapCell extends Availability {
  place: string
  kind: string
  lat_deg_north: number | null
  lon_deg_east: number | null
  metrics: SelectedMapMetric[]
  timeseries: Timeseries | null
  anomaly_mm: number | null
  anomaly_pct: number | null
  status: string
}

export interface MapView {
  rainfallGrid: Grid | null
  assimilation: Assimilation | null
  selectedCell: SelectedMapCell
  layers: MapLayer[]
  reservoirs: Availability
  wind: Availability
  timeline: Array<{ date: string; kind: "observed" | "forecast"; selected: boolean }>
}

const layerDefinitions = [
  ["rain", "Rainfall", true], ["wind", "Wind flow", false], ["tmax", "Max temperature", false], ["tmin", "Min temperature", false], ["lst", "INSAT LST", false], ["reservoirs", "Reservoirs", false], ["rivers", "Rivers", false], ["soil", "Soil moisture", false], ["flood", "Flood risk zones", false],
] as const

export function findNearestGridCell(grid: Grid | null | undefined, latitude: number, longitude: number): { row: number; column: number; lat: number | null; lon: number | null } | null {
  if (!grid || grid.lat.length === 0 || grid.lon.length === 0 || !Number.isFinite(latitude) || !Number.isFinite(longitude)) return null
  const row = grid.lat.reduce((best, value, index) => Math.abs(value - latitude) < Math.abs(grid.lat[best] - latitude) ? index : best, 0)
  const column = grid.lon.reduce((best, value, index) => Math.abs(value - longitude) < Math.abs(grid.lon[best] - longitude) ? index : best, 0)
  return { row, column, lat: grid.lat[row] ?? null, lon: grid.lon[column] ?? null }
}

export function gridToSelectedCell(grid: Grid | null | undefined, latitude: number, longitude: number): number | null {
  const cell = findNearestGridCell(grid, latitude, longitude)
  const value = cell ? grid?.values[cell.row]?.[cell.column] : null
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export function timeseriesValueAt(timeseries: Timeseries | null | undefined, variable: "rain" | "tmax" | "tmin" | "insat_lst" | "insat_rain", index: number): number | null {
  const values = variable === "rain" ? timeseries?.rain_mm_per_day : variable === "tmax" ? timeseries?.tmax_c : variable === "tmin" ? timeseries?.tmin_c : variable === "insat_lst" ? timeseries?.insat_lst_c : timeseries?.insat_rain_mm_per_day
  const value = values?.[index]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export function transformMap(
  rainfallGrid: Grid | null | undefined,
  timeseries: Timeseries | null | undefined,
  assimilation?: Assimilation | null,
  selectedLatitude?: number,
  selectedLongitude?: number,
): MapView {
  const grid = rainfallGrid ?? null
  const latitude = selectedLatitude ?? timeseries?.lat_deg_north ?? NaN
  const longitude = selectedLongitude ?? timeseries?.lon_deg_east ?? NaN
  const nearest = findNearestGridCell(grid, latitude, longitude)
  const selectedLat = timeseries?.lat_deg_north ?? nearest?.lat ?? null
  const selectedLon = timeseries?.lon_deg_east ?? nearest?.lon ?? null
  const metric = (label: string, value: number | null, unit: string, tone: SemanticTone): SelectedMapMetric => ({ label, value, unit, tone, available: value !== null, unavailableLabel: value === null ? "Unavailable" : null })
  const latest = (variable: Parameters<typeof timeseriesValueAt>[1]): number | null => timeseries && timeseries.dates.length > 0 ? timeseriesValueAt(timeseries, variable, timeseries.dates.length - 1) : null
  const selectedCell: SelectedMapCell = {
    place: "Selected grid cell",
    kind: "Grid cell",
    lat_deg_north: selectedLat,
    lon_deg_east: selectedLon,
    metrics: [metric("Rainfall", gridToSelectedCell(grid, latitude, longitude), grid?.unit ?? "mm/day", "info"), metric("Tmax", latest("tmax"), "degC", "warning"), metric("Tmin", latest("tmin"), "degC", "info"), metric("INSAT LST", latest("insat_lst"), "degC", "warning")],
    timeseries: timeseries ?? null,
    anomaly_mm: null,
    anomaly_pct: null,
    status: grid ? "Current grid value" : "Unavailable",
    available: Boolean(grid || timeseries),
    unavailableLabel: grid || timeseries ? null : "Selected cell data unavailable",
  }
  const layers: MapLayer[] = layerDefinitions.map(([id, label, enabled], index) => {
    const available = id === "rain" ? grid !== null : id === "tmax" || id === "tmin" || id === "lst" ? timeseries !== null && timeseries !== undefined : false
    return { id, label, enabled: enabled && available, selected: index === 0 && available, available, unavailableLabel: available ? null : "Unavailable from API" }
  })
  const dates = timeseries?.dates ?? (grid ? [grid.date_ist.slice(0, 10)] : [])
  return { rainfallGrid: grid, assimilation: assimilation ?? null, selectedCell, layers, reservoirs: { available: false, unavailableLabel: "Reservoir data unavailable from API" }, wind: { available: false, unavailableLabel: "Wind data unavailable from API" }, timeline: dates.map((date, index) => ({ date, kind: index === dates.length - 1 ? "observed" : "observed", selected: index === dates.length - 1 })) }
}

export function forecastGridForSelectedCell(cube: NullableGrid[] | null | undefined, grid: Grid | null | undefined, latitude: number, longitude: number): Array<number | null> {
  const cell = findNearestGridCell(grid, latitude, longitude)
  if (!cell || !cube) return []
  return cube.map((day) => {
    const value = day?.[cell.row]?.[cell.column]
    return typeof value === "number" && Number.isFinite(value) ? value : null
  })
}
