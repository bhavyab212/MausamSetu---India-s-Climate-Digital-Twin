"use client"

import useSWR, { type SWRConfiguration } from "swr"
import useSWRMutation from "swr/mutation"

import { apiFetcher, apiPost, type QueryParams } from "./client"
import {
  alertsSchema,
  assimilationSchema,
  currentStateSchema,
  datesSchema,
  forecastSchema,
  gridSchema,
  healthSchema,
  heatSchema,
  hydrologySchema,
  reportTemplatesSchema,
  rupeeSchema,
  scenarioPresetsSchema,
  scenarioRunResponseSchema,
  thresholdsResponseSchema,
  timeseriesSchema,
  validationDateMetricsSchema,
  validationMetricsSchema,
  variablesSchema,
  type Alerts,
  type Assimilation,
  type CurrentState,
  type Dates,
  type Forecast,
  type Grid,
  type Health,
  type Heat,
  type Hydrology,
  type ReportTemplates,
  type Rupee,
  type ScenarioPresets,
  type ScenarioRunRequest,
  type ScenarioRunResponse,
  type ThresholdsResponse,
  type Timeseries,
  type ValidationDateMetrics,
  type ValidationMetrics,
  type Variables,
} from "./schemas"

const LIVE_REFRESH: SWRConfiguration = {
  refreshInterval: 60_000,
  revalidateOnFocus: true,
}

const STATIC_CACHE: SWRConfiguration = {
  revalidateOnFocus: false,
  revalidateIfStale: false,
  dedupingInterval: 60_000,
}

function useTypedSWR<T>(
  key: readonly [string, QueryParams?] | null,
  schema: Parameters<typeof apiFetcher>[0],
  config?: SWRConfiguration,
) {
  return useSWR<T>(key, apiFetcher(schema) as never, config)
}

export function useHealth() {
  return useTypedSWR<Health>(["/health"], healthSchema, LIVE_REFRESH)
}

export function useCurrentState(date?: string) {
  const params = date ? { date } : undefined
  return useTypedSWR<CurrentState>(["/state/current", params], currentStateSchema, LIVE_REFRESH)
}

export function useDates() {
  return useTypedSWR<Dates>(["/data/dates"], datesSchema, STATIC_CACHE)
}

export function useVariables() {
  return useTypedSWR<Variables>(["/data/variables"], variablesSchema, STATIC_CACHE)
}

export function useGrid(date: string | null, variable = "rain") {
  const key = date
    ? ([`/data/grid/${date}`, { var: variable }] as const)
    : null
  return useTypedSWR<Grid>(key, gridSchema, STATIC_CACHE)
}

export function useTimeseries(lat: number, lon: number, days: number, endDate?: string) {
  const params = { lat, lon, days, end_date: endDate }
  return useTypedSWR<Timeseries>(["/data/timeseries", params], timeseriesSchema, STATIC_CACHE)
}

export function useForecast(date: string | null, horizon = 7) {
  const key = date ? (["/forecast/" + date, { horizon }] as const) : null
  return useTypedSWR<Forecast>(key, forecastSchema, STATIC_CACHE)
}

export function useAssimilation(date: string | null, variable = "rain") {
  const key = date
    ? ([`/forecast/assimilation/${date}`, { var: variable }] as const)
    : null
  return useTypedSWR<Assimilation>(key, assimilationSchema, STATIC_CACHE)
}

export function useScenariosPresets() {
  return useTypedSWR<ScenarioPresets>(["/scenarios/presets"], scenarioPresetsSchema, STATIC_CACHE)
}

export function useRunScenario() {
  return useSWRMutation<
    ScenarioRunResponse,
    Error,
    "/scenarios/run",
    ScenarioRunRequest
  >("/scenarios/run", (key, { arg }) =>
    apiPost(key, scenarioRunResponseSchema, arg),
  )
}

export function useImpactHydrology(startDate: string, days: number) {
  const params = { start_date: startDate, days }
  return useTypedSWR<Hydrology>(["/impacts/hydrology", params], hydrologySchema, STATIC_CACHE)
}

export function useImpactHeat(startDate: string, days: number) {
  const params = { start_date: startDate, days }
  return useTypedSWR<Heat>(["/impacts/heat", params], heatSchema, STATIC_CACHE)
}

export function useImpactRupee(startDate: string, days: number, deltaTempC: number, deltaRainPct: number) {
  const params = {
    start_date: startDate,
    days,
    delta_temp_c: deltaTempC,
    delta_rain_pct: deltaRainPct,
  }
  return useTypedSWR<Rupee>(["/impacts/rupee", params], rupeeSchema, STATIC_CACHE)
}

export function useValidationMetrics(leadDay?: number) {
  const params = leadDay !== undefined ? { lead_day: leadDay } : undefined
  return useTypedSWR<ValidationMetrics>(
    ["/validation/metrics", params],
    validationMetricsSchema,
    STATIC_CACHE,
  )
}

export function useValidationDateMetrics(date: string | null, variable = "rain") {
  const key = date
    ? ([`/validation/metrics/date/${date}`, { variable }] as const)
    : null
  return useTypedSWR<ValidationDateMetrics>(
    key,
    validationDateMetricsSchema,
    STATIC_CACHE,
  )
}

export function useAlerts(date?: string) {
  const params = date ? { date } : undefined
  return useTypedSWR<Alerts>(["/alerts", params], alertsSchema, LIVE_REFRESH)
}

export function useSettingsThresholds() {
  return useTypedSWR<ThresholdsResponse>(
    ["/settings/thresholds"],
    thresholdsResponseSchema,
    STATIC_CACHE,
  )
}

export function useReportTemplates() {
  return useTypedSWR<ReportTemplates>(
    ["/reports/templates"],
    reportTemplatesSchema,
    STATIC_CACHE,
  )
}
