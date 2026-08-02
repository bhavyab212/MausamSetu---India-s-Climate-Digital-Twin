import type { ScenarioPresets, ScenarioRunRequest, ScenarioRunResponse } from "../api/schemas"
import type { Availability, SemanticTone } from "./overview"

export interface ScenarioPresetView {
  id: string
  label: string
  delta_temp_c: number
  delta_rain_pct: number
  available: boolean
  unavailableLabel: string | null
}

export interface ScenarioSeriesPoint {
  date: string
  baseline: number | null
  scenario: number | null
  delta: number | null
  unit: string
  available: boolean
  unavailableLabel: string | null
}

export interface ScenarioImpactView extends Availability {
  label: string
  delta: number | null
  unit: string
  before: string
  after: string
  status: string
  tone: SemanticTone
  series: Array<number | null>
}

export interface ScenarioView {
  response: ScenarioRunResponse | null
  request: ScenarioRunRequest | null
  presets: ScenarioPresetView[]
  series: Record<string, ScenarioSeriesPoint[]>
  impacts: {
    hydrology: ScenarioRunResponse["impacts"]["hydrology"] | null
    heat: ScenarioRunResponse["impacts"]["heat"] | null
    rupee: ScenarioRunResponse["impacts"]["rupee"] | null
  }
  spatial: Availability
  advisories: Availability
}

export function transformScenarioPresets(response: ScenarioPresets | null | undefined): ScenarioPresetView[] {
  return (response?.presets ?? []).map((preset, index) => ({
    id: `preset-${index + 1}`,
    label: preset.label,
    delta_temp_c: preset.delta_temp_c,
    delta_rain_pct: preset.delta_rain_pct,
    available: true,
    unavailableLabel: null,
  }))
}

function blockValue(record: Record<string, { unit: string; basin_mean: Array<number | null> }> | undefined, variable: string): { unit: string; values: Array<number | null> } {
  const block = record?.[variable]
  return block ? { unit: block.unit, values: block.basin_mean } : { unit: "", values: [] }
}

export function transformScenarioRun(
  response: ScenarioRunResponse | null | undefined,
  request?: ScenarioRunRequest | null,
  presets?: ScenarioPresets | null,
): ScenarioView {
  const scenario = response ?? null
  const variables = new Set<string>([
    ...Object.keys(scenario?.baseline ?? {}),
    ...Object.keys(scenario?.scenario ?? {}),
    ...Object.keys(scenario?.delta ?? {}),
  ])
  const series: Record<string, ScenarioSeriesPoint[]> = {}
  for (const variable of Array.from(variables)) {
    const baseline = blockValue(scenario?.baseline, variable)
    const scenarioValues = blockValue(scenario?.scenario, variable)
    const delta = blockValue(scenario?.delta, variable)
    const dates = scenario?.dates ?? []
    series[variable] = dates.map((date, index) => {
      const values = [baseline.values[index] ?? null, scenarioValues.values[index] ?? null, delta.values[index] ?? null]
      const available = values.some((value) => value !== null)
      return { date, baseline: values[0], scenario: values[1], delta: values[2], unit: baseline.unit || scenarioValues.unit || delta.unit, available, unavailableLabel: available ? null : "Unavailable" }
    })
  }
  return {
    response: scenario,
    request: request ?? null,
    presets: transformScenarioPresets(presets),
    series,
    impacts: { hydrology: scenario?.impacts.hydrology ?? null, heat: scenario?.impacts.heat ?? null, rupee: scenario?.impacts.rupee ?? null },
    spatial: { available: false, unavailableLabel: "Spatial scenario grids unavailable from API" },
    advisories: { available: false, unavailableLabel: "Scenario advisories unavailable from API" },
  }
}

export function scenarioImpactCards(response: ScenarioRunResponse | null | undefined): ScenarioImpactView[] {
  const impact = response?.impacts
  if (!impact) return []
  const hydrology = impact.hydrology
  const heat = impact.heat
  return [
    { label: "Basin runoff", delta: hydrology.delta_pct, unit: "%", before: `${hydrology.baseline_total_m3} m³`, after: `${hydrology.scenario_total_m3} m³`, status: hydrology.delta_pct < 0 ? "Reduced" : "Improved", tone: hydrology.delta_pct < 0 ? "critical" : "positive", series: [], available: true, unavailableLabel: null },
    { label: "Heat stress days", delta: heat.delta_hot_pixel_days, unit: "days", before: `${heat.baseline_hot_pixel_days} days`, after: `${heat.scenario_hot_pixel_days} days`, status: heat.delta_hot_pixel_days > 0 ? "Elevated" : "Stable", tone: heat.delta_hot_pixel_days > 0 ? "warning" : "positive", series: [], available: true, unavailableLabel: null },
    { label: "Extreme heat days", delta: heat.delta_extreme_pixel_days, unit: "days", before: `${heat.baseline_extreme_pixel_days} days`, after: `${heat.scenario_extreme_pixel_days} days`, status: heat.delta_extreme_pixel_days > 0 ? "Elevated" : "Stable", tone: heat.delta_extreme_pixel_days > 0 ? "critical" : "positive", series: [], available: true, unavailableLabel: null },
    { label: "Economic risk", delta: impact.rupee.total_crore, unit: impact.rupee.unit, before: "Unavailable", after: `${impact.rupee.total_crore} ${impact.rupee.unit}`, status: "Estimated", tone: "warning", series: [], available: true, unavailableLabel: null },
  ]
}
