import type { MetricBundle, ValidationDateMetrics, ValidationMetrics } from "../api/schemas"
import type { Availability, SemanticTone } from "./overview"

export interface ValidationMetricView extends Availability {
  key: string
  label: string
  value: number | null
  unit: string
  displayValue: string
}

export interface BaselineComparisonView extends Availability {
  model: string
  metric: string
  value: number | null
  unit: string
  status: "baseline" | "selected"
  tone: SemanticTone
  displayValue: string
}

export interface ValidationView {
  metrics: ValidationMetrics | null
  dateMetrics: ValidationDateMetrics | null
  metricCards: ValidationMetricView[]
  baselineComparison: BaselineComparisonView[]
  audit: Availability
  scatter: Availability
  spatialError: Availability
}

const metricLabels: Record<string, string> = {
  mae: "MAE", rmse: "RMSE", bias: "Bias", "pod@1mm": "POD >1 mm", "far@1mm": "FAR >1 mm", "csi@1mm": "CSI >1 mm", "hss@1mm": "HSS >1 mm", "pod@10mm": "POD >10 mm", "far@10mm": "FAR >10 mm", "csi@10mm": "CSI >10 mm",
}

export function formatValidationMetric(value: number | null, unit: string): string {
  return value === null ? "Unavailable" : `${value} ${unit}`.trim()
}

export function metricBundleDisplay(bundle: MetricBundle | null | undefined, unit: string): ValidationMetricView[] {
  return Object.entries(metricLabels).map(([key, label]) => {
    const value = bundle?.[key as keyof MetricBundle]
    const numeric = typeof value === "number" && Number.isFinite(value) ? value : null
    return { key, label, value: numeric, unit, displayValue: formatValidationMetric(numeric, unit), available: numeric !== null, unavailableLabel: numeric === null ? "Unavailable" : null }
  })
}

export function toBaselineDisplay(metrics: ValidationMetrics | ValidationDateMetrics | null | undefined): BaselineComparisonView[] {
  if (!metrics) return []
  const rows: Array<[string, MetricBundle]> = [["Persistence", metrics.persistence], ["Climatology", metrics.climatology], ["MausamSetu", metrics.ours]]
  return rows.map(([model, bundle]) => {
    const value = typeof bundle.rmse === "number" && Number.isFinite(bundle.rmse) ? bundle.rmse : null
    const selected = model === "MausamSetu"
    return { model, metric: "RMSE", value, unit: metrics.unit, status: selected ? "selected" : "baseline", tone: selected ? "info" : "neutral", displayValue: formatValidationMetric(value, metrics.unit), available: value !== null, unavailableLabel: value === null ? "Unavailable" : null }
  })
}

export function transformValidation(metrics: ValidationMetrics | null | undefined, dateMetrics?: ValidationDateMetrics | null): ValidationView {
  const value = metrics ?? null
  const dateValue = dateMetrics ?? null
  return {
    metrics: value,
    dateMetrics: dateValue,
    metricCards: metricBundleDisplay(dateValue?.ours ?? value?.ours, dateValue?.unit ?? value?.unit ?? ""),
    baselineComparison: toBaselineDisplay(dateValue ?? value),
    audit: { available: value !== null, unavailableLabel: value ? null : "Validation audit unavailable from API" },
    scatter: { available: false, unavailableLabel: "Observed/predicted scatter data unavailable from API" },
    spatialError: { available: false, unavailableLabel: "Spatial validation error unavailable from API" },
  }
}
