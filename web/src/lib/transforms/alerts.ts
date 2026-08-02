import type { AlertItem, Alerts, ThresholdConfig } from "../api/schemas"
import type { Availability, SemanticTone } from "./overview"

export type AlertPresentationSeverity = "Advisory" | "Elevated" | "Severe"

export interface AlertPresentation extends Availability {
  id: string
  type: AlertItem["type"]
  severity: AlertItem["severity"]
  severityLabel: AlertPresentationSeverity
  title: string
  description: string
  observedValue: number | null
  threshold: number | null
  unit: string
  evaluatedDate: string
  dataSource: string
  thresholdSource: string
  tone: SemanticTone
  timeIst: string
}

export interface AlertsView {
  evaluatedDateIst: string | null
  alerts: AlertPresentation[]
  thresholds: ThresholdConfig | null
  thresholdAvailability: Availability
  incident: Availability
  advisories: Availability
}

export function alertSeverityLabel(severity: AlertItem["severity"]): AlertPresentationSeverity {
  if (severity === "critical") return "Severe"
  if (severity === "warning") return "Elevated"
  return "Advisory"
}

export function alertTone(severity: AlertItem["severity"]): SemanticTone {
  return severity === "critical" ? "critical" : severity === "warning" ? "warning" : "info"
}

export function transformAlertItem(item: AlertItem | null | undefined): AlertPresentation | null {
  if (!item) return null
  return {
    id: item.id,
    type: item.type,
    severity: item.severity,
    severityLabel: alertSeverityLabel(item.severity),
    title: item.title,
    description: item.description,
    observedValue: item.observed_value,
    threshold: item.threshold,
    unit: item.unit,
    evaluatedDate: item.evaluated_date,
    dataSource: item.data_source,
    thresholdSource: item.threshold_source,
    tone: alertTone(item.severity),
    timeIst: item.evaluated_date,
    available: true,
    unavailableLabel: null,
  }
}

export function transformAlerts(response: Alerts | null | undefined): AlertsView {
  const alerts = (response?.alerts ?? []).map(transformAlertItem).filter((item): item is AlertPresentation => item !== null)
  return {
    evaluatedDateIst: response?.evaluated_date_ist ?? null,
    alerts,
    thresholds: response?.thresholds ?? null,
    thresholdAvailability: { available: response?.thresholds !== undefined, unavailableLabel: response?.thresholds ? null : "Threshold configuration unavailable" },
    incident: { available: false, unavailableLabel: "Incident command data unavailable from API" },
    advisories: { available: false, unavailableLabel: "Response coordination unavailable from API" },
  }
}
