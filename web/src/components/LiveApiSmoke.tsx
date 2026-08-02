"use client"

import { useEffect, useMemo, useState } from "react"
import { Activity } from "lucide-react"

import {
  AlertRow,
  DonutGauge,
  ErrorState,
  GlassCard,
  IncidentBanner,
  InlineLegend,
  MetricTile,
  MiniBarChart,
  MiniSparkline,
  SectionHeader,
  Skeleton,
  StatusPill,
} from "@/components"
import {
  useAlerts,
  useCurrentState,
  useDates,
  useForecast,
  useHealth,
  useReportTemplates,
  useRunScenario,
  useSettingsThresholds,
  useValidationMetrics,
} from "@/lib/api"

const CATALOG_DATE = "2023-07-15"

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--"
  return value.toFixed(digits)
}

function firstFiniteSeries(grid: (number | null)[][][] | undefined, lat: number, lon: number) {
  if (!grid) return []
  return grid.map((frame) => frame[lat]?.[lon] ?? null)
}

function meanOverGrid(frame: (number | null)[][] | undefined): number | null {
  if (!frame) return null
  let sum = 0
  let count = 0
  for (const row of frame) {
    for (const cell of row) {
      if (cell === null || !Number.isFinite(cell)) continue
      sum += cell
      count += 1
    }
  }
  return count === 0 ? null : sum / count
}

export function LiveApiSmoke() {
  const health = useHealth()
  const currentState = useCurrentState(CATALOG_DATE)
  const dates = useDates()
  const forecast = useForecast(CATALOG_DATE, 7)
  const validation = useValidationMetrics()
  const alerts = useAlerts(CATALOG_DATE)
  const thresholds = useSettingsThresholds()
  const reports = useReportTemplates()
  const scenario = useRunScenario()

  const [scenarioLoaded, setScenarioLoaded] = useState(false)
  useEffect(() => {
    if (scenarioLoaded) return
    scenario
      .trigger({
        baseline_mode: "historical",
        start_date: "2023-06-01",
        days: 14,
        delta_temp_c: 1.5,
        delta_rain_pct: -10,
        label: "kitchen-sink",
      })
      .then(() => setScenarioLoaded(true))
      .catch(() => setScenarioLoaded(true))
  }, [scenario, scenarioLoaded])

  const forecastSeries = useMemo(() => {
    if (!forecast.data) return []
    const middleLat = Math.floor(forecast.data.lat.length / 2)
    const middleLon = Math.floor(forecast.data.lon.length / 2)
    return firstFiniteSeries(forecast.data.rain.p50, middleLat, middleLon)
  }, [forecast.data])

  const forecastMean = useMemo(() => {
    if (!forecast.data) return null
    const values = forecast.data.rain.p50
      .map((frame) => meanOverGrid(frame))
      .filter((value): value is number => value !== null)
    if (values.length === 0) return null
    return values.reduce((sum, value) => sum + value, 0) / values.length
  }, [forecast.data])

  const scenarioBars = useMemo(() => {
    if (!scenario.data) return []
    return [
      {
        id: "baseline",
        label: "Baseline",
        value: scenario.data.impacts.hydrology.baseline_total_m3 / 1e9,
        tone: "info" as const,
      },
      {
        id: "scenario",
        label: "Scenario",
        value: scenario.data.impacts.hydrology.scenario_total_m3 / 1e9,
        tone: "warning" as const,
      },
    ]
  }, [scenario.data])

  const rainCsi = validation.data?.ours["csi@1mm"] ?? null
  const persistenceCsi = validation.data?.persistence["csi@1mm"] ?? null

  return (
    <section aria-labelledby="live-api-smoke" className="space-y-workspace">
      <SectionHeader
        title="Live API smoke test"
        subtitle="Every /api/v1 endpoint rendered through one shared component with real 2023 data."
        headingLevel={2}
        rightSlot={
          <div className="flex flex-wrap gap-2">
            <StatusPill
              label={health.data?.status === "ok" ? "API online" : "API offline"}
              tone={health.data?.status === "ok" ? "positive" : "critical"}
              size="sm"
            />
            <StatusPill
              label={`${dates.data?.count ?? 0} datacube days`}
              tone="info"
              size="sm"
            />
          </div>
        }
      />

      {health.error ? (
        <ErrorState
          title="Backend unreachable"
          description={
            "Start the FastAPI service (uvicorn on 127.0.0.1:8011) so this section can render. " +
            String((health.error as Error).message ?? "")
          }
        />
      ) : null}

      {health.data && !health.data.model_loaded ? (
        <IncidentBanner
          title="Model checkpoint not loaded"
          description="The forecaster is unavailable, so p10/p50/p90 fields will remain empty."
        />
      ) : null}

      <div className="grid grid-cols-1 gap-column-gap md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Basin rainfall (basin mean)"
          value={formatNumber(currentState.data?.basin_rainfall_mm_per_day, 2)}
          unit="mm/day"
          loading={currentState.isLoading}
          visual={
            <MiniSparkline
              ariaLabel="Live basin rainfall trend"
              points={forecastSeries.filter((value): value is number => value !== null)}
              tone="info"
            />
          }
        />
        <MetricTile
          variant="with-gauge"
          label="Data completeness"
          value={formatNumber(currentState.data?.completeness.completeness_pct, 1)}
          unit="%"
          loading={currentState.isLoading}
          visual={
            <DonutGauge
              value={currentState.data?.completeness.completeness_pct ?? 0}
              ariaLabel="Data completeness for the requested day"
              tone="positive"
            />
          }
        />
        <MetricTile
          variant="with-delta"
          label="Forecast p50 rain (mean)"
          value={formatNumber(forecastMean, 2)}
          unit="mm/day"
          loading={forecast.isLoading}
          delta={
            forecast.data
              ? `${forecast.data.horizon_days}-day horizon`
              : "Awaiting forecast"
          }
          deltaTone="info"
        />
        <MetricTile
          label="Reservoir fill (Mettur)"
          value="n/a"
          unit=""
          delta="Not in cauvery.nc; kitchen-sink shows data completeness instead."
          deltaTone="neutral"
        />
      </div>

      <div className="grid grid-cols-1 gap-column-gap md:grid-cols-2">
        <GlassCard>
          <SectionHeader
            size="compact"
            title="Model skill vs baselines"
            subtitle={
              validation.data
                ? `Real 2023 test window (${validation.data.n_windows} windows).`
                : "Warming validation bundle from the FastAPI service..."
            }
            rightSlot={
              validation.data ? (
                <StatusPill
                  label={`CSI@1mm ${formatNumber(rainCsi, 2)}`}
                  tone={
                    rainCsi !== null && persistenceCsi !== null && rainCsi > persistenceCsi
                      ? "positive"
                      : "warning"
                  }
                  size="sm"
                />
              ) : null
            }
          />
          {validation.isLoading ? <Skeleton variant="block" /> : null}
          {validation.data ? (
            <MiniBarChart
              ariaLabel="Rain RMSE ours vs persistence vs climatology"
              bars={[
                {
                  id: "ours",
                  label: "Ours",
                  value: validation.data.ours.rmse ?? 0,
                  tone: "info",
                },
                {
                  id: "persistence",
                  label: "Persistence",
                  value: validation.data.persistence.rmse ?? 0,
                  tone: "warning",
                },
                {
                  id: "climatology",
                  label: "Climatology",
                  value: validation.data.climatology.rmse ?? 0,
                  tone: "neutral",
                },
              ]}
            />
          ) : null}
        </GlassCard>

        <GlassCard>
          <SectionHeader
            size="compact"
            title="Scenario impact (historical warm/dry)"
            subtitle={
              scenario.data
                ? `${scenario.data.dates.length}-day baseline; SCS-CN inflow.`
                : "Running scenario against 2023-06 window..."
            }
          />
          {scenario.isMutating || !scenario.data ? (
            <Skeleton variant="block" />
          ) : (
            <>
              <MiniBarChart
                ariaLabel="Baseline versus scenario basin inflow"
                bars={scenarioBars}
              />
              <InlineLegend
                className="mt-3"
                items={[
                  {
                    id: "delta-m3",
                    label: "Δ inflow",
                    tone: "warning",
                    value: `${(scenario.data.impacts.hydrology.delta_pct).toFixed(1)}%`,
                  },
                  {
                    id: "rupee",
                    label: "₹ risk",
                    tone: "critical",
                    value: `${scenario.data.impacts.rupee.total_crore.toFixed(1)} cr`,
                  },
                ]}
              />
            </>
          )}
        </GlassCard>
      </div>

      <GlassCard>
        <SectionHeader
          size="compact"
          title="Derived alerts"
          subtitle={
            alerts.data
              ? `Evaluated ${alerts.data.evaluated_date_ist} · thresholds resolved from ${
                  Object.values(alerts.data.thresholds.sources).join(", ")
                }`
              : "Fetching alerts..."
          }
          rightSlot={
            alerts.data ? (
              <StatusPill
                label={`${alerts.data.alerts.length} active`}
                tone={alerts.data.alerts.length ? "warning" : "positive"}
                size="sm"
              />
            ) : null
          }
        />
        {alerts.data && alerts.data.alerts.length > 0 ? (
          <div className="mt-3 space-y-3">
            {alerts.data.alerts.slice(0, 3).map((alert) => (
              <AlertRow
                key={alert.id}
                severity={alert.severity === "critical" ? "critical" : alert.severity === "warning" ? "warning" : "info"}
                timestamp={alert.evaluated_date}
                title={alert.title}
                description={`${alert.description} (threshold ${alert.threshold} ${alert.unit}, source ${alert.threshold_source})`}
              />
            ))}
          </div>
        ) : (
          <div className="mt-3">
            <AlertRow
              severity="positive"
              timestamp={alerts.data?.evaluated_date_ist.slice(0, 10) ?? CATALOG_DATE}
              title="No advisories generated"
              description={
                thresholds.data
                  ? `No basin metric crossed rain ${thresholds.data.thresholds.rain_warning_mm_per_day.toFixed(1)} mm/day or heat ${thresholds.data.thresholds.heat_warning_c}°C thresholds for this day.`
                  : "Thresholds not loaded yet."
              }
            />
          </div>
        )}
      </GlassCard>

      <GlassCard>
        <SectionHeader
          size="compact"
          title="Report templates"
          subtitle="Real catalog returned from /api/v1/reports/templates."
        />
        {reports.isLoading ? <Skeleton variant="block" /> : null}
        {reports.data ? (
          <ul className="mt-3 space-y-2">
            {reports.data.templates.map((template) => (
              <li key={template.id} className="rounded-card border border-cardBorder bg-card-muted p-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-primary" aria-hidden="true" />
                  <p className="text-label text-textPrimary">{template.name}</p>
                </div>
                <p className="mt-1 text-caption text-textSecondary">{template.description}</p>
                <p className="mt-1 font-mono text-caption text-textTertiary">
                  {template.required_fields.join(" · ")}
                </p>
              </li>
            ))}
          </ul>
        ) : null}
      </GlassCard>
    </section>
  )
}
