"use client"

import { useCallback, useEffect, type ReactNode } from "react"
import { AlertTriangle, ChevronRight, Clock3, Download, MessageSquare, Radio, Share2 } from "lucide-react"

import { CardSkeleton, DashboardHeader, EmptyState, ErrorState, GlassCard, StatusPill } from "@/components"
import { useAlerts, useHealth, useSettingsThresholds } from "@/lib/api"
import { transformAlerts } from "@/lib/transforms"

const toneClasses: Record<string, { rail: string; text: string }> = {
  critical: { rail: "bg-critical", text: "text-critical" }, warning: { rail: "bg-warning", text: "text-warning" }, info: { rail: "bg-primary", text: "text-primary" }, neutral: { rail: "bg-textTertiary", text: "text-textSecondary" },
}

const formatIst = (value: string | null | undefined, dateOnly = false) => {
  if (!value) return "Unavailable"
  const date = new Date(dateOnly ? `${value}T00:00:00+05:30` : value)
  if (Number.isNaN(date.getTime())) return "Unavailable"
  return `${new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: dateOnly ? undefined : "short", timeZone: "Asia/Kolkata" }).format(date)}${dateOnly ? "" : " IST"}`
}

const formatNumber = (value: number | null | undefined) => value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : value.toLocaleString("en-IN", { maximumFractionDigits: 1 })

function PanelHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return <div className="flex items-start justify-between gap-3 border-b border-divider pb-2"><div><h2 className="text-title text-textPrimary">{title}</h2>{subtitle ? <p className="mt-1 text-caption text-textSecondary">{subtitle}</p> : null}</div>{right}</div>
}

function UnsupportedPanel({ title, description }: { title: string; description: string }) {
  return <EmptyState title={title} description={description} className="mt-3 p-5" />
}

function AlertFeed({ alerts }: { alerts: ReturnType<typeof transformAlerts>["alerts"] }) {
  return <div className="mt-2 divide-y divide-divider">{alerts.map((item) => { const tone = toneClasses[item.tone] ?? toneClasses.neutral; return <article key={item.id} className="relative py-2.5 pl-3 first:pt-1"><span className={`absolute bottom-2 left-0 top-2 w-1 rounded-pill ${tone.rail}`} /><div className="flex items-center gap-2"><span className={`text-eyebrow uppercase ${tone.text}`}>{item.severityLabel}</span><time className="font-mono text-[9px] text-textTertiary">{formatIst(item.timeIst, item.timeIst.length <= 10)}</time></div><h3 className="mt-1 text-[11px] font-semibold text-textPrimary">{item.title}</h3><p className="mt-0.5 text-[10px] text-textSecondary">{item.description}</p><p className="mt-1 text-[9px] text-textTertiary">Observed {formatNumber(item.observedValue)} {item.unit} · threshold {formatNumber(item.threshold)} {item.unit}</p><p className="mt-1 text-[9px] text-textTertiary">Source: {item.dataSource} · {item.thresholdSource}</p></article> })}</div>
}

export default function AlertsPage() {
  const alerts = useAlerts()
  const settings = useSettingsThresholds()
  const health = useHealth()
  const view = transformAlerts(alerts.data)
  const thresholdConfig = settings.data?.thresholds ?? view.thresholds
  const isLoading = alerts.isLoading || settings.isLoading || health.isLoading
  const error = alerts.error ?? settings.error ?? health.error
  const { mutate: mutateAlerts } = alerts
  const { mutate: mutateSettings } = settings
  const { mutate: mutateHealth } = health
  const refresh = useCallback(() => Promise.all([mutateAlerts(), mutateSettings(), mutateHealth()]), [mutateAlerts, mutateSettings, mutateHealth])

  useEffect(() => {
    const handleFocus = () => { void refresh() }
    window.addEventListener("focus", handleFocus)
    return () => window.removeEventListener("focus", handleFocus)
  }, [refresh])

  const severeAlert = view.alerts.find((item) => item.severity === "critical") ?? view.alerts[0]
  const apiOnline = health.data?.status === "ok"

  return <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto pr-1">
    <DashboardHeader title="Alerts & Incident Command" metadata={`${view.alerts.length} active alert${view.alerts.length === 1 ? "" : "s"} · Evaluated ${formatIst(view.evaluatedDateIst)}`} notificationCount={view.alerts.length} actions={<><button type="button" className="inline-flex min-h-control items-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary"><Download className="h-4 w-4 text-positive" />Export</button><button type="button" className="inline-flex min-h-control items-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary"><Share2 className="h-4 w-4 text-positive" />Share</button><StatusPill label={apiOnline ? "API online" : "API unavailable"} tone={apiOnline ? "positive" : "neutral"} size="md" /></>} />
    {isLoading ? <div className="grid grid-cols-1 gap-3 md:grid-cols-2"><CardSkeleton label="Loading alerts" /><CardSkeleton label="Loading alert thresholds" /></div> : null}
    {error ? <ErrorState title="Alert data unavailable" description={String((error as Error).message ?? "The alert service could not be reached.")} onRetry={refresh} /> : null}
    {!isLoading && !error && alerts.data && view.alerts.length === 0 ? <EmptyState title="No active alerts" description={`No rainfall or heat thresholds are exceeded for ${formatIst(view.evaluatedDateIst, (view.evaluatedDateIst ?? "").length <= 10)}.`} /> : null}
    {!isLoading && !error && alerts.data && view.alerts.length > 0 ? <>
      <section role="alert" className="grid shrink-0 grid-cols-[1.25fr_1.25fr_1.25fr_.8fr] gap-4 rounded-card border border-critical border-l-4 bg-card px-5 py-4 shadow-elevation-card"><div className="min-w-0"><p className="text-eyebrow uppercase text-critical">{severeAlert?.severityLabel ?? "Alert"}</p><h2 className="mt-1 text-[17px] font-semibold text-textPrimary">{severeAlert?.title ?? "Active weather alert"}</h2><p className="mt-1 text-[11px] text-textSecondary">{severeAlert?.description ?? "Unavailable"}</p><p className="mt-2 text-[10px] text-textSecondary">Evaluated {formatIst(severeAlert?.evaluatedDate, severeAlert?.evaluatedDate.length <= 10)}</p></div><div className="border-l border-divider pl-4"><p className="text-[21px] font-semibold tabular-nums text-textPrimary">{formatNumber(severeAlert?.observedValue)} {severeAlert?.unit ?? ""}</p><p className="text-caption text-textSecondary">Observed value</p><p className="mt-3 text-[21px] font-semibold tabular-nums text-textPrimary">{formatNumber(severeAlert?.threshold)} {severeAlert?.unit ?? ""}</p><p className="text-caption text-textSecondary">Configured threshold</p></div><div className="border-l border-divider pl-4"><p className="text-[21px] font-semibold tabular-nums text-textPrimary">{view.alerts.length}</p><p className="text-caption text-textSecondary">Active alerts</p><p className="mt-3 text-caption text-textSecondary">Service: <strong className={apiOnline ? "text-positive" : "text-warning"}>{health.data?.status ?? "Unavailable"}</strong></p></div><div className="flex flex-col justify-center gap-2"><button type="button" className="inline-flex min-h-control items-center justify-center gap-2 rounded-button bg-critical px-3 text-label font-semibold text-textInverse"><AlertTriangle className="h-4 w-4" />Review alert</button><button type="button" className="inline-flex min-h-control items-center justify-center gap-1 rounded-button border border-cardBorder px-3 text-label text-textSecondary">Alert details <ChevronRight className="h-3.5 w-3.5" /></button></div></section>
      <div className="flex shrink-0 items-center gap-5 px-1 text-[10px] text-textSecondary"><span className="inline-flex items-center gap-1"><Clock3 className="h-3.5 w-3.5" />Evaluated <strong className="text-textPrimary">{formatIst(view.evaluatedDateIst)}</strong></span><span><strong className="text-textPrimary">{view.alerts.length}</strong> alert{view.alerts.length === 1 ? "" : "s"} returned</span><span>Model <strong className="text-textPrimary">{health.data?.model_loaded ? "loaded" : "Unavailable"}</strong></span></div>
      <div className="grid min-h-0 shrink-0 grid-cols-[1.08fr_1fr] gap-3"><div className="flex min-h-0 flex-col gap-3"><GlassCard padding="default"><PanelHeader title="Incident zone" subtitle="Spatial incident data is not exposed by the alerts API" right={<StatusPill label="Unavailable" tone="neutral" size="sm" />} /><UnsupportedPanel title="Incident map unavailable" description="Rainfall intensity grids and district boundaries are not returned by this endpoint." /></GlassCard><GlassCard padding="default"><PanelHeader title="Incident timeline" subtitle="Alert event history" /><UnsupportedPanel title="Incident timeline unavailable" description="Incident lifecycle events are not exposed by the current API contract." /></GlassCard></div><div className="flex min-h-0 flex-col gap-3"><GlassCard padding="default"><PanelHeader title="Live alert feed" subtitle={`Evaluated ${formatIst(view.evaluatedDateIst)}`} right={<span className="inline-flex items-center gap-1 text-[10px] font-semibold text-positive"><Radio className="h-3 w-3" />Live</span>} /><AlertFeed alerts={view.alerts} /></GlassCard><GlassCard padding="default"><PanelHeader title="Response coordination" subtitle="Agency data" /><UnsupportedPanel title="Coordination unavailable" description={view.advisories.unavailableLabel ?? "Response coordination is unavailable from the API."} /></GlassCard></div></div>
      <div className="grid shrink-0 grid-cols-[1.12fr_.88fr] gap-3"><GlassCard padding="default"><PanelHeader title="Outgoing broadcast" subtitle="Broadcast workflow" /><UnsupportedPanel title="Broadcast unavailable" description="Broadcast drafting and delivery are not exposed by the current API." /></GlassCard><GlassCard padding="default"><PanelHeader title="Threshold configuration" right={<MessageSquare className="h-4 w-4 text-textTertiary" />} />{thresholdConfig ? <div className="mt-2 space-y-1 text-[10px] text-textSecondary"><div className="flex justify-between border-b border-divider py-1.5"><span>Rain warning</span><strong className="text-textPrimary">{thresholdConfig.rain_warning_mm_per_day} mm/day</strong></div><div className="flex justify-between border-b border-divider py-1.5"><span>Rain critical</span><strong className="text-textPrimary">{thresholdConfig.rain_critical_mm_per_day} mm/day</strong></div><div className="flex justify-between border-b border-divider py-1.5"><span>Heat warning</span><strong className="text-textPrimary">{thresholdConfig.heat_warning_c} °C</strong></div><div className="flex justify-between py-1.5"><span>Heat critical</span><strong className="text-textPrimary">{thresholdConfig.heat_critical_c} °C</strong></div></div> : <UnsupportedPanel title="Thresholds unavailable" description={view.thresholdAvailability.unavailableLabel ?? "Threshold configuration is unavailable."} />}</GlassCard></div>
      <GlassCard padding="default" className="shrink-0"><PanelHeader title="Decision log" subtitle="Authority actions" right={<StatusPill label="Unavailable" tone="neutral" size="sm" />} /><UnsupportedPanel title="Decision log unavailable" description="Authority actions and audit history are not exposed by the alerts API." /></GlassCard>
    </> : null}
  </div>
}
