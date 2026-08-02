"use client"

import type { ReactNode } from "react"
import { AlertTriangle, Check, Download, Info, Share2 } from "lucide-react"

import {
  CardSkeleton,
  DashboardHeader,
  EmptyState,
  ErrorState,
  GlassCard,
  RichCycleStrip,
  StatusPill,
} from "@/components"
import { useDates, useValidationDateMetrics, useValidationMetrics } from "@/lib/api"
import { formatValidationMetric, transformValidation } from "@/lib/transforms"

const validationPipeline = [
  { label: "INGEST", status: "completed" as const },
  { label: "REGRID", status: "completed" as const },
  { label: "ASSIMILATE", status: "completed" as const },
  { label: "FORECAST", status: "completed" as const },
  { label: "IMPACT", status: "completed" as const },
  { label: "VALIDATE", status: "active" as const },
] as const

const formatNumber = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "Unavailable"
    : value.toLocaleString("en-IN", { maximumFractionDigits: digits })

const formatIst = (value: string | null | undefined) => {
  if (!value) return "Unavailable"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return "Unavailable"
  return `${new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Kolkata",
  }).format(date)} IST`
}

function PanelHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return <div className="flex items-start justify-between gap-3 border-b border-divider pb-3"><div><h2 className="text-title text-textPrimary">{title}</h2>{subtitle ? <p className="mt-1 text-caption text-textSecondary">{subtitle}</p> : null}</div>{right}</div>
}

function UnsupportedPanel({ title, description }: { title: string; description: string }) {
  return <EmptyState title={title} description={description} className="mt-3 p-5" />
}

function ThresholdChart({ ours, unit }: { ours: Record<string, number | null | undefined>; unit: string }) {
  const rows = [
    { threshold: "1", pod: ours["pod@1mm"], far: ours["far@1mm"], csi: ours["csi@1mm"] },
    { threshold: "10", pod: ours["pod@10mm"], far: ours["far@10mm"], csi: ours["csi@10mm"] },
  ]
  const colors = ["#0061FE", "#F57602", "#07945B"]
  const names = ["POD (detection)", "FAR (false alarm)", "CSI (overall skill)"]
  return <><div className="mt-3 flex items-center gap-4 text-[10px] text-textSecondary">{names.map((name, index) => <span key={name} className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-sm" style={{ backgroundColor: colors[index] }} />{name}</span>)}</div><div className="mt-3 grid grid-cols-[44px_1fr] gap-2"><div className="flex h-[172px] flex-col justify-between text-[9px] text-textTertiary"><span>1.0</span><span>0.75</span><span>0.5</span><span>0.25</span><span>0.0</span></div><div className="relative h-[172px] border-b border-l border-divider bg-[linear-gradient(to_bottom,transparent_24%,#eef0f2_25%,transparent_26%,transparent_49%,#eef0f2_50%,transparent_51%,transparent_74%,#eef0f2_75%,transparent_76%)]"><div className="flex h-full items-end justify-around px-3">{rows.map((row) => <div key={row.threshold} className="flex h-full w-16 flex-col justify-end"><div className="flex h-[145px] items-end justify-center gap-1">{[row.pod, row.far, row.csi].map((value, index) => <div key={index} className="w-3 rounded-t-sm" style={{ height: value == null ? 0 : `${Math.max(0, Math.min(1, value)) * 145}px`, backgroundColor: colors[index] }} title={`${names[index]} ${formatValidationMetric(value ?? null, "")}`} />)}</div><span className="mt-2 text-center text-[10px] font-medium text-textSecondary">&gt;{row.threshold} {unit}</span></div>)}</div></div></div><p className="mt-3 border-t border-divider pt-2 text-[10px] leading-4 text-textSecondary"><span className="font-semibold text-warning">API coverage:</span> Threshold metrics are available for the 1 and 10 mm thresholds. Other threshold and scatter data are unavailable.</p></>
}

export default function ValidationPage() {
  const dates = useDates()
  const selectedDate = dates.data?.dates.at(-1) ?? null
  const metrics = useValidationMetrics()
  const dateMetrics = useValidationDateMetrics(selectedDate)
  const view = transformValidation(metrics.data, dateMetrics.data)
  const isLoading = dates.isLoading || metrics.isLoading || dateMetrics.isLoading
  const error = dates.error ?? metrics.error ?? dateMetrics.error
  const retry = () => Promise.all([dates.mutate(), metrics.mutate(), dateMetrics.mutate()])
  const source = view.dateMetrics ?? view.metrics
  const aggregateMetrics = view.metrics
  const unit = source?.unit ?? "mm/day"
  const model = source?.ours
  const rows = view.baselineComparison
  const modelRmse = model?.rmse ?? null
  const climatologyRmse = source?.climatology.rmse ?? null
  const improvement = modelRmse !== null && climatologyRmse !== null && climatologyRmse !== 0 ? ((climatologyRmse - modelRmse) / climatologyRmse) * 100 : null

  return <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto pr-1">
    <DashboardHeader title="Validation" metadata={`${aggregateMetrics?.split ?? "Validation metrics"} · ${aggregateMetrics?.years?.join(", ") ?? "Date unavailable"} · Evaluated ${formatIst(view.dateMetrics?.date_ist)}`} notificationCount={0} actions={<><button type="button" className="inline-flex min-h-control items-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary"><Download className="h-4 w-4 text-positive" />Export</button><button type="button" className="inline-flex min-h-control items-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary"><Share2 className="h-4 w-4 text-positive" />Share</button><StatusPill label={view.audit.available ? "Available" : "Unavailable"} tone={view.audit.available ? "positive" : "neutral"} size="md" /></>} />
    {isLoading ? <div className="grid grid-cols-1 gap-3 md:grid-cols-2"><CardSkeleton label="Loading validation metrics" /><CardSkeleton label="Loading date-level validation metrics" /></div> : null}
    {error ? <ErrorState title="Validation data unavailable" description={String((error as Error).message ?? "The validation service could not be reached.")} onRetry={retry} /> : null}
    {!isLoading && !error && !source ? <EmptyState title="No validation metrics" description="The API returned no validation metrics for the available dates." /> : null}
    {!isLoading && !error && source ? <>
      <GlassCard padding="none" className="grid shrink-0 grid-cols-4 divide-x divide-divider">{[{ label: "Model", value: "MausamSetu", detail: source.variable }, { label: "Holdout", value: aggregateMetrics?.split ?? "Date-level", detail: aggregateMetrics ? `${aggregateMetrics.n_windows} windows` : `${view.dateMetrics?.n_forecast_days ?? "Unavailable"} forecast days` }, { label: "Date-level", value: formatIst(view.dateMetrics?.date_ist), detail: `${view.dateMetrics?.n_forecast_days ?? "Unavailable"} forecast days` }, { label: "Unit", value: unit, detail: `Lead day ${aggregateMetrics?.lead_day ?? "all"}` }].map((item) => <div key={item.label} className="px-5 py-3"><p className="text-eyebrow uppercase text-textTertiary">{item.label}</p><p className="mt-1 text-[18px] font-semibold text-textPrimary">{item.value}</p><p className="mt-1 text-caption text-textSecondary">{item.detail}</p></div>)}</GlassCard>
      <div className="grid shrink-0 grid-cols-2 gap-3"><GlassCard padding="default"><PanelHeader title="Observed vs predicted" subtitle="Daily rainfall · scatter payload not exposed by API" /><UnsupportedPanel title="Scatter data unavailable" description={view.scatter.unavailableLabel ?? "Observed and predicted point data is unavailable."} /></GlassCard><GlassCard padding="default"><PanelHeader title="Detection skill · by rainfall threshold" subtitle={`POD · FAR · CSI · ${unit}`} /><ThresholdChart ours={model ?? {}} unit={unit} /></GlassCard></div>
      <div className="grid shrink-0 grid-cols-[1.1fr_.9fr_1fr] gap-3"><GlassCard padding="default"><PanelHeader title="Spatial error · Cauvery Basin" subtitle="Where the model struggles" /><UnsupportedPanel title="Spatial error unavailable" description={view.spatialError.unavailableLabel ?? "Spatial validation data is unavailable from the API."} /></GlassCard><GlassCard padding="default"><PanelHeader title="Skill vs baselines" subtitle={`Date-level RMSE (${unit}) · lower is better`} /><div className="mt-4 space-y-3">{rows.map((row) => <div key={row.model}><div className="flex justify-between text-caption"><span className="font-medium">{row.model}</span><span className={row.status === "selected" ? "font-semibold text-positive" : "text-textSecondary"}>{row.displayValue}</span></div><div className="mt-1 h-2 rounded-pill bg-card-muted"><div className={`h-2 rounded-pill ${row.status === "selected" ? "bg-positive" : "bg-textTertiary"}`} style={{ width: row.value == null ? "0%" : `${Math.min(100, (row.value / Math.max(modelRmse ?? 1, 1)) * 70)}%` }} /></div></div>)}</div><div className="mt-4 flex items-end gap-3"><span className="text-[38px] font-medium leading-none text-positive">{improvement === null ? "Unavailable" : `+${formatNumber(improvement, 0)}%`}</span><div className="text-caption text-textSecondary">skill improvement<br /><span className="font-semibold text-positive">against climatology</span></div></div></GlassCard><GlassCard padding="default"><PanelHeader title="Drought-year stress test" subtitle="Model tested on analogue years" /><UnsupportedPanel title="Drought stress data unavailable" description="Drought-year series are not exposed by the validation API." /></GlassCard></div>
      <GlassCard padding="none" className="grid shrink-0 grid-cols-3 divide-x divide-divider"><div className="p-4"><div className="flex items-center gap-2 text-caption font-semibold"><AlertTriangle className="h-4 w-4 text-warning" />Known limitation</div><p className="mt-2 text-[10px] leading-4 text-textSecondary">Extreme-event and spatial diagnostics are not included in the current API response.</p><p className="mt-1 text-[10px] leading-4 text-textTertiary">The supported metric bundles above remain sourced from the validation service.</p></div><div className="p-4"><div className="flex items-center gap-2 text-caption font-semibold"><Check className="h-4 w-4 text-positive" />Validation data window</div><p className="mt-2 text-[10px] leading-4 text-textSecondary">{aggregateMetrics?.years.length ? aggregateMetrics.years.join(", ") : "Unavailable"} · {aggregateMetrics?.n_windows ?? view.dateMetrics?.n_forecast_days ?? "Unavailable"} windows.</p><p className="mt-1 text-[10px] leading-4 text-textTertiary">Evaluated {formatIst(view.dateMetrics?.date_ist)}</p></div><div className="p-4"><div className="flex items-center gap-2 text-caption font-semibold"><Info className="h-4 w-4 text-primary" />What we don&apos;t yet expose</div><p className="mt-2 text-[10px] leading-4 text-textSecondary">Scatter, spatial error, and drought-year diagnostics.</p><p className="mt-1 text-[10px] leading-4 text-textTertiary">Unavailable from the current API contract.</p></div></GlassCard>
      <RichCycleStrip cycleLabel="VALIDATION" stages={validationPipeline} className="shrink-0" />
    </> : null}
  </div>
}
