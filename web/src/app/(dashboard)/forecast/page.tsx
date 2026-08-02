"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  Bell,
  Check,
  ChevronDown,
  Download,
  Info,
  Layers3,
  Link2,
  MoreHorizontal,
  Share2,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import {
  DashboardHeader,
  EmptyState,
  ErrorState,
  GlassCard,
  SectionHeader,
  Skeleton,
  StaticBasinMap,
  TabPill,
} from "@/components";
import { useCurrentState, useForecast, useValidationMetrics } from "@/lib/api";
import { aggregateForecastGrid, transformForecast } from "@/lib/transforms";

 type Metric = "rain" | "tmax" | "tmin";

const metricLabels: Record<Metric, string> = { rain: "Rainfall", tmax: "Max temp", tmin: "Min temp" };
const metricUnits: Record<Metric, string> = { rain: "mm/day", tmax: "°C", tmin: "°C" };

type PlotPoint = {
  date: string;
  observed: number | null;
  p10: number;
  p50: number;
  p90: number;
  climatology: number | null;
};

function formatDate(date: string | null | undefined) {
  if (!date) return "Unavailable";
  const parsed = new Date(`${date.slice(0, 10)}T00:00:00+05:30`);
  if (Number.isNaN(parsed.getTime())) return date;
  return new Intl.DateTimeFormat("en-IN", { month: "short", day: "numeric", timeZone: "Asia/Kolkata" }).format(parsed);
}

function formatIst(value: string | null | undefined) {
  if (!value) return "IST time unavailable";
  const parsed = new Date(value.includes("T") ? value : `${value}T00:00:00+05:30`);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
    timeZoneName: "short",
  }).format(parsed);
}

function displayNumber(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : value.toFixed(digits);
}

function ForecastPlot({ metric, points }: { metric: Metric; points: PlotPoint[] }) {
  const width = 920;
  const height = 220;
  const left = 46;
  const right = 18;
  const top = 18;
  const bottom = 38;
  const max = metric === "rain" ? 60 : metric === "tmax" ? 38 : 30;
  const x = (index: number) => left + (index / Math.max(1, points.length - 1)) * (width - left - right);
  const y = (value: number) => top + (height - top - bottom) - (value / max) * (height - top - bottom);
  const path = (key: "p50" | "climatology") => points
    .filter((point) => point[key] !== null)
    .map((point) => {
      const index = points.indexOf(point);
      return `${index ? "L" : "M"} ${x(index).toFixed(1)} ${y(point[key] as number).toFixed(1)}`;
    })
    .join(" ");
  const upper = points.map((point, index) => `${x(index).toFixed(1)},${y(point.p90).toFixed(1)}`).join(" ");
  const lower = [...points].reverse().map((point, reverseIndex) => {
    const index = points.length - reverseIndex - 1;
    return `${x(index).toFixed(1)},${y(point.p10).toFixed(1)}`;
  }).join(" ");
  const peakIndex = points.reduce((best, point, index) => point.p50 > points[best].p50 ? index : best, 0);
  const forecastStart = metric === "rain" ? 0 : 0;
  const ticks = [0, 15, 30, 45, 60].filter((tick) => tick <= max);

  return (
    <div className="min-h-0 flex-1 px-2 pb-1" role="img" aria-label={`${metricLabels[metric]} forecast with p10 to p90 uncertainty band`}>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" preserveAspectRatio="none">
        <title>{metricLabels[metric]} forecast with p10 to p90 uncertainty</title>
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="stroke-divider" />
            <text x={left - 9} y={y(tick) + 4} textAnchor="end" className="fill-textTertiary text-[10px]">{tick}</text>
          </g>
        ))}
        <text x={left - 30} y={12} className="fill-textSecondary text-[10px]">{metricUnits[metric]}</text>
        <polygon points={`${upper} ${lower}`} className="fill-primary-soft" opacity="0.9" />
        {metric === "rain" ? <>
          <line x1={left} x2={width - right} y1={y(40)} y2={y(40)} className="stroke-warning" strokeDasharray="6 5" />
          <text x={width - right} y={y(40) - 6} textAnchor="end" className="fill-warning text-[10px]">Extreme threshold (40 mm)</text>
        </> : null}
        {points.some((point) => point.climatology !== null) ? <path d={path("climatology")} fill="none" className="stroke-textTertiary" strokeDasharray="5 5" strokeWidth="1.5" /> : null}
        <path d={path("p50")} fill="none" className="stroke-primary" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        {metric === "rain" && forecastStart > 0 ? <line x1={x(forecastStart - 0.5)} x2={x(forecastStart - 0.5)} y1={top} y2={height - bottom} className="stroke-divider" strokeDasharray="4 4" /> : null}
        {points.map((point, index) => (
          <g key={point.date}>
            <circle cx={x(index)} cy={y(point.observed ?? point.p50)} r={point.observed !== null ? 4 : 3.2} className="fill-primary stroke-card-solid" strokeWidth="2" />
            <text x={x(index)} y={height - 13} textAnchor="middle" className="fill-textSecondary text-[10px]">{formatDate(point.date)}</text>
          </g>
        ))}
        <g transform={`translate(${Math.min(width - 145, Math.max(65, x(peakIndex) - 52))} ${Math.max(2, y(points[peakIndex].p90) - 27)})`}>
          <rect width="138" height="22" rx="6" className="fill-card-solid stroke-cardBorder" />
          <text x="69" y="15" textAnchor="middle" className="fill-textPrimary text-[10px]">Peak {displayNumber(points[peakIndex].p50)} {metricUnits[metric]} · {formatDate(points[peakIndex].date)}</text>
        </g>
        {points.some((point) => point.climatology !== null) ? <text x={left + 6} y={y(points.find((point) => point.climatology !== null)?.climatology ?? 0) - 6} className="fill-textTertiary text-[10px]">Climatology</text> : null}
      </svg>
    </div>
  );
}

function CardTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return <div><h2 className="text-title text-textPrimary">{title}</h2>{subtitle ? <p className="mt-0.5 text-caption text-textSecondary">{subtitle}</p> : null}</div>;
}

export default function ForecastPage() {
  const [metric, setMetric] = useState<Metric>("rain");
  const [selectedFrame, setSelectedFrame] = useState("day-1");
  const current = useCurrentState();
  const forecastDate = current.data?.date_ist?.slice(0, 10) ?? null;
  const forecast = useForecast(forecastDate, 7);
  const validation = useValidationMetrics();
  const view = transformForecast(forecast.data);
  const retry = () => {
    void Promise.all([current.mutate(), forecast.mutate(), validation.mutate()]);
  };
  const points = useMemo<PlotPoint[]>(() => {
    if (!forecast.data) return [];
    if (metric === "rain") {
      return view.basinSeries
        .filter((point) => point.available)
        .map((point) => {
          const p50 = point.p50 ?? point.p10 ?? point.p90 ?? 0;
          return { date: point.date, observed: point.observed, p10: point.p10 ?? p50, p50, p90: point.p90 ?? p50, climatology: point.climatology };
        });
    }
    const block = forecast.data[metric];
    const p10 = aggregateForecastGrid(block.p10);
    const p50 = aggregateForecastGrid(block.p50);
    const p90 = aggregateForecastGrid(block.p90);
    return forecast.data.dates.flatMap((date, index) => {
      const middle = p50[index] ?? p10[index] ?? p90[index] ?? null;
      if (middle === null) return [];
      return [{ date, observed: null, p10: p10[index] ?? middle, p50: middle, p90: p90[index] ?? middle, climatology: null }];
    });
  }, [forecast.data, metric, view.basinSeries]);
  const peak = view.summary.peak_intensity_mm;
  const peakDate = view.summary.peak_date;
  const skillItems = validation.data ? [
    { label: "MAE", value: validation.data.ours.mae ?? "Unavailable", unit: validation.data.unit },
    { label: "RMSE", value: validation.data.ours.rmse ?? "Unavailable", unit: validation.data.unit },
    { label: "Bias", value: validation.data.ours.bias ?? "Unavailable", unit: validation.data.unit },
    { label: "CSI @ 1mm", value: validation.data.ours["csi@1mm"] ?? "Unavailable", unit: "score" },
    { label: "POD @ 1mm", value: validation.data.ours["pod@1mm"] ?? "Unavailable", unit: "score" },
    { label: "FAR @ 1mm", value: validation.data.ours["far@1mm"] ?? "Unavailable", unit: "score" },
  ] : [];
  const frameSelection = view.spatialFrames.find((frame) => frame.id === selectedFrame) ?? view.spatialFrames[0];

  if (current.error || forecast.error) {
    return <ErrorState title="Forecast unavailable" description="The latest basin state or forecast could not be loaded." onRetry={retry} />;
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 overflow-auto pr-1">
      <DashboardHeader title="Forecast" metadata={`Cauvery Basin · ${view.summary.model ?? "Live forecast"} · Updated ${formatIst(view.summary.updated_ist)}`} />
      <div className="flex shrink-0 items-center justify-end gap-2 -mt-1"><button type="button" className="inline-flex h-8 items-center gap-2 rounded-control border border-cardBorder bg-card px-3 text-caption text-textSecondary shadow-elevation-card"><Download className="h-3.5 w-3.5" />Export</button><button type="button" className="inline-flex h-8 items-center gap-2 rounded-control border border-cardBorder bg-card px-3 text-caption text-textSecondary shadow-elevation-card"><Share2 className="h-3.5 w-3.5" />Share</button><button type="button" aria-label="Notifications" className="relative flex h-8 w-8 items-center justify-center rounded-control border border-cardBorder bg-card text-textSecondary"><Bell className="h-4 w-4" /></button><button type="button" className="flex h-8 items-center gap-2 rounded-control border border-cardBorder bg-card px-2 text-caption font-semibold"><span className="flex h-6 w-6 items-center justify-center rounded-pill bg-primary text-[10px] text-white">BC</span><ChevronDown className="h-3.5 w-3.5 text-textTertiary" /></button></div>

      <section className="grid shrink-0 grid-cols-4 divide-x divide-divider rounded-panel border border-cardBorder bg-card px-2 py-3 shadow-elevation-card" aria-label="Forecast summary">
        {[{ label: "Forecast horizon", value: view.summary.horizon_days ? `${view.summary.horizon_days} days` : "Unavailable", detail: view.summary.available ? "live guidance" : "Awaiting forecast" }, { label: "Ensemble members", value: view.summary.ensemble_members ?? "Unavailable", detail: view.ensembleUnavailableLabel ?? "API value" }, { label: "Peak intensity", value: peak === null ? "Unavailable" : `${displayNumber(peak)} ${forecast.data?.rain.unit ?? "mm/day"}`, detail: peakDate ? formatDate(peakDate) : "Unavailable" }, { label: "Confidence", value: view.summary.confidence ?? "Unavailable", detail: view.summary.confidence ? "validated forecast" : "Not provided by API" }].map((item) => <div key={item.label} className="px-4"><p className="text-caption text-textSecondary">{item.label}</p><p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-textPrimary">{item.value}</p><p className="text-caption text-primary">{item.detail}</p></div>)}
      </section>

      <GlassCard padding="none" className="flex min-h-[300px] shrink-0 flex-col overflow-hidden">
        <div className="flex min-h-[58px] items-center justify-between gap-4 border-b border-divider px-5"><CardTitle title="Basin-mean forecast" subtitle="Cauvery Basin · daily aggregates · p10–p90 uncertainty" /><div className="flex items-center gap-2"><div className="flex items-center rounded-control border border-cardBorder p-0.5" role="tablist" aria-label="Forecast metric">{(Object.keys(metricLabels) as Metric[]).map((key) => <TabPill key={key} active={metric === key} size="sm" onClick={() => setMetric(key)}>{metricLabels[key]}</TabPill>)}</div><span className="hidden h-8 items-center gap-1 rounded-control border border-cardBorder px-2 text-caption text-textSecondary sm:flex">{view.summary.model ?? "Model unavailable"}<ChevronDown className="h-3.5 w-3.5" /></span><button type="button" aria-label="Compare with persistence" className="flex h-8 items-center gap-1 rounded-control border border-cardBorder px-2 text-caption text-textSecondary"><Link2 className="h-3.5 w-3.5" />vs Persistence</button></div></div>
        {forecast.isLoading || current.isLoading ? <div className="flex flex-1 items-center p-5"><Skeleton variant="block" className="h-full" /></div> : points.length ? <ForecastPlot metric={metric} points={points} /> : <EmptyState title={`${metricLabels[metric]} unavailable`} description="No p10–p90 guidance is available for this metric." className="m-5 flex-1 justify-center" />}
        <div className="flex min-h-[30px] items-center gap-5 border-t border-divider px-5 text-caption text-textSecondary"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-pill bg-primary" />Observed</span><span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4 bg-primary" />p50 forecast</span><span className="inline-flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm bg-primary-soft" />p10–p90 band</span><span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4 border-t border-dashed border-textTertiary" />Climatology</span></div>
      </GlassCard>

      <section className="grid min-h-[205px] shrink-0 grid-cols-1 gap-2 xl:grid-cols-[1.1fr_1fr]"><GlassCard padding="none" className="overflow-hidden"><div className="flex items-center justify-between px-5 py-3"><CardTitle title="Daily forecast" subtitle="spatial · click any frame to expand" /><Layers3 className="h-4 w-4 text-textTertiary" /></div>{forecast.isLoading ? <div className="p-4"><Skeleton variant="block" /></div> : view.spatialFrames.length ? <div className="grid grid-cols-7 gap-1 px-4 pb-3">{view.spatialFrames.map((frame) => <button key={frame.id} type="button" onClick={() => setSelectedFrame(frame.id)} aria-label={`Open ${formatDate(frame.date)} forecast, ${frame.basin_mean_rainfall_mm === null ? "unavailable" : `${displayNumber(frame.basin_mean_rainfall_mm)} millimetres`}`} className={`rounded-control p-1 text-center transition-colors focus-visible:ring-2 focus-visible:ring-primary ${frameSelection?.id === frame.id ? "bg-primary-soft ring-1 ring-primary" : "hover:bg-card-muted"}`}><StaticBasinMap values={frame.values_mm_per_day} compact showControls={false} className="aspect-square w-full rounded-sm" /><span className="mt-1 block text-[10px] font-medium text-textSecondary">{formatDate(frame.date)}</span><span className="block font-mono text-[11px] font-semibold text-textPrimary">{frame.basin_mean_rainfall_mm === null ? "Unavailable" : `${displayNumber(frame.basin_mean_rainfall_mm)} mm`}</span></button>)}</div> : <EmptyState title="Spatial forecast unavailable" description="No rainfall frames were returned by the API." className="m-4" />}</GlassCard><GlassCard padding="none" className="flex min-h-0 flex-col overflow-hidden"><div className="flex items-center justify-between px-5 py-3"><CardTitle title="Ensemble spread" subtitle="Members · mm" /><Info className="h-4 w-4 text-textTertiary" /></div><div className="min-h-0 flex-1 px-3"><EmptyState title="Ensemble unavailable" description={view.ensembleUnavailableLabel ?? "Ensemble member data is not available from the API."} className="h-full justify-center border-0 bg-transparent p-4" /></div><div className="grid grid-cols-3 divide-x divide-divider border-t border-divider py-2 text-center"><div><p className="text-caption text-textSecondary">Median</p><p className="font-mono text-label">{peak === null ? "Unavailable" : `${displayNumber(peak)} mm`}</p></div><div><p className="text-caption text-textSecondary">Spread</p><p className="font-mono text-label">Unavailable</p></div><div><p className="text-caption text-textSecondary">Coherence</p><p className="font-mono text-label">Unavailable</p></div></div></GlassCard></section>

      <section className="grid shrink-0 grid-cols-1 gap-2 lg:grid-cols-3"><GlassCard className="min-h-[175px] overflow-hidden"><SectionHeader size="compact" title="Monsoon pulse" rightSlot={<span className="text-caption text-textSecondary">Unavailable</span>} /><EmptyState title="Monsoon pulse unavailable" description="This derived indicator is not provided by the API." className="border-0 bg-transparent p-3" /></GlassCard><GlassCard className="min-h-[175px] overflow-hidden"><SectionHeader size="compact" title="Forecast attribution" rightSlot={<span className="text-caption text-textSecondary">Unavailable</span>} /><EmptyState title="Attribution unavailable" description="Integrated Gradients data is not provided by the API." className="border-0 bg-transparent p-3" /></GlassCard><GlassCard className="min-h-[175px] overflow-hidden"><SectionHeader size="compact" title="Model skill" rightSlot={<span className="text-caption text-textSecondary">this forecast</span>} />{validation.isLoading ? <Skeleton variant="block" /> : validation.error ? <ErrorState title="Model skill unavailable" description="Validation metrics could not be loaded." onRetry={() => void validation.mutate()} /> : skillItems.length ? <div className="grid grid-cols-3 gap-x-3 gap-y-2">{skillItems.map((item) => <div key={item.label}><p className="text-caption text-textSecondary">{item.label}</p><p className="mt-0.5 font-mono text-label tabular-nums">{typeof item.value === "number" ? displayNumber(item.value, 2) : item.value} <span className="font-sans text-[10px] text-textTertiary">{item.unit}</span></p></div>)}</div> : <EmptyState title="Validation unavailable" description="No model skill metrics were returned." className="border-0 bg-transparent p-3" />}{validation.data ? <div className="mt-3 flex items-center gap-1.5 border-t border-divider pt-2 text-caption text-positive"><TrendingUp className="h-3.5 w-3.5" />Live validation metrics</div> : null}</GlassCard></section>

      <div className="flex min-h-[38px] shrink-0 items-center justify-between rounded-card border border-cardBorder bg-card px-4 text-caption shadow-elevation-card"><div className="flex items-center gap-4"><span className="font-mono font-semibold text-textPrimary">CYCLE</span>{["INGEST", "REGRID", "ASSIMILATE", "FORECAST", "IMPACT", "RENDER"].map((stage, index) => <span key={stage} className={`hidden items-center gap-1 sm:inline-flex ${index < 3 ? "text-positive" : index === 3 ? "font-semibold text-primary" : "text-textTertiary"}`}>{index < 3 ? <Check className="h-3.5 w-3.5" /> : index === 3 ? <Activity className="h-3.5 w-3.5" /> : <Sparkles className="h-3 w-3" />}{stage}</span>)}</div><MoreHorizontal className="h-4 w-4 text-textTertiary" /></div>
    </div>
  );
}
