"use client";

import {
  CloudRain,
  Droplets,
  Info,
  Layers3,
  Sprout,
  Sun,
  Target,
  TrendingUp,
  Waves,
} from "lucide-react";

import {
  DashboardHeader,
  EmptyState,
  ErrorState,
  ForecastRibbonChart,
  GlassCard,
  MapPlaybackBar,
  MetricMatrix,
  OverviewMetricCard,
  RichCycleStrip,
  SectionHeader,
  Skeleton,
  StaticBasinMap,
  StatusList,
  TabPill,
} from "@/components";
import { useAlerts, useCurrentState, useForecast, useGrid, useHealth, useValidationMetrics } from "@/lib/api";
import { transformAlerts, transformOverview } from "@/lib/transforms";

const metricIcons = {
  rainfall: <CloudRain className="h-5 w-5" aria-hidden="true" />,
  tmax: <Sun className="h-5 w-5" aria-hidden="true" />,
  reservoir: <Droplets className="h-5 w-5" aria-hidden="true" />,
  peak: <CloudRain className="h-5 w-5" aria-hidden="true" />,
  skill: <Target className="h-5 w-5" aria-hidden="true" />,
} as const;

const sectorIcons = {
  "Water security": <Droplets className="h-4 w-4" aria-hidden="true" />,
  Agriculture: <Sprout className="h-4 w-4" aria-hidden="true" />,
  "Heat stress": <Sun className="h-4 w-4" aria-hidden="true" />,
  "Flood risk": <Waves className="h-4 w-4" aria-hidden="true" />,
} as const;

function formatIst(value: string | null | undefined) {
  if (!value) return "IST time unavailable";
  const date = new Date(value.includes("T") ? value : `${value}T00:00:00+05:30`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
    timeZoneName: "short",
  }).format(date);
}

function formatDate(value: string | null | undefined) {
  if (!value) return "Unavailable";
  return new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", timeZone: "Asia/Kolkata" }).format(new Date(`${value.slice(0, 10)}T00:00:00+05:30`));
}

export default function OverviewPage() {
  const health = useHealth();
  const current = useCurrentState();
  const forecastDate = current.data?.date_ist?.slice(0, 10) ?? null;
  const forecast = useForecast(forecastDate, 7);
  const grid = useGrid(forecastDate, "rain");
  const alerts = useAlerts(forecastDate ?? undefined);
  const validation = useValidationMetrics();
  const view = transformOverview(current.data, grid.data, forecast.data, health.data);
  const alertView = transformAlerts(alerts.data);
  const primaryLoading = current.isLoading || (forecastDate !== null && forecast.isLoading);
  const primaryError = current.error || forecast.error;
  const retry = () => {
    void Promise.all([current.mutate(), forecast.mutate(), grid.mutate(), health.mutate(), alerts.mutate(), validation.mutate()]);
  };

  const metrics = view.kpis.filter((item) => ["rainfall", "tmax", "reservoir", "peak", "skill"].includes(item.id));
  const rainfallPoints = view.forecast
    .filter((point) => point.available)
    .map((point) => {
      const fallback = point.rainfall_p50_mm ?? point.rainfall_p10_mm ?? point.rainfall_p90_mm ?? 0;
      return {
        date: point.date,
        p10: point.rainfall_p10_mm ?? fallback,
        p50: point.rainfall_p50_mm ?? fallback,
        p90: point.rainfall_p90_mm ?? fallback,
        climatology: point.climatology_mm ?? 0,
      };
    });
  const peakPoint = [...rainfallPoints].sort((a, b) => b.p50 - a.p50)[0];
  const modelMetrics = validation.data
    ? [
        { label: "MAE", value: validation.data.ours.mae ?? "Unavailable", unit: validation.data.unit },
        { label: "RMSE", value: validation.data.ours.rmse ?? "Unavailable", unit: validation.data.unit },
        { label: "CSI @ 1mm", value: validation.data.ours["csi@1mm"] ?? "Unavailable", unit: "score" },
      ]
    : [];
  const sectors = [
    { sector: "Water security", status: "Unavailable", tone: "neutral" as const },
    { sector: "Agriculture", status: "Unavailable", tone: "neutral" as const },
    { sector: "Heat stress", status: "Unavailable", tone: "neutral" as const },
    { sector: "Flood risk", status: "Unavailable", tone: "neutral" as const },
  ];
  const cycle = ["INGEST", "REGRID", "ASSIMILATE", "FORECAST", "IMPACT", "RENDER"].map((label, index) => ({
    label,
    status: index < 3 ? "completed" as const : index === 3 ? "active" as const : "pending" as const,
  }));

  if (primaryError) {
    return <ErrorState title="Overview unavailable" description="The latest basin state or forecast could not be loaded." onRetry={retry} />;
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-row-gap">
      <DashboardHeader title="Overview" metadata={`Cauvery Basin · Updated ${formatIst(view.updatedIst)}`} />

      <section className="grid h-40 shrink-0 grid-cols-5 gap-column-gap" aria-label="Current climate metrics">
        {primaryLoading ? Array.from({ length: 5 }, (_, index) => <Skeleton key={index} variant="card" />) : metrics.map((metric) => (
          <OverviewMetricCard
            key={metric.id}
            label={metric.label}
            value={metric.available && metric.value !== null ? metric.value : "Unavailable"}
            unit={metric.available ? metric.unit : ""}
            detail={metric.detail}
            tone={metric.tone}
            icon={metricIcons[metric.id as keyof typeof metricIcons]}
            visual={metric.visual}
            series={metric.series}
          />
        ))}
      </section>

      <section className="grid min-h-96 flex-1 grid-cols-5 gap-column-gap" aria-label="Climate state and forecast">
        <GlassCard padding="none" className="col-span-3 flex min-h-0 flex-col overflow-hidden">
          <div className="flex min-h-cycle items-center justify-between gap-card border-b border-divider px-card">
            <div className="flex items-center gap-3"><h2 className="text-title text-textPrimary">Live climate state</h2><span className="inline-flex items-center gap-1 text-caption text-primary"><span className="h-2 w-2 rounded-pill bg-primary" aria-hidden="true" />Live</span></div>
            <div className="flex items-center gap-1" role="tablist" aria-label="Climate variable"><TabPill active size="sm" className="rounded-control border-positive bg-card text-textPrimary">Rainfall</TabPill><TabPill size="sm" className="rounded-control border-transparent bg-transparent">Temperature</TabPill><TabPill size="sm" className="rounded-control border-transparent bg-transparent">Wind</TabPill><TabPill size="sm" className="rounded-control border-transparent bg-transparent">LST</TabPill><button type="button" className="ml-1 flex h-control w-control items-center justify-center rounded-control border border-cardBorder bg-card text-textSecondary hover:text-primary" aria-label="Open map layers"><Layers3 className="h-4 w-4" aria-hidden="true" /></button></div>
          </div>
          {grid.isLoading ? <div className="flex min-h-0 flex-1 items-center justify-center p-6"><Skeleton variant="block" className="h-full" /></div> : view.rainfallGrid?.values.length ? <StaticBasinMap className="min-h-0 flex-1" values={view.rainfallGrid.values} /> : <EmptyState title="Rainfall map unavailable" description="No grid values are available for the current basin date." className="m-4 flex-1 justify-center" />}
          <MapPlaybackBar startLabel={formatDate(view.forecast[0]?.date)} currentLabel={`${formatDate(view.current?.date_ist)} · Today`} endLabel={formatDate(view.forecast.at(-1)?.date)} />
        </GlassCard>

        <div className="col-span-2 flex min-h-0 flex-col gap-row-gap">
          <GlassCard padding="none" className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="flex min-h-cycle items-center justify-between px-card"><h2 className="text-title text-textPrimary">7-day forecast</h2><span className="inline-flex min-h-control items-center rounded-control border border-cardBorder bg-card px-3 text-label text-textSecondary">Rainfall · mm/day</span></div>
            {forecast.isLoading ? <div className="flex flex-1 items-center p-4"><Skeleton variant="block" className="h-full" /></div> : rainfallPoints.length ? <ForecastRibbonChart className="min-h-0 flex-1 px-2" points={rainfallPoints} unit={forecast.data?.rain.unit ?? "mm/day"} peakLabel={peakPoint ? `Peak ${peakPoint.p50} ${forecast.data?.rain.unit ?? "mm/day"} · ${formatDate(peakPoint.date)}` : undefined} /> : <EmptyState title="Forecast unavailable" description="No p10–p90 rainfall guidance is available." className="m-4 flex-1 justify-center" />}
            {peakPoint ? <div className="flex min-h-control items-center gap-2 border-t border-warning bg-warning-soft px-card text-caption text-textPrimary"><span className="h-2 w-2 rounded-pill bg-warning" aria-hidden="true" />Basin mean p10–p90 guidance available</div> : null}
          </GlassCard>
          <GlassCard padding="none" className="h-40 shrink-0 overflow-hidden"><div className="flex h-control items-center px-card"><h2 className="text-title text-textPrimary">Sector status</h2></div><StatusList className="px-card" density="compact" items={sectors.map((item) => ({ id: item.sector, label: item.sector, value: item.status, tone: item.tone, icon: sectorIcons[item.sector as keyof typeof sectorIcons] }))} /></GlassCard>
        </div>
      </section>

      <section className="grid h-48 shrink-0 grid-cols-5 gap-column-gap" aria-label="Model and operational status">
        <div className="col-span-3 grid grid-cols-2 gap-column-gap">
          <GlassCard className="min-h-0 overflow-hidden"><SectionHeader size="compact" title="Model performance" rightSlot={<Info className="h-3.5 w-3.5 text-textTertiary" aria-hidden="true" />} />{validation.isLoading ? <Skeleton variant="block" /> : modelMetrics.length ? <MetricMatrix items={modelMetrics} /> : <EmptyState title="Validation unavailable" description="No model metrics were returned." />}{validation.data?.ours["csi@1mm"] !== null && validation.data?.ours["csi@1mm"] !== undefined ? <div className="mt-row-gap flex items-center gap-2 border-t border-divider pt-2 text-caption text-positive"><TrendingUp className="h-4 w-4" aria-hidden="true" />Validation metrics loaded</div> : null}</GlassCard>
          <GlassCard className="min-h-0 overflow-hidden"><SectionHeader size="compact" title="System health" rightSlot={<Info className="h-3.5 w-3.5 text-textTertiary" aria-hidden="true" />} />{health.isLoading ? <Skeleton variant="block" /> : health.data ? <StatusList density="compact" items={[{ id: "api", label: "Data service", value: health.data.status, detail: `${health.data.datacube_days} days`, tone: health.data.status === "ok" ? "positive" : "critical" }, { id: "model", label: "Model", value: health.data.model_loaded ? "Ready" : "Unavailable", tone: health.data.model_loaded ? "positive" : "warning" }]} /> : <EmptyState title="System health unavailable" description="The health endpoint returned no data." />}</GlassCard>
        </div>
        <GlassCard className="col-span-2 min-h-0 overflow-hidden"><SectionHeader size="compact" title="Recent alerts" rightSlot={<span className="text-caption text-primary">{alertView.evaluatedDateIst ? formatIst(alertView.evaluatedDateIst) : "Unavailable"}</span>} />{alerts.isLoading ? <Skeleton variant="block" /> : alertView.alerts.length ? <StatusList density="compact" markerPosition="start" items={alertView.alerts.slice(0, 3).map((alert) => ({ id: alert.id, label: alert.title, value: `${alert.observedValue ?? "Unavailable"} ${alert.unit}`, detail: formatIst(alert.timeIst), tone: alert.tone }))} /> : <EmptyState title="No active alerts" description="No alerts were returned for the current basin date." />}</GlassCard>
      </section>

      <RichCycleStrip cycleLabel="CYCLE" stages={cycle} />
    </div>
  );
}
