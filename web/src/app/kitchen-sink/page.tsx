"use client"

import { useState } from "react"
import {
  Activity,
  Bell,
  CloudRain,
  Database,
  Gauge,
  Layers3,
  MapPin,
  RefreshCw,
  Settings2,
  Wind,
} from "lucide-react"

import {
  AlertRow,
  CardSkeleton,
  ChartSkeleton,
  CommandK,
  ConnectionError,
  CountdownPill,
  CycleStrip,
  DashboardHeader,
  DashboardShell,
  Divider,
  DonutGauge,
  Dropdown,
  EmptyState,
  ErrorState,
  ForecastRibbonChart,
  GlassCard,
  IncidentBanner,
  InlineLegend,
  KPIHeaderStrip,
  LivePulseDot,
  MainLayout,
  MapPlaybackBar,
  MetricMatrix,
  MetricTile,
  MiniBarChart,
  MiniSparkline,
  OverviewMetricCard,
  RichCycleStrip,
  ScaleLegend,
  SectionHeader,
  Sidebar,
  StaticBasinMap,
  StatusList,
  Skeleton,
  Slider,
  TableSkeleton,
  StatusPill,
  TabPill,
  TimelineDot,
  TogglePill,
  TopBar,
  type CyclePhase,
} from "@/components"
import { LiveApiSmoke } from "@/components/LiveApiSmoke"
import {
  overviewForecast,
  overviewModelMetrics,
  overviewPipeline,
  overviewRainfallGrid,
  overviewReservoirs,
  overviewSectorStatus,
} from "@/lib/mock"

const buttonClass =
  "inline-flex min-h-target items-center justify-center gap-2 rounded-button bg-primary px-4 text-label text-primary-foreground transition-colors duration-fast hover:bg-primary-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
const secondaryButtonClass =
  "inline-flex min-h-target items-center justify-center gap-2 rounded-button border border-cardBorder bg-card px-4 text-label text-textSecondary transition-colors duration-fast hover:border-primary hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"

const swatches = [
  ["Background", "#F7F8FA", "bg-background"],
  ["Workspace", "rgba(249,250,250,.92)", "bg-workspace"],
  ["Card", "rgba(255,255,255,.88)", "bg-card"],
  ["Card solid", "#FEFFFE", "bg-card-solid"],
  ["Card muted", "#F1F6FA", "bg-card-muted"],
  ["Card border", "#E1E4E8", "bg-cardBorder"],
  ["Divider", "#D8DCE2", "bg-divider"],
  ["Text primary", "#111318", "bg-textPrimary"],
  ["Text secondary", "#56606B", "bg-textSecondary"],
  ["Text tertiary", "#7A8390", "bg-textTertiary"],
  ["Text inverse", "#FFFFFF", "bg-textInverse"],
  ["Primary", "#0061FE", "bg-primary"],
  ["Primary strong", "#0854FE", "bg-primary-strong"],
  ["Primary soft", "#EAF2FE", "bg-primary-soft"],
  ["Positive", "#07945B", "bg-positive"],
  ["Positive soft", "#E5FCED", "bg-positive-soft"],
  ["Warning", "#F57602", "bg-warning"],
  ["Warning soft", "#FEF7EA", "bg-warning-soft"],
  ["Critical", "#ED3335", "bg-critical"],
  ["Critical strong", "#EC3736", "bg-critical-strong"],
  ["Critical soft", "#FDEBEC", "bg-critical-soft"],
  ["Rain 0", "#DDEBFF", "bg-rain-0"],
  ["Rain 25", "#ABD2FF", "bg-rain-25"],
  ["Rain 50", "#8FBEFD", "bg-rain-50"],
  ["Rain 75", "#3D91FB", "bg-rain-75"],
  ["Rain 100", "#0061FE", "bg-rain-100"],
  ["Temperature low", "#FEF3C7", "bg-temperature-low"],
  ["Temperature high", "#B91C1C", "bg-temperature-high"],
  ["Anomaly negative", "#E56604", "bg-anomaly-negative"],
  ["Anomaly neutral", "#FFFFFF", "bg-anomaly-neutral"],
  ["Anomaly positive", "#143CB9", "bg-anomaly-positive"],
  ["Missing", "#E5E7EB", "bg-missing"],
] as const

const stations = [
  { value: "delhi", label: "New Delhi" },
  { value: "mumbai", label: "Mumbai" },
  { value: "chennai", label: "Chennai" },
  { value: "offline", label: "Offline station", disabled: true },
] as const

const phases: readonly CyclePhase[] = [
  "INGEST",
  "REGRID",
  "ASSIMILATE",
  "FORECAST",
  "IMPACT",
  "RENDER",
]

function CatalogSection({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle: string
  children: React.ReactNode
}) {
  return (
    <section aria-labelledby={title.toLowerCase().replaceAll(" ", "-")}>
      <SectionHeader
        title={title}
        subtitle={subtitle}
        headingLevel={2}
        rightSlot={<StatusPill label="Catalog" tone="info" size="sm" />}
      />
      {children}
    </section>
  )
}

export default function KitchenSinkPage() {
  const [observations, setObservations] = useState(true)
  const [confidence, setConfidence] = useState(72)
  const [station, setStation] = useState("delhi")
  const [tab, setTab] = useState("overview")
  const [phase, setPhase] = useState<CyclePhase>("ASSIMILATE")
  const [retryCount, setRetryCount] = useState(0)

  return (
    <MainLayout
      density="default"
      sidebarSlot={<Sidebar />}
      topBarSlot={
        <TopBar
          breadcrumbs={[
            { id: "system", label: "MausamSetu मौसम सेतु" },
            { id: "catalog", label: "Component catalog" },
          ]}
          actions={
            <>
              <LivePulseDot label="Catalog live" />
              <CommandK label="Commands" title="Catalog commands" />
            </>
          }
        />
      }
      mainAriaLabel="MausamSetu component catalog"
    >
      <div className="mx-auto max-w-7xl space-y-workspace">
        <GlassCard variant="floating" padding="spacious">
          <div className="flex flex-wrap items-start justify-between gap-card-lg">
            <div>
              <p className="text-eyebrow uppercase text-primary">Internal component catalog</p>
              <h1 className="mt-2 text-page text-textPrimary">MausamSetu मौसम सेतु</h1>
              <p className="mt-2 max-w-2xl text-body text-textSecondary">
                Desktop-first reference for shared weather workspace components, interaction states, and design tokens.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusPill label="27 components" tone="positive" />
              <CountdownPill value="Cycle 12:30" label="Next catalog refresh" />
            </div>
          </div>
        </GlassCard>

        <CatalogSection title="Color system" subtitle="Every configured semantic and climate color with its exact source value and Tailwind utility.">
          <div className="grid grid-cols-3 gap-column-gap xl:grid-cols-6">
            {swatches.map(([name, hex, className]) => (
              <GlassCard key={name} padding="none" className="overflow-hidden">
                <div className={`h-16 ${className}`} aria-hidden="true" />
                <div className="p-3">
                  <p className="text-label text-textPrimary">{name}</p>
                  <p className="font-mono text-caption text-textSecondary">{hex}</p>
                  <p className="mt-1 font-mono text-caption text-textTertiary">{className}</p>
                </div>
              </GlassCard>
            ))}
          </div>
        </CatalogSection>

        <CatalogSection title="Typography specimen" subtitle="Sans hierarchy and tabular mono data styles.">
          <GlassCard>
            <div className="grid grid-cols-2 gap-card-lg">
              <div className="space-y-row-gap">
                <p className="text-hero text-textPrimary">48.7</p>
                <p className="text-page text-textPrimary">Page heading / 24</p>
                <p className="text-title text-textPrimary">Title / 16</p>
                <p className="text-body text-textSecondary">Body / 14 — Regional guidance and forecast context.</p>
                <p className="text-label text-textSecondary">Label / 13</p>
                <p className="text-caption text-textTertiary">Caption / 11</p>
                <p className="text-eyebrow uppercase text-primary">Eyebrow / tracked</p>
              </div>
              <div className="rounded-card bg-card-muted p-card-lg">
                <p className="text-eyebrow uppercase text-textTertiary">Mono specimen</p>
                <p className="mt-3 font-mono text-page tabular-nums text-textPrimary">2026-07-25 12:30 IST</p>
                <p className="mt-2 font-mono text-body tabular-nums text-textSecondary">RAIN 084.2 mm · WIND 18 km/h</p>
              </div>
            </div>
          </GlassCard>
        </CatalogSection>

        <CatalogSection title="Layout primitives" subtitle="MainLayout, Sidebar, and TopBar frame this page; compact shell specimens appear below.">
          <GlassCard padding="none" className="overflow-hidden">
            <TopBar
              density="compact"
              position="static"
              breadcrumbs={[{ id: "workspace", label: "Workspace" }, { id: "overview", label: "Overview" }]}
              actions={<button className={secondaryButtonClass}>Share</button>}
            />
            <div className="grid grid-cols-4 bg-background">
              <div className="border-r border-divider p-card">
                <p className="text-eyebrow uppercase text-textTertiary">Sidebar region</p>
                <p className="mt-2 text-label text-primary">Overview active</p>
                <p className="mt-2 text-label text-textTertiary">Forecast</p>
              </div>
              <div className="col-span-3 p-card-lg text-body text-textSecondary">MainLayout content region</div>
            </div>
          </GlassCard>
        </CatalogSection>

        <CatalogSection title="Dashboard route shell" subtitle="Reference-aligned shell, linked sidebar navigation, pilot selector, identity block, and page action header.">
          <div className="h-screen overflow-hidden rounded-shell border border-cardBorder">
            <DashboardShell className="h-full p-card" contentClassName="p-card">
              <DashboardHeader
                title="Overview"
                metadata="Cauvery Basin · Updated 09:34 IST"
              />
              <div className="mt-card rounded-panel border border-dashed border-divider bg-card-muted p-card-lg">
                <p className="text-label text-textPrimary">Dashboard route content</p>
                <p className="mt-1 text-body text-textSecondary">
                  Step 1 pages intentionally contain headings only while shared navigation is verified.
                </p>
              </div>
            </DashboardShell>
          </div>
        </CatalogSection>

        <CatalogSection title="Overview visualization components" subtitle="Reusable metric, map, uncertainty-chart, status, matrix, legend, playback, and connected-cycle components introduced for Screen 1.">
          <div className="grid grid-cols-3 gap-column-gap">
            <OverviewMetricCard
              label="BASIN RAINFALL 24H"
              value={32.4}
              unit="mm"
              detail="+78% vs normal"
              tone="info"
              icon={<CloudRain className="h-5 w-5" />}
              visual="sparkline"
              series={[14, 17, 15, 22, 26, 24, 32.4]}
            />
            <GlassCard className="col-span-2" padding="none">
              <StaticBasinMap
                className="h-48 rounded-card"
                values={overviewRainfallGrid.values}
                reservoirs={overviewReservoirs}
              />
              <MapPlaybackBar startLabel="Jul 12" currentLabel="Jul 15 · Today" endLabel="Jul 18" />
            </GlassCard>
          </div>
          <div className="mt-column-gap grid grid-cols-2 gap-column-gap">
            <GlassCard>
              <ForecastRibbonChart
                className="h-52"
                points={overviewForecast.map((point) => ({ date: point.date, p10: point.rainfall_p10_mm, p50: point.rainfall_p50_mm, p90: point.rainfall_p90_mm, climatology: point.climatology_mm }))}
                unit="mm"
                peakLabel="Peak 48 mm · Jul 19"
              />
              <ScaleLegend className="mt-2" label="Rainfall (mm)" ticks={["0", "25", "50", "75", "100+"]} />
            </GlassCard>
            <GlassCard>
              <MetricMatrix items={overviewModelMetrics} />
              <Divider className="my-card" />
              <StatusList items={overviewSectorStatus.map((item) => ({ id: item.sector, label: item.sector, value: item.status, tone: item.tone }))} />
            </GlassCard>
          </div>
          <RichCycleStrip className="mt-column-gap" cycleLabel="CYCLE 46" stages={overviewPipeline} />
        </CatalogSection>

        <CatalogSection title="Container primitives" subtitle="Card elevation, padding, interaction, disabled treatment, dividers, and KPI grouping.">
          <div className="grid grid-cols-5 gap-column-gap">
            <GlassCard><p className="text-label">Default card</p></GlassCard>
            <GlassCard variant="muted"><p className="text-label">Muted card</p></GlassCard>
            <GlassCard variant="floating"><p className="text-label">Floating card</p></GlassCard>
            <GlassCard variant="interactive" tabIndex={0}><p className="text-label">Interactive — hover or focus</p></GlassCard>
            <GlassCard disabled><p className="text-label">Disabled card</p></GlassCard>
          </div>
          <GlassCard className="mt-column-gap">
            <SectionHeader title="Section header" subtitle="Default header with an action slot." rightSlot={<button className={secondaryButtonClass}>Action</button>} />
            <Divider />
            <div className="mt-card flex h-16 items-stretch gap-card"><span className="text-body text-textSecondary">Horizontal</span><Divider orientation="vertical" emphasis="subtle" /><span className="text-body text-textSecondary">Vertical subtle</span></div>
          </GlassCard>
          <KPIHeaderStrip density="compact" className="mt-column-gap">
            <div><p className="text-caption text-textSecondary">Stations</p><p className="font-mono text-title">142</p></div>
            <div><p className="text-caption text-textSecondary">Reporting</p><p className="font-mono text-title">98.6%</p></div>
            <div><p className="text-caption text-textSecondary">Cycle</p><p className="font-mono text-title">12 UTC</p></div>
          </KPIHeaderStrip>
        </CatalogSection>

        <CatalogSection title="Data display" subtitle="Metrics, gauges, trends, bars, legends, and every semantic status tone.">
          <div className="grid grid-cols-5 gap-column-gap">
            <MetricTile label="Temperature" value="31.4" unit="°C" visual={<MiniSparkline ariaLabel="Temperature trend" points={[28, 29, 28, 31, 30, 32]} />} />
            <MetricTile variant="with-gauge" label="Humidity" value="84" unit="%" visual={<DonutGauge value={84} ariaLabel="Humidity 84 percent" tone="info" />} />
            <MetricTile variant="with-delta" label="Rainfall" value="48.7" unit="mm" delta="+12.4% vs normal" deltaTone="positive" />
            <MetricTile label="Loading metric" value="--" loading />
            <MetricTile label="Disabled metric" value="18" unit="km/h" disabled />
          </div>
          <div className="mt-column-gap grid grid-cols-3 gap-column-gap">
            <GlassCard>
              <p className="text-label text-textPrimary">Rainfall by window</p>
              <MiniBarChart ariaLabel="Rainfall by forecast window" bars={[{ id: "6h", label: "6h", value: 18, tone: "info" }, { id: "12h", label: "12h", value: 35, tone: "warning" }, { id: "24h", label: "24h", value: 52, tone: "critical" }]} />
            </GlassCard>
            <GlassCard>
              <p className="text-label text-textPrimary">Gauge tones</p>
              <div className="mt-2 flex gap-card"><DonutGauge value={72} ariaLabel="Availability 72 percent" tone="positive" /><DonutGauge value={38} ariaLabel="Risk 38 percent" tone="warning" disabled /></div>
            </GlassCard>
            <GlassCard>
              <p className="text-label text-textPrimary">Legend and sparkline</p>
              <MiniSparkline ariaLabel="Pressure trend" points={[1004, 1008, 1006, 1011, 1010]} tone="positive" />
              <InlineLegend className="mt-3" items={[{ id: "normal", label: "Normal", tone: "positive", value: "62%" }, { id: "watch", label: "Watch", tone: "warning", value: "24%" }, { id: "risk", label: "Risk", tone: "critical", value: "14%" }]} />
            </GlassCard>
          </div>
          <div className="mt-column-gap flex flex-wrap gap-2">
            <StatusPill label="Positive" tone="positive" />
            <StatusPill label="Warning" tone="warning" />
            <StatusPill label="Critical" tone="critical" />
            <StatusPill label="Information" tone="info" />
            <StatusPill label="Neutral" tone="neutral" />
            <StatusPill label="Disabled" tone="neutral" disabled />
          </div>
        </CatalogSection>

        <CatalogSection title="Navigation and controls" subtitle="Controlled inputs, active states, hover targets, command menu, and disabled controls.">
          <GlassCard>
            <div className="flex flex-wrap items-end gap-card-lg">
              <div role="tablist" aria-label="Catalog views" className="flex gap-2">
                {[["overview", "Overview"], ["forecast", "Forecast"], ["impact", "Impact"]].map(([id, label]) => <TabPill key={id} active={tab === id} onClick={() => setTab(id)}>{label}</TabPill>)}
                <TabPill disabled>Disabled</TabPill>
              </div>
              <TogglePill pressed={observations} onPressedChange={setObservations}><Activity className="h-4 w-4" />Observations</TogglePill>
              <TogglePill disabled><Bell className="h-4 w-4" />Disabled</TogglePill>
              <button className={buttonClass}><CloudRain className="h-4 w-4" />Primary hover</button>
              <button className={secondaryButtonClass}><Settings2 className="h-4 w-4" />Secondary</button>
              <button className={buttonClass} disabled>Disabled action</button>
              <CommandK label="Open command menu" enableShortcut={false} />
            </div>
            <div className="mt-card-lg grid grid-cols-3 gap-card-lg">
              <Dropdown label="Station" options={stations} value={station} onValueChange={setStation} />
              <Dropdown label="Disabled station" options={stations} defaultValue="mumbai" disabled />
              <Slider label="Forecast confidence" value={confidence} onValueChange={setConfidence} formatValue={(value) => `${value}%`} />
            </div>
          </GlassCard>
        </CatalogSection>

        <CatalogSection title="Timeline and phase" subtitle="CycleStrip active phases and timeline statuses.">
          <GlassCard>
            <CycleStrip activePhase={phase} phases={phases} />
            <div className="mt-card flex flex-wrap gap-2">
              {phases.map((item) => <button key={item} className={phase === item ? buttonClass : secondaryButtonClass} onClick={() => setPhase(item)}>{item}</button>)}
            </div>
            <Divider className="my-card-lg" />
            <ol className="grid grid-cols-4 gap-card">
              {[{ status: "completed" as const, label: "Data received", time: "12:04" }, { status: "active" as const, label: "Model running", time: "12:16" }, { status: "pending" as const, label: "Guidance pending", time: "12:28" }, { status: "critical" as const, label: "Threshold event", time: "12:31" }].map((item) => <li key={item.label} className="flex gap-3"><TimelineDot status={item.status} label={item.label} /><div><p className="text-label text-textPrimary">{item.label}</p><p className="font-mono text-caption text-textTertiary">{item.time}</p></div></li>)}
            </ol>
          </GlassCard>
        </CatalogSection>

        <CatalogSection title="Alerts" subtitle="All alert severities plus a prominent critical incident.">
          <div className="grid grid-cols-2 gap-column-gap">
            <AlertRow severity="info" timestamp="12:04 IST" title="Cycle initialized" description="Observation ingest started across the national network." />
            <AlertRow severity="positive" timestamp="12:08 IST" title="Quality checks passed" description="All mandatory surface parameters are available." />
            <AlertRow severity="warning" timestamp="12:14 IST" title="Station latency" description="Three stations are reporting outside the expected window." action={<button className={secondaryButtonClass}>Review</button>} />
            <AlertRow severity="critical" timestamp="12:19 IST" title="Extreme rainfall threshold" description="Nowcast guidance exceeds district response criteria." />
          </div>
          <IncidentBanner className="mt-column-gap" title="Cloudburst risk elevated" description="High-impact guidance is active for two monitored districts." actions={<><button className={buttonClass}>Open incident</button><button className={secondaryButtonClass}>Dismiss</button></>} />
        </CatalogSection>

        <CatalogSection title="Live indicators" subtitle="Compact live, countdown, connectivity, and update treatments.">
          <GlassCard>
            <div className="flex flex-wrap items-center gap-card-lg">
              <span className="inline-flex items-center gap-2 text-label text-textPrimary"><LivePulseDot />Live observations</span>
              <CountdownPill value="00:08:42" />
              <StatusPill label="API connected" tone="positive" />
              <StatusPill label="2 delayed" tone="warning" />
              <MiniSparkline className="max-w-xs" ariaLabel="Live wind speed" points={[8, 10, 9, 14, 12, 16, 15]} tone="info" />
            </div>
          </GlassCard>
        </CatalogSection>

        <CatalogSection title="Utility states" subtitle="Skeleton shapes, empty content, retryable error, loading, and disabled examples.">
          <div className="grid grid-cols-5 gap-column-gap">
            <GlassCard><Skeleton variant="text" /><Skeleton variant="title" className="mt-3" /></GlassCard>
            <GlassCard><Skeleton variant="avatar" /></GlassCard>
            <Skeleton variant="card" />
            <GlassCard><Skeleton variant="block" /></GlassCard>
            <GlassCard disabled><p className="text-label text-textPrimary">Disabled container</p><Skeleton className="mt-3" /></GlassCard>
          </div>
          <div className="mt-column-gap grid grid-cols-2 gap-column-gap">
            <EmptyState title="No advisories" description="No district advisories match the current filters." icon={<Database className="h-8 w-8" />} action={<button className={secondaryButtonClass}>Clear filters</button>} />
            <ErrorState title="Forecast unavailable" description={`The latest guidance could not be loaded. Retry attempts: ${retryCount}.`} onRetry={() => setRetryCount((count) => count + 1)} />
          </div>
          <div className="mt-column-gap grid grid-cols-2 gap-column-gap">
            <CardSkeleton label="Loading metric card" />
            <ChartSkeleton label="Loading forecast chart" />
          </div>
          <div className="mt-column-gap grid grid-cols-2 gap-column-gap">
            <TableSkeleton rows={3} columns={3} label="Loading alert table" />
            <ConnectionError
              title="API connection lost"
              description="The live service is unavailable; retry when it is reachable again."
              onRetry={() => setRetryCount((count) => count + 1)}
            />
          </div>
        </CatalogSection>

        <CatalogSection title="Sample composition" subtitle="A light Overview reference assembled exclusively from shared components.">
          <div className="rounded-panel border border-cardBorder bg-background p-workspace">
            <SectionHeader size="spacious" title="National Overview" subtitle="Saturday, 25 July · 12 UTC cycle" rightSlot={<div className="flex gap-2"><StatusPill label="Operational" tone="positive" /><CountdownPill value="Next update 20m" /></div>} />
            <KPIHeaderStrip>
              <div className="flex items-center gap-3"><MapPin className="h-5 w-5 text-primary" /><div><p className="text-caption text-textSecondary">Reporting stations</p><p className="font-mono text-title">142 / 146</p></div></div>
              <div className="flex items-center gap-3"><CloudRain className="h-5 w-5 text-primary" /><div><p className="text-caption text-textSecondary">24h rainfall</p><p className="font-mono text-title">48.7 mm</p></div></div>
              <div className="flex items-center gap-3"><Wind className="h-5 w-5 text-primary" /><div><p className="text-caption text-textSecondary">Peak wind</p><p className="font-mono text-title">34 km/h</p></div></div>
              <div className="flex items-center gap-3"><Gauge className="h-5 w-5 text-primary" /><div><p className="text-caption text-textSecondary">Confidence</p><p className="font-mono text-title">84%</p></div></div>
            </KPIHeaderStrip>
            <div className="mt-column-gap grid grid-cols-3 gap-column-gap">
              <GlassCard className="col-span-2">
                <SectionHeader size="compact" title="Rainfall outlook" subtitle="Six-hour accumulation guidance" rightSlot={<InlineLegend items={[{ id: "light", label: "Light", tone: "info" }, { id: "heavy", label: "Heavy", tone: "critical" }]} />} />
                <MiniBarChart ariaLabel="Overview rainfall outlook" bars={[{ id: "d1", label: "Now", value: 12 }, { id: "d2", label: "+6h", value: 25 }, { id: "d3", label: "+12h", value: 38, tone: "warning" }, { id: "d4", label: "+18h", value: 52, tone: "critical" }, { id: "d5", label: "+24h", value: 31, tone: "warning" }]} />
              </GlassCard>
              <GlassCard variant="muted">
                <SectionHeader size="compact" title="System health" rightSlot={<LivePulseDot />} />
                <div className="flex items-center gap-card"><DonutGauge value={96} ariaLabel="System health 96 percent" tone="positive" /><div className="space-y-2"><StatusPill label="Feeds healthy" tone="positive" size="sm" /><StatusPill label="3 delayed" tone="warning" size="sm" /></div></div>
              </GlassCard>
            </div>
            <CycleStrip className="mt-column-gap" activePhase="FORECAST" />
          </div>
        </CatalogSection>

        <LiveApiSmoke />

        <GlassCard variant="muted">
          <div className="flex items-center justify-between gap-card">
            <div className="flex items-center gap-3"><Layers3 className="h-5 w-5 text-primary" /><p className="text-body text-textSecondary">Catalog + live API smoke — no product route content.</p></div>
            <button className={secondaryButtonClass} onClick={() => setRetryCount(0)}><RefreshCw className="h-4 w-4" />Reset demo</button>
          </div>
        </GlassCard>
      </div>
    </MainLayout>
  )
}
