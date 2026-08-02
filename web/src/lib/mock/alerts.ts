import { MOCK_IST_TIMESTAMP, buildCauveryGrid, type PipelineStatus, type SemanticTone } from "./_shared";

export type MockAlertSeverity = "info" | "warning" | "critical";
export type MockAlertType = "rainfall" | "heat" | "extreme_heat";

export interface MockAlertItem {
  id: string;
  type: MockAlertType;
  severity: MockAlertSeverity;
  title: string;
  description: string;
  observed_value: number;
  threshold: number;
  unit: string;
  evaluated_date: string;
  data_source: string;
  threshold_source: string;
}

export const alertsResponse = {
  evaluated_date_ist: MOCK_IST_TIMESTAMP,
  alerts: [
    { id: "rain-kodagu-20260715", type: "rainfall", severity: "critical", title: "Heavy rain warning", description: "Kodagu district rainfall exceeds the critical daily threshold.", observed_value: 82, threshold: 45, unit: "mm/day", evaluated_date: "2026-07-15", data_source: "cauvery.nc rain", threshold_source: "train_p99_2020_2021" },
    { id: "heat-trichy-20260715", type: "heat", severity: "warning", title: "Heat stress watch", description: "Tiruchirappalli maximum temperature exceeds the heat warning threshold.", observed_value: 40.8, threshold: 40, unit: "degC", evaluated_date: "2026-07-15", data_source: "cauvery.nc tmax", threshold_source: "config.HEAT_STRESS_TEMP" },
  ] satisfies MockAlertItem[],
  thresholds: {
    rain_warning_mm_per_day: 30,
    rain_critical_mm_per_day: 45,
    heat_warning_c: 40,
    heat_critical_c: 45,
    sources: {
      rain_warning_mm_per_day: "train_p95_2020_2021",
      rain_critical_mm_per_day: "train_p99_2020_2021",
      heat_warning_c: "config.HEAT_STRESS_TEMP",
      heat_critical_c: "config.EXTREME_HEAT_TEMP",
    },
  },
};

export const activeIncident = {
  id: "INC-CAV-2026-0715",
  status: "ACTIVE INCIDENT",
  title: "Heavy rainfall warning · Kodagu district · Karnataka",
  projected_peak_ist: "2026-07-19T14:00:00+05:30",
  severity: "Severe",
  started_at_ist: "2026-07-15T08:47:00+05:30",
  population_at_risk_people: 340_000,
  estimated_evacuations_people: 4_200,
  critical_infrastructure_facilities: 59,
  agencies_notified: 3,
  response_coordinator: "MausamSetu",
} as const;

export const incidentRainfallGrid = {
  unit: "mm/day",
  values: buildCauveryGrid(42, 19, 0, 110),
  district: "Kodagu",
  updated_at_ist: MOCK_IST_TIMESTAMP,
};

export const liveAlertFeed = [
  { time_ist: "2026-07-15T09:34:00+05:30", label: "Severe", title: "Heavy rain warning", detail: "Kodagu district · >75 mm/day expected Jul 18–20", tone: "critical", actions: ["Acknowledge", "Escalate"] },
  { time_ist: "2026-07-15T09:12:00+05:30", label: "Elevated", title: "Soil saturation", detail: "Kabini sub-basin at 87% saturation, still rising", tone: "warning", actions: ["Acknowledge", "Escalate"] },
  { time_ist: "2026-07-15T08:47:00+05:30", label: "Advisory", title: "Reservoir inflow", detail: "Mettur expected +40% inflow Jul 19–21", tone: "info", actions: ["Acknowledge", "Escalate"] },
  { time_ist: "2026-07-15T08:23:00+05:30", label: "Info", title: "Assimilation cycle 46 complete", detail: "42 ms latency · Nominal", tone: "neutral", actions: [] },
  { time_ist: "2026-07-15T08:00:00+05:30", label: "Info", title: "Daily model report generated", detail: "JJAS outlook and scenario pack ready", tone: "neutral", actions: [] },
] satisfies Array<{ time_ist: string; label: string; title: string; detail: string; tone: SemanticTone; actions: string[] }>;

export const incidentTimeline = [
  { label: "Detected", time_ist: "2026-07-15T08:47:00+05:30", detail: "Assimilation flagged surge", status: "completed" },
  { label: "Advisories issued", time_ist: "2026-07-15T09:22:00+05:30", detail: "3 agencies notified", status: "completed" },
  { label: "Response coordination", time_ist: "2026-07-15T09:34:00+05:30", detail: "Live monitoring active", status: "active" },
  { label: "Onset window", time_ist: "2026-07-18T06:00:00+05:30", detail: "First heavy bands expected", status: "pending" },
  { label: "PEAK · 82 mm/hr", time_ist: "2026-07-19T14:00:00+05:30", detail: "Highest expected intensity", status: "critical" },
] satisfies Array<{ label: string; time_ist: string; detail: string; status: PipelineStatus }>;

export const responseCoordination = [
  { agency: "NDMA", role: "National Command", detail: "Acknowledged by Additional Secretary", status: "Notified · 09:22 IST", tone: "positive" },
  { agency: "TN State DMA", role: "State command", detail: "Active coordination · 3h 12m response", status: "Acknowledged · 09:15 IST", tone: "positive" },
  { agency: "Karnataka DMA", role: "Field response", detail: "Field teams mobilized to Kodagu", status: "Active response", tone: "info" },
  { agency: "IMD Bengaluru", role: "Observations", detail: "Ground observations flowing every 15 min", status: "Data sharing · Live", tone: "positive" },
  { agency: "District Collectors · 3", role: "District command", detail: "Kodagu, Hassan, Mysuru", status: "Advisory issued · Pending confirmation", tone: "warning" },
] satisfies Array<{ agency: string; role: string; detail: string; status: string; tone: SemanticTone }>;

export const broadcastDraft = {
  version: 2,
  generated_by: "MausamSetu",
  message: "The Cauvery Basin is under an active monsoon surge. Kodagu district should prepare for flash-flood risk between Jul 18 and Jul 20 with peak intensity of 82 mm/hr expected on Jul 19. Follow local advisories from Karnataka DMA. Emergency line: 1077.",
  languages: ["English", "Kannada"],
  character_count: 287,
  character_limit: 320,
  sensitivity: "Public",
  drafted_at_ist: MOCK_IST_TIMESTAMP,
} as const;

export const distributionChannels = [
  { channel: "SMS", provider: "Kisan Suvidha", audience: "24,382 subscribers" },
  { channel: "WhatsApp", provider: "TN broadcast", audience: "8,712 subscribers" },
  { channel: "Twitter", provider: "@MausamSetu", audience: "public feed" },
  { channel: "IVR", provider: "Emergency line", audience: "district officers" },
  { channel: "Public advisory API", provider: "Partner network", audience: "12 partner apps" },
] as const;

export const alertDecisionLog = [
  { time_ist: "2026-07-15T08:12:00+05:30", authority: "TN Irrigation Dept", action: "Sluice opening planning", detail: "Initiated at Mettur", tone: "info" },
  { time_ist: "2026-07-15T08:47:00+05:30", authority: "Karnataka DMA", action: "Kodagu flash-flood", detail: "Watch issued", tone: "critical" },
  { time_ist: "2026-07-15T09:15:00+05:30", authority: "BWSSB Bengaluru", action: "Water distribution", detail: "Schedule normalized", tone: "positive" },
  { time_ist: "2026-07-15T09:22:00+05:30", authority: "Agriculture Dept KA", action: "Mandya paddy sowing", detail: "Advisory drafted", tone: "warning" },
] as const;

export const alertPipeline: readonly { label: string; status: PipelineStatus }[] = [
  { label: "INGEST", status: "completed" }, { label: "REGRID", status: "completed" },
  { label: "ASSIMILATE", status: "completed" }, { label: "FORECAST", status: "completed" },
  { label: "IMPACT", status: "completed" }, { label: "ALERT", status: "critical" },
];
