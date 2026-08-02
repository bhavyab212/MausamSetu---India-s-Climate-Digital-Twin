import { MOCK_IST_TIMESTAMP, addDays, type PipelineStatus, type SemanticTone } from "./_shared";

const sectorDates = Array.from({ length: 30 }, (_, day) => addDays("2026-07-15", day));

export const sectorHydrologyResponse = {
  start_date_ist: MOCK_IST_TIMESTAMP,
  dates: sectorDates,
  inflow_m3_per_day: sectorDates.map((_, day) => Math.round(25_000_000 + Math.sin(day / 4) * 5_000_000 + day * 180_000)),
  total_m3: 813_300_000,
};

export const sectorHeatResponse = {
  start_date_ist: MOCK_IST_TIMESTAMP,
  dates: sectorDates,
  heat_stress_days: 2,
  extreme_heat_days: 0,
  heat_stress_threshold_c: 40,
  extreme_heat_threshold_c: 45,
};

export const sectorRupeeResponse = {
  start_date_ist: MOCK_IST_TIMESTAMP,
  dates: sectorDates,
  total_crore: 38.4,
  rainfall_component_crore: 25.6,
  heat_component_crore: 12.8,
  unit: "₹ crore",
};

export const sectorTabs: readonly { id: string; label: string; summary: string; tone: SemanticTone; selected: boolean }[] = [
  { id: "water", label: "Water security", summary: "Reservoirs +18% inflow", tone: "positive", selected: true },
  { id: "agriculture", label: "Agriculture", summary: "Soil moisture adequate", tone: "positive", selected: false },
  { id: "heat", label: "Heat stress", summary: "2 days >38°C forecast", tone: "warning", selected: false },
  { id: "flood", label: "Flood risk", summary: "Kodagu, Hassan zones", tone: "critical", selected: false },
];

export const reservoirNetwork = {
  updated_ist: MOCK_IST_TIMESTAMP,
  unit: "MCft",
  discharge_m3_per_s: 340,
  reservoirs: [
    { id: "krs", name: "KRS", state: "Karnataka", storage_pct: 79, storage_mcft: 91, status: "Nominal", tone: "positive" },
    { id: "kabini", name: "Kabini", state: "Karnataka", storage_pct: 72, storage_mcft: 68, status: "Stable", tone: "positive" },
    { id: "mettur", name: "Mettur", state: "Tamil Nadu", storage_pct: 87, storage_mcft: 142, status: "Rising · +18%", tone: "info" },
  ] satisfies Array<{ id: string; name: string; state: string; storage_pct: number; storage_mcft: number; status: string; tone: SemanticTone }>,
  links: [{ from: "krs", to: "mettur" }, { from: "kabini", to: "mettur" }],
  destination: { name: "Cauvery Delta", description: "Agricultural command area", area_hectares: 800_000 },
} as const;

export const supplyDemandSeries = sectorDates.map((date, day) => ({
  date,
  supply_mcft_per_day: Number((148 - day * 1.4 + Math.sin(day / 3) * 14).toFixed(1)),
  demand_mcft_per_day: Number((102 + day * 0.8 + Math.cos(day / 4) * 10).toFixed(1)),
}));

export const waterBalance = {
  evaluated_at_ist: MOCK_IST_TIMESTAMP,
  inputs: [
    { label: "Rainfall", value: 32.4, unit: "mm/day", share_pct: 80 },
    { label: "Groundwater", value: 8, unit: "mm/day", share_pct: 20 },
    { label: "Transfer", value: 0, unit: "mm/day", share_pct: 0 },
  ],
  stocks: [
    { label: "Basin storage", value: 87, unit: "%" },
    { label: "Soil moisture", value: 74, unit: "%" },
  ],
  outputs: [
    { label: "Irrigation", share_pct: 45 }, { label: "Evapotranspiration", share_pct: 30 },
    { label: "Domestic", share_pct: 15 }, { label: "Environmental flow", share_pct: 10 },
  ],
} as const;

export const stakeholderAdvisories = [
  { id: "tn-irrigation", sector: "Irrigation", region: "Tamil Nadu", category: "PRIORITY", recommendation: "Open Mettur sluice gates by Jul 20 to accommodate projected +40% inflow. Recommended discharge: 12,000 cusecs.", recipient: "TN Public Works Dept", status: "Acknowledged", response_time: "3h 12m", tone: "info" },
  { id: "bengaluru-water", sector: "Urban water", region: "Bengaluru", category: "INFO", recommendation: "Reservoir buffer stable. Current supply covers 15 days at normal demand. No consumer action required.", recipient: "BWSSB", status: "Acknowledged", response_time: "47m", tone: "positive" },
  { id: "mandya-agriculture", sector: "Agriculture", region: "Mandya district", category: "OPPORTUNITY", recommendation: "Soil moisture optimal for kharif paddy sowing. Farmer advisory window: Jul 16–22.", recipient: "KAD + AgriDept KA", status: "Pending", response_time: "—", tone: "warning" },
] satisfies Array<{ id: string; sector: string; region: string; category: string; recommendation: string; recipient: string; status: string; response_time: string; tone: SemanticTone }>;

export const sectorDecisionLog = [
  { time_ist: "2026-07-15T08:12:00+05:30", authority: "TN Irrigation Dept", action: "Sluice opening planning", detail: "Initiated at Mettur", tone: "info" },
  { time_ist: "2026-07-15T08:47:00+05:30", authority: "Karnataka DMA", action: "Kodagu flash-flood", detail: "Watch issued", tone: "critical" },
  { time_ist: "2026-07-15T09:15:00+05:30", authority: "BWSSB Bengaluru", action: "Water distribution", detail: "Schedule normalized", tone: "positive" },
  { time_ist: "2026-07-15T09:22:00+05:30", authority: "Agriculture Dept KA", action: "Mandya paddy sowing", detail: "Advisory drafted", tone: "warning" },
] as const;

export const sectorPipeline: readonly { label: string; status: PipelineStatus }[] = [
  { label: "INGEST", status: "completed" }, { label: "REGRID", status: "completed" },
  { label: "ASSIMILATE", status: "completed" }, { label: "FORECAST", status: "completed" },
  { label: "IMPACT", status: "active" }, { label: "VALIDATE", status: "pending" },
];
