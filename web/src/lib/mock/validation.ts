import { MOCK_IST_TIMESTAMP, buildCauveryGrid, type PipelineStatus, type SemanticTone } from "./_shared";

interface MockMetricBundle {
  mae: number | null;
  rmse: number | null;
  bias: number | null;
  "pod@1mm"?: number | null;
  "far@1mm"?: number | null;
  "csi@1mm"?: number | null;
  "hss@1mm"?: number | null;
  "pod@10mm"?: number | null;
  "far@10mm"?: number | null;
  "csi@10mm"?: number | null;
}

export interface MockValidationMetrics {
  variable: string;
  unit: string;
  split: string;
  years: number[];
  lead_day: number | null;
  n_windows: number;
  ours: MockMetricBundle;
  persistence: MockMetricBundle;
  climatology: MockMetricBundle;
}

export const validationMetricsResponse: MockValidationMetrics = {
  variable: "rain",
  unit: "mm/day",
  split: "test",
  years: [2023],
  lead_day: null,
  n_windows: 353,
  ours: { mae: 5.12, rmse: 11.39, bias: 1.25, "pod@1mm": 0.62, "far@1mm": 0.64, "csi@1mm": 0.291, "hss@1mm": 0.222, "pod@10mm": 0.28, "far@10mm": 0.82, "csi@10mm": 0.125 },
  persistence: { mae: 5.21, rmse: 13.56, bias: 0.77, "pod@1mm": 0.4, "far@1mm": 0.63, "csi@1mm": 0.237, "hss@1mm": 0.185, "pod@10mm": 0.2, "far@10mm": 0.84, "csi@10mm": 0.1 },
  climatology: { mae: 5.04, rmse: 11.38, bias: 1.12, "pod@1mm": 0.6, "far@1mm": 0.64, "csi@1mm": 0.29, "hss@1mm": 0.222, "pod@10mm": 0.28, "far@10mm": 0.82, "csi@10mm": 0.124 },
};

export const validationDateMetricsResponse = {
  variable: "rain",
  unit: "mm/day",
  date_ist: MOCK_IST_TIMESTAMP,
  n_forecast_days: 7,
  ours: { mae: 4.8, rmse: 8.2, bias: -1.2, "pod@1mm": 0.82, "far@1mm": 0.22, "csi@1mm": 0.67, "pod@10mm": 0.71, "far@10mm": 0.31, "csi@10mm": 0.52 },
  persistence: { mae: 6.4, rmse: 11.5, bias: 1.8, "pod@1mm": 0.61, "far@1mm": 0.35, "csi@1mm": 0.44 },
  climatology: { mae: 5.9, rmse: 9.8, bias: 1.1, "pod@1mm": 0.67, "far@1mm": 0.3, "csi@1mm": 0.5 },
};

export const validationAuditSummary = {
  status: "Passing",
  audited_at_ist: MOCK_IST_TIMESTAMP,
  model: "ConvLSTM v2.3",
  framework: "PyTorch",
  training_window: "1991–2021",
  training_years: 30,
  training_samples: 32_850,
  holdout: "2022–2024",
  holdout_weighting: "JJAS-weighted",
  retrained: "2026-04-01",
  cadence: "Quarterly",
} as const;

export const observedPredictedPoints = Array.from({ length: 180 }, (_, index) => {
  const observed_mm_per_day = Number(Math.min(100, (index % 45) * 1.8 + Math.sin(index * 0.9) * 4 + 3).toFixed(1));
  const compression = observed_mm_per_day > 75 ? 0.82 : 0.96;
  return { observed_mm_per_day, predicted_mm_per_day: Number(Math.max(0, observed_mm_per_day * compression + Math.cos(index * 1.3) * 7).toFixed(1)) };
});

export const validationScatterStats = { r_squared: 0.74, bias_mm_per_day: -1.2, sample_pairs: 15_240 } as const;

export const thresholdSkill = [
  { threshold_mm: 1, pod: 0.82, far: 0.22, csi: 0.67 },
  { threshold_mm: 10, pod: 0.71, far: 0.31, csi: 0.52 },
  { threshold_mm: 25, pod: 0.58, far: 0.38, csi: 0.41 },
  { threshold_mm: 50, pod: 0.43, far: 0.45, csi: 0.3 },
] as const;

export const spatialError = {
  mean_bias_unit: "mm/day",
  mean_bias: buildCauveryGrid(0, 2.6, -5, 5),
  rmse_unit: "mm/day",
  rmse: buildCauveryGrid(7.5, 3.2, 0, 16),
} as const;

export const baselineComparison = [
  { model: "Persistence", rmse_mm_per_day: 11.5, status: "baseline", tone: "neutral" },
  { model: "Climatology", rmse_mm_per_day: 9.8, status: "baseline", tone: "neutral" },
  { model: "MausamSetu", rmse_mm_per_day: 8.2, status: "selected", tone: "info" },
] satisfies Array<{ model: string; rmse_mm_per_day: number; status: string; tone: SemanticTone }>;

export const droughtStressTests = [2002, 2009, 2015].map((year, index) => ({
  year,
  csi: [0.47, 0.44, 0.41][index],
  outcome: "Deficit detected",
  model_pct_of_normal: [96, 89, 82, 74, 69, 63, 58].map((value, point) => value - index * 2 + Math.sin(point) * 2),
  observed_pct_of_normal: [98, 91, 80, 72, 65, 60, 55].map((value) => value - index),
}));

export const validationDisclosures = [
  { title: "Known limitation", body: "Extreme events >75 mm/day show reduced CSI.", detail: "Mitigated by ensemble spread and explicit uncertainty bands on every forecast." },
  { title: "Training data window", body: "1991–2021 · 30 years · 32,850 daily samples.", detail: "Retrained quarterly. Last: April 2026. Next: July 2026." },
  { title: "What we don't yet model", body: "Aerosol-cloud interactions, urban heat island effects, and glacial melt contributions.", detail: "On roadmap for v3." },
] as const;

export const validationPipeline: readonly { label: string; status: PipelineStatus }[] = [
  { label: "INGEST", status: "completed" }, { label: "REGRID", status: "completed" },
  { label: "ASSIMILATE", status: "completed" }, { label: "FORECAST", status: "completed" },
  { label: "IMPACT", status: "completed" }, { label: "VALIDATE", status: "active" },
];
