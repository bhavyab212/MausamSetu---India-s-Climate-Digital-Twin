import {
  CAUVERY_LAT,
  CAUVERY_LON,
  MOCK_DATE,
  MOCK_IST_TIMESTAMP,
  addDays,
  buildForecastCube,
  type NullableGrid,
} from "./_shared";

export interface MockForecastBlock {
  unit: string;
  p10: NullableGrid[];
  p50: NullableGrid[];
  p90: NullableGrid[];
}

export interface MockForecast {
  forecast_start_ist: string;
  horizon_days: number;
  lat: number[];
  lon: number[];
  dates: string[];
  rain: MockForecastBlock;
  tmax: MockForecastBlock;
  tmin: MockForecastBlock;
}

const dates = Array.from({ length: 7 }, (_, day) => addDays(MOCK_DATE, day + 1));
const rainP50 = [32, 34, 44, 48, 41, 28, 22];
const tmaxP50 = [30.1, 30.8, 31.6, 32.4, 33.1, 32.6, 31.9];
const tminP50 = [22.1, 22.4, 22.8, 23.2, 23.7, 23.4, 22.9];

export const forecastResponse: MockForecast = {
  forecast_start_ist: MOCK_IST_TIMESTAMP,
  horizon_days: 7,
  lat: CAUVERY_LAT,
  lon: CAUVERY_LON,
  dates,
  rain: {
    unit: "mm/day",
    p10: buildForecastCube(rainP50.map((value) => value * 0.68), 5, 0, 100),
    p50: buildForecastCube(rainP50, 7, 0, 100),
    p90: buildForecastCube(rainP50.map((value) => value * 1.28), 9, 0, 120),
  },
  tmax: {
    unit: "degC",
    p10: buildForecastCube(tmaxP50.map((value) => value - 1.4), 0.8, 24, 38),
    p50: buildForecastCube(tmaxP50, 1, 24, 38),
    p90: buildForecastCube(tmaxP50.map((value) => value + 1.5), 1.1, 24, 38),
  },
  tmin: {
    unit: "degC",
    p10: buildForecastCube(tminP50.map((value) => value - 1.1), 0.6, 18, 30),
    p50: buildForecastCube(tminP50, 0.7, 18, 30),
    p90: buildForecastCube(tminP50.map((value) => value + 1.2), 0.8, 18, 30),
  },
};

export const forecastSummary = {
  horizon_days: 7,
  ensemble_members: 32,
  peak_intensity_mm: 48,
  peak_date: "2026-07-19",
  confidence: "High",
  model: "ConvLSTM v2.3",
  updated_ist: MOCK_IST_TIMESTAMP,
} as const;

export const basinForecastSeries = [
  { date: "2026-07-08", observed_mm: 12, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-09", observed_mm: 16, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-10", observed_mm: 14, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-11", observed_mm: 22, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-12", observed_mm: 27, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-13", observed_mm: 25, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-14", observed_mm: 30, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  { date: "2026-07-15", observed_mm: 32.4, p10_mm: null, p50_mm: null, p90_mm: null, climatology_mm: 18, extreme_threshold_mm: 40 },
  ...dates.map((date, index) => ({ date, observed_mm: null, p10_mm: Number((rainP50[index] * 0.68).toFixed(1)), p50_mm: rainP50[index], p90_mm: Number((rainP50[index] * 1.28).toFixed(1)), climatology_mm: 18, extreme_threshold_mm: 40 })),
];

export const spatialForecastFrames = dates.map((date, index) => ({
  id: `day-${index + 1}`,
  date,
  basin_mean_rainfall_mm: rainP50[index],
  values_mm_per_day: forecastResponse.rain.p50[index],
  selected: index === 0,
}));

export const ensembleMembers = Array.from({ length: 32 }, (_, member) => ({
  id: `member-${String(member + 1).padStart(2, "0")}`,
  rainfall_mm: rainP50.map((value, day) => Number(Math.max(0, value + Math.sin((member + 1) * 0.73 + day) * (4 + member / 5)).toFixed(1))),
}));

export const daySevenDistribution = [
  { range_mm: "0–10", members: 2 }, { range_mm: "10–20", members: 8 },
  { range_mm: "20–30", members: 13 }, { range_mm: "30–40", members: 6 },
  { range_mm: "40–50", members: 3 },
] as const;

export const monsoonPulse = {
  score: 78,
  score_max: 100,
  onset_date: "2026-06-03",
  season_to_date_mm: 412,
  percent_of_normal: 108,
  phase: "Active surge phase",
} as const;

export const forecastAttribution = [
  { label: "Yesterday's rainfall @ Coorg", contribution_pct: 78 },
  { label: "Arabian Sea SST +1.2°C", contribution_pct: 62 },
  { label: "Indian Ocean Dipole index", contribution_pct: 45 },
  { label: "Soil moisture · Kabini basin", contribution_pct: 41 },
  { label: "500 hPa geopotential · Deccan", contribution_pct: 33 },
] as const;

export const forecastSkill = [
  { label: "RMSE (7d)", value: 8.2, unit: "mm" }, { label: "MAE (7d)", value: 5.3, unit: "mm" },
  { label: "ACC", value: 0.68, unit: "score" }, { label: "CSI (>10mm)", value: 0.52, unit: "score" },
  { label: "POD", value: 0.71, unit: "score" }, { label: "FSS", value: 0.61, unit: "score" },
] as const;
