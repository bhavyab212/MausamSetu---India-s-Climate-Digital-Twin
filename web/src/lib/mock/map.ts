import {
  CAUVERY_BASIN_CELLS,
  CAUVERY_LAT,
  CAUVERY_LON,
  MOCK_DATE,
  MOCK_IST_TIMESTAMP,
  addDays,
  buildCauveryGrid,
  type NullableGrid,
  type SemanticTone,
} from "./_shared";

interface MockCompleteness { valid_cells: number; basin_cells: number; completeness_pct: number }
interface MockGrid { date_ist: string; variable: string; unit: string; lat: number[]; lon: number[]; values: NullableGrid; completeness: MockCompleteness }

export const mapRainfallGrid: MockGrid = {
  date_ist: MOCK_IST_TIMESTAMP,
  variable: "rain",
  unit: "mm/day",
  lat: CAUVERY_LAT,
  lon: CAUVERY_LON,
  values: buildCauveryGrid(28, 14, 0, 96),
  completeness: { valid_cells: 311, basin_cells: CAUVERY_BASIN_CELLS, completeness_pct: 100 },
};

export const mapAssimilation = {
  date_ist: MOCK_IST_TIMESTAMP,
  variable: "rain",
  unit: "mm/day",
  lat: CAUVERY_LAT,
  lon: CAUVERY_LON,
  observed: buildCauveryGrid(28, 14, 0, 96),
  model_mean: buildCauveryGrid(24, 11, 0, 90),
  corrected: buildCauveryGrid(27, 13, 0, 94),
  rmse_before: 9.4,
  rmse_after: 6.1,
  reduction_pct: 35.1,
};

const historyDates = Array.from({ length: 14 }, (_, index) => addDays(MOCK_DATE, index - 13));
export const selectedCellTimeseries = {
  lat_deg_north: 12.25,
  lon_deg_east: 76.5,
  dates: historyDates,
  rain_mm_per_day: [4, 7, 2, 0, 8, 12, 18, 15, 21, 27, 19, 25, 29, 32.4],
  tmax_c: [31.2, 30.8, 32.1, 33.4, 31.8, 30.4, 29.8, 29.2, 28.9, 28.5, 29.1, 28.8, 28.4, 28.1],
  tmin_c: [22.8, 22.4, 23.1, 23.7, 23.2, 22.8, 22.4, 22.1, 21.9, 21.7, 21.8, 21.5, 21.4, 21.3],
  insat_lst_c: [33.2, 32.8, 34.1, 35.4, 33.8, 32.4, 31.8, 31.2, 30.9, 30.5, 30.1, 29.9, 29.8, 29.8],
  insat_rain_mm_per_day: [3.5, 6.8, 1.8, 0, 7.4, 11.6, 17.2, 14.4, 20.1, 25.8, 18.6, 24.2, 28.4, 31.7],
};

export const mapLayers = [
  { id: "rain", label: "Rainfall", enabled: true, selected: true },
  { id: "wind", label: "Wind flow", enabled: true, selected: false },
  { id: "tmax", label: "Max temperature", enabled: false, selected: false },
  { id: "tmin", label: "Min temperature", enabled: false, selected: false },
  { id: "lst", label: "INSAT LST", enabled: false, selected: false },
  { id: "reservoirs", label: "Reservoirs", enabled: true, selected: false },
  { id: "rivers", label: "Rivers", enabled: true, selected: false },
  { id: "soil", label: "Soil moisture", enabled: false, selected: false },
  { id: "flood", label: "Flood risk zones", enabled: false, selected: false },
] as const;

export const selectedMapCell = {
  place: "Mysuru",
  kind: "Grid cell",
  lat_deg_north: 12.25,
  lon_deg_east: 76.5,
  metrics: [
    { label: "Rainfall", value: 32.4, unit: "mm/day", tone: "info" },
    { label: "Tmax", value: 28.1, unit: "°C", tone: "warning" },
    { label: "Tmin", value: 21.3, unit: "°C", tone: "info" },
    { label: "INSAT LST", value: 29.8, unit: "°C", tone: "warning" },
  ] satisfies Array<{ label: string; value: number; unit: string; tone: SemanticTone }>,
  anomaly_mm: 14.2,
  anomaly_pct: 78,
  status: "Monsoon surge active",
} as const;

export const mapReservoirs = [
  { id: "kabini", name: "Kabini", storage_pct: 72, lat_deg_north: 11.95, lon_deg_east: 76.35 },
  { id: "krs", name: "KRS", storage_pct: 79, lat_deg_north: 12.42, lon_deg_east: 76.57 },
  { id: "mettur", name: "Mettur", storage_pct: 87, lat_deg_north: 11.8, lon_deg_east: 77.8 },
] as const;

export const monsoonDepression = {
  id: "L-14",
  pressure_hpa: 998,
  label: "monsoon depression",
  center: { lat_deg_north: 13.1, lon_deg_east: 82.2 },
  track: [
    { time_ist: "2026-07-15T03:30:00+05:30", lat_deg_north: 13.4, lon_deg_east: 83.4 },
    { time_ist: "2026-07-15T06:30:00+05:30", lat_deg_north: 13.3, lon_deg_east: 82.9 },
    { time_ist: MOCK_IST_TIMESTAMP, lat_deg_north: 13.1, lon_deg_east: 82.2 },
  ],
} as const;

export const mapTimeline = Array.from({ length: 15 }, (_, index) => {
  const date = addDays("2026-07-08", index);
  return { date, kind: index <= 7 ? "observed" : "forecast", selected: date === MOCK_DATE } as const;
});

export const mapLiveMetrics = [
  { label: "Rainfall", value: 32.4, unit: "mm/day", series: [14, 18, 16, 23, 27, 29, 32.4] },
  { label: "Wind", value: 18, unit: "km/h", series: [11, 13, 12, 16, 15, 17, 18] },
  { label: "Humidity", value: 86, unit: "%", series: [72, 74, 77, 79, 82, 84, 86] },
] as const;

export const assimilationStatus = {
  cycle: 46,
  status: "Complete",
  next_cycle_seconds: 1548,
  completed_at_ist: MOCK_IST_TIMESTAMP,
} as const;
