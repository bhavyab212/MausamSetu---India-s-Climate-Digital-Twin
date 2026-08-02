export const MOCK_IST_TIMESTAMP = "2026-07-15T09:30:00+05:30";
export const MOCK_DATE = "2026-07-15";

export const CAUVERY_LAT = Array.from({ length: 19 }, (_, index) => 10 + index * 0.25);
export const CAUVERY_LON = Array.from({ length: 17 }, (_, index) => 75.5 + index * 0.25);
export const CAUVERY_BASIN_CELLS = 311;

export type NullableGrid = Array<Array<number | null>>;
export type SemanticTone = "info" | "positive" | "warning" | "critical" | "neutral";
export type PipelineStatus = "completed" | "active" | "pending" | "critical";

const outsideBasin = new Set([
  "0:0",
  "0:1",
  "0:15",
  "0:16",
  "1:0",
  "1:16",
  "17:0",
  "17:16",
  "18:0",
  "18:1",
  "18:15",
  "18:16",
]);

export function buildCauveryGrid(
  base: number,
  variation: number,
  minimum = 0,
  maximum = Number.POSITIVE_INFINITY,
): NullableGrid {
  return CAUVERY_LAT.map((_, row) =>
    CAUVERY_LON.map((_, column) => {
      if (outsideBasin.has(`${row}:${column}`)) return null;
      const wave = Math.sin((row + 1) * 0.63) + Math.cos((column + 2) * 0.47);
      const ridge = Math.max(0, 1 - Math.abs(column - 4) / 6) * 0.65;
      return Number(Math.min(maximum, Math.max(minimum, base + wave * variation + ridge * variation)).toFixed(2));
    }),
  );
}

export function buildForecastCube(
  dailyBases: readonly number[],
  variation: number,
  minimum: number,
  maximum: number,
): NullableGrid[] {
  return dailyBases.map((base, day) =>
    buildCauveryGrid(base, variation + day * 0.03, minimum, maximum),
  );
}

export function addDays(date: string, dayOffset: number): string {
  const value = new Date(`${date}T00:00:00Z`);
  value.setUTCDate(value.getUTCDate() + dayOffset);
  return value.toISOString().slice(0, 10);
}

export function sequence(length: number, start: number, step: number): number[] {
  return Array.from({ length }, (_, index) => Number((start + index * step).toFixed(2)));
}
