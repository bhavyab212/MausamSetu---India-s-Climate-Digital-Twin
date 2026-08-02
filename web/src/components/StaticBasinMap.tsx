import { LocateFixed, Minus, Plus } from "lucide-react";

import { ScaleLegend } from "@/components/ScaleLegend";
import { cn } from "@/lib/utils";

export interface BasinMapReservoir {
  id: string;
  name: string;
  storage_pct: number;
  lat_deg_north: number;
  lon_deg_east: number;
}

export interface StaticBasinMapProps {
  values: readonly (readonly (number | null)[])[];
  reservoirs?: readonly BasinMapReservoir[];
  className?: string;
  showControls?: boolean;
  compact?: boolean;
}

const MAP_WIDTH = 720;
const MAP_HEIGHT = 360;
const MIN_LON = 74;
const MAX_LON = 81;
const MIN_LAT = 8.5;
const MAX_LAT = 16.5;
const GRID_MIN_LON = 75.5;
const GRID_MAX_LON = 79.5;
const GRID_MIN_LAT = 10;
const GRID_MAX_LAT = 14.5;

function project(lon: number, lat: number) {
  return {
    x: ((lon - MIN_LON) / (MAX_LON - MIN_LON)) * MAP_WIDTH,
    y: ((MAX_LAT - lat) / (MAX_LAT - MIN_LAT)) * MAP_HEIGHT,
  };
}

function rainClass(value: number) {
  if (value >= 65) return "fill-rain-100";
  if (value >= 45) return "fill-rain-75";
  if (value >= 25) return "fill-rain-50";
  if (value >= 10) return "fill-rain-25";
  return "fill-rain-0";
}



function markerPosition(lat: number, lon: number) {
  return project(lon, lat);
}

export function StaticBasinMap({ values, reservoirs = [], className, showControls = true, compact = false }: StaticBasinMapProps) {
  const gridWidth = ((GRID_MAX_LON - GRID_MIN_LON) / (values[0]?.length || 17)) * (MAP_WIDTH / (MAX_LON - MIN_LON));
  const gridHeight = ((GRID_MAX_LAT - GRID_MIN_LAT) / (values.length || 19)) * (MAP_HEIGHT / (MAX_LAT - MIN_LAT));

  return (
    <div className={cn("relative overflow-hidden bg-[#eaf3f9]", className)}>
      <div
        role="img"
        aria-label="South India state boundary base map"
        className="absolute inset-0 h-full w-full bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: "url('/assets/map/cauvery-basemap.webp')" }}
      />
      <svg viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} role="img" aria-label="Cauvery Basin rainfall grid with reservoir markers" className="relative h-full w-full" preserveAspectRatio="none">
        <title>Cauvery Basin rainfall grid with reservoir markers</title>
        <g opacity="0.78">
          {values.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
            if (value === null) return null;
            const lon = GRID_MIN_LON + columnIndex * ((GRID_MAX_LON - GRID_MIN_LON) / 16);
            const lat = GRID_MIN_LAT + rowIndex * ((GRID_MAX_LAT - GRID_MIN_LAT) / 18);
            const topLeft = project(lon - 0.125, lat + 0.125);
            return (
              <rect
                key={`${rowIndex}-${columnIndex}`}
                x={topLeft.x}
                y={topLeft.y}
                width={gridWidth}
                height={gridHeight}
                className={rainClass(value)}
                opacity={0.42 + Math.min(value, 80) / 145}
              />
            );
          }))}
        </g>
        <path
          d="M 112 72 C 144 54 174 59 195 84 C 217 110 235 143 260 158 C 284 172 302 188 319 211 C 339 238 371 246 395 268 C 414 286 426 306 412 322 C 389 338 354 335 327 321 C 301 307 276 298 250 293 C 220 288 195 272 178 251 C 163 232 158 207 146 188 C 134 168 112 151 102 129 C 92 106 94 84 112 72 Z"
          fill="none"
          className="stroke-positive"
          strokeWidth="2.4"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <g fill="none" className="stroke-primary" strokeWidth="1.4" opacity="0.78">
          <path d="M120 214 C188 181 244 208 302 238 S422 300 518 318" />
          <path d="M190 135 C237 164 286 190 341 218" />
          <path d="M352 96 C402 130 440 165 483 189" />
        </g>
        <text x="285" y="260" className="fill-positive text-title">Cauvery Basin</text>
        {reservoirs.map((reservoir) => {
          const position = markerPosition(reservoir.lat_deg_north, reservoir.lon_deg_east);
          return (
            <g key={reservoir.id} transform={`translate(${position.x} ${position.y})`}>
              <circle r="8" className="fill-card-solid stroke-positive" strokeWidth="3" />
              {!compact ? <><text x="14" y="-2" className="fill-textPrimary text-caption">{reservoir.name}</text><text x="14" y="12" className="fill-textSecondary text-caption">{reservoir.storage_pct}%</text></> : null}
            </g>
          );
        })}
      </svg>
      {!compact ? <ScaleLegend label="Rainfall (mm)" ticks={["0", "25", "50", "75", "100+"]} className="absolute bottom-3 left-3" /> : null}
      {showControls && !compact ? (
        <div className="absolute right-3 top-1/2 flex -translate-y-1/2 flex-col gap-2">
          {[{ label: "Zoom in", icon: Plus }, { label: "Zoom out", icon: Minus }, { label: "Locate basin", icon: LocateFixed }].map(({ label, icon: Icon }) => (
            <button key={label} type="button" aria-label={label} className="flex h-control w-control items-center justify-center rounded-control border border-cardBorder bg-card shadow-elevation-card text-textSecondary hover:text-primary"><Icon className="h-4 w-4" aria-hidden="true" /></button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
