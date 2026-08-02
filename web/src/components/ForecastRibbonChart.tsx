import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export interface ForecastRibbonPoint {
  date: string;
  p10: number;
  p50: number;
  p90: number;
  climatology: number;
}

export interface ForecastRibbonChartProps extends HTMLAttributes<HTMLDivElement> {
  points: readonly ForecastRibbonPoint[];
  unit: string;
  peakLabel?: string;
}

const width = 520;
const height = 180;
const margin = { top: 22, right: 16, bottom: 30, left: 36 };
const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;

function x(index: number, count: number) {
  return margin.left + (index / Math.max(1, count - 1)) * plotWidth;
}

function y(value: number, max = 60) {
  return margin.top + plotHeight - (value / max) * plotHeight;
}

function linePath(points: readonly ForecastRibbonPoint[], field: "p50" | "climatology") {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(index, points.length).toFixed(1)} ${y(point[field]).toFixed(1)}`).join(" ");
}

export function ForecastRibbonChart({ points, unit, peakLabel, className, ...props }: ForecastRibbonChartProps) {
  const upper = points.map((point, index) => `${x(index, points.length).toFixed(1)},${y(point.p90).toFixed(1)}`);
  const lower = [...points].reverse().map((point, reverseIndex) => {
    const index = points.length - reverseIndex - 1;
    return `${x(index, points.length).toFixed(1)},${y(point.p10).toFixed(1)}`;
  });
  const peakIndex = points.reduce((best, point, index) => point.p50 > points[best].p50 ? index : best, 0);

  return (
    <div className={cn("w-full", className)} {...props}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Seven-day rainfall forecast in ${unit}`} className="h-full w-full">
        <title>Seven-day rainfall forecast with p10 to p90 uncertainty</title>
        {[0, 15, 30, 45, 60].map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} className="stroke-divider" strokeWidth="1" />
            <text x={margin.left - 8} y={y(tick) + 4} textAnchor="end" className="fill-textTertiary text-caption">{tick}</text>
          </g>
        ))}
        <text x={margin.left - 24} y={12} className="fill-textSecondary text-caption">{unit}</text>
        <polygon points={[...upper, ...lower].join(" ")} className="fill-primary-soft" opacity="0.9" />
        <path d={linePath(points, "climatology")} fill="none" className="stroke-textTertiary" strokeDasharray="4 4" strokeWidth="1.5" />
        <path d={linePath(points, "p50")} fill="none" className="stroke-primary" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((point, index) => (
          <g key={point.date}>
            <circle cx={x(index, points.length)} cy={y(point.p50)} r="3.5" className="fill-primary stroke-card-solid" strokeWidth="2" />
            <text x={x(index, points.length)} y={height - 9} textAnchor="middle" className="fill-textSecondary text-caption">{new Date(`${point.date}T00:00:00Z`).getUTCDate()}</text>
          </g>
        ))}
        <text x={margin.left} y={height - 9} textAnchor="start" className="fill-textSecondary text-caption">Jul</text>
        <text x={width - margin.right} y={y(points[0]?.climatology ?? 18) - 6} textAnchor="end" className="fill-textTertiary text-caption">Climatology (18 mm)</text>
        {peakLabel ? (
          <g transform={`translate(${Math.min(width - 150, Math.max(70, x(peakIndex, points.length) - 65))} ${Math.max(2, y(points[peakIndex].p90) - 26)})`}>
            <rect width="132" height="22" rx="6" className="fill-card-solid stroke-cardBorder" />
            <text x="66" y="15" textAnchor="middle" className="fill-textPrimary text-caption">{peakLabel}</text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}
