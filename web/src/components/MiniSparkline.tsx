import type { SVGAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import type { SemanticTone } from "@/components/InlineLegend";
import { cn } from "@/lib/utils";

export type SparklinePoint = number | { x: number; y: number };

const sparklineVariants = cva("h-8 w-full", {
  variants: {
    tone: {
      positive: "text-positive",
      warning: "text-warning",
      critical: "text-critical",
      info: "text-primary",
      neutral: "text-textSecondary",
    } satisfies Record<SemanticTone, string>,
    disabled: {
      true: "opacity-50 grayscale",
      false: "",
    },
  },
  defaultVariants: {
    tone: "info",
    disabled: false,
  },
});

function createPath(points: readonly SparklinePoint[]): string {
  const finitePoints = points
    .map((point, index) =>
      typeof point === "number"
        ? { x: index, y: point }
        : { x: point.x, y: point.y },
    )
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (finitePoints.length === 0) return "";

  const xValues = finitePoints.map((point) => point.x);
  const yValues = finitePoints.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const xRange = maxX - minX;
  const yRange = maxY - minY;

  return finitePoints
    .map((point, index) => {
      const x = xRange === 0 ? 50 : ((point.x - minX) / xRange) * 100;
      const y = yRange === 0 ? 16 : 30 - ((point.y - minY) / yRange) * 28;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export interface MiniSparklineProps
  extends Omit<
      SVGAttributes<SVGSVGElement>,
      "children" | "color" | "points"
    >,
    VariantProps<typeof sparklineVariants> {
  ariaLabel: string;
  pathData?: string;
  points?: readonly SparklinePoint[];
}

export function MiniSparkline({
  ariaLabel,
  pathData,
  points = [],
  tone,
  disabled = false,
  className,
  ...props
}: MiniSparklineProps) {
  const path = pathData ?? createPath(points);

  return (
    <svg
      viewBox="0 0 100 32"
      role="img"
      aria-label={ariaLabel}
      className={cn(sparklineVariants({ tone, disabled }), className)}
      preserveAspectRatio="none"
      {...props}
    >
      <title>{ariaLabel}</title>
      {path ? (
        <path
          d={path}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
    </svg>
  );
}

export { createPath, sparklineVariants };
