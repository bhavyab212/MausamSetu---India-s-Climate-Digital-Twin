import type { HTMLAttributes } from "react";
import { cva } from "class-variance-authority";

import type { SemanticTone } from "@/components/InlineLegend";
import { cn } from "@/lib/utils";

export interface MiniBarChartItem {
  id: string;
  label: string;
  value: number;
  tone?: SemanticTone;
}

const barVariants = cva("fill-current", {
  variants: {
    tone: {
      positive: "text-positive",
      warning: "text-warning",
      critical: "text-critical",
      info: "text-primary",
      neutral: "text-textTertiary",
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

export interface MiniBarChartProps
  extends Omit<HTMLAttributes<HTMLElement>, "children"> {
  bars: readonly MiniBarChartItem[];
  ariaLabel: string;
  disabled?: boolean;
}

export function MiniBarChart({
  bars,
  ariaLabel,
  disabled = false,
  className,
  ...props
}: MiniBarChartProps) {
  const finiteValues = bars.map((bar) =>
    Number.isFinite(bar.value) ? Math.max(0, bar.value) : 0,
  );
  const maximum = Math.max(...finiteValues, 0);
  const slotWidth = bars.length > 0 ? 120 / bars.length : 120;
  const barWidth = slotWidth * 0.6;

  return (
    <figure
      className={cn("space-y-2", disabled && "opacity-50", className)}
      aria-disabled={disabled || undefined}
      {...props}
    >
      <svg
        viewBox="0 0 120 36"
        role="img"
        aria-label={ariaLabel}
        className="h-16 w-full"
        preserveAspectRatio="none"
      >
        <title>{ariaLabel}</title>
        {bars.map((bar, index) => {
          const value = finiteValues[index] ?? 0;
          const height = maximum === 0 ? 0 : (value / maximum) * 32;
          const x = index * slotWidth + (slotWidth - barWidth) / 2;

          return (
            <rect
              key={bar.id}
              x={x}
              y={34 - height}
              width={barWidth}
              height={height}
              rx="2"
              className={barVariants({
                tone: bar.tone ?? "info",
                disabled,
              })}
            />
          );
        })}
      </svg>
      {bars.length > 0 ? (
        <figcaption className="flex" aria-hidden="true">
          {bars.map((bar) => (
            <span
              key={bar.id}
              className="min-w-0 flex-1 truncate text-center text-caption text-textSecondary"
            >
              {bar.label}
            </span>
          ))}
        </figcaption>
      ) : null}
      <span className="sr-only">
        {bars.map((bar) => `${bar.label}: ${bar.value}`).join(", ")}
      </span>
    </figure>
  );
}

export { barVariants };
