import type { SVGAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import type { SemanticTone } from "@/components/InlineLegend";
import { cn } from "@/lib/utils";

const donutGaugeVariants = cva("h-24 w-24", {
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

export interface DonutGaugeProps
  extends Omit<SVGAttributes<SVGSVGElement>, "children" | "color" | "value">,
    VariantProps<typeof donutGaugeVariants> {
  value: number;
  displayValue?: string;
  ariaLabel: string;
}

export function DonutGauge({
  value,
  displayValue,
  ariaLabel,
  tone,
  disabled = false,
  className,
  ...props
}: DonutGaugeProps) {
  const normalizedValue = Number.isFinite(value)
    ? Math.min(100, Math.max(0, value))
    : 0;
  const centerValue = displayValue ?? `${Math.round(normalizedValue)}%`;

  return (
    <svg
      viewBox="0 0 100 100"
      role="img"
      aria-label={ariaLabel}
      className={cn(donutGaugeVariants({ tone, disabled }), className)}
      {...props}
    >
      <title>{ariaLabel}</title>
      <circle
        cx="50"
        cy="50"
        r="42"
        fill="none"
        stroke="currentColor"
        strokeWidth="8"
        className="text-missing"
      />
      <circle
        cx="50"
        cy="50"
        r="42"
        pathLength="100"
        fill="none"
        stroke="currentColor"
        strokeWidth="8"
        strokeLinecap="round"
        strokeDasharray={`${normalizedValue} ${100 - normalizedValue}`}
        transform="rotate(-90 50 50)"
      />
      <text
        x="50"
        y="50"
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-current font-mono text-title tabular-nums"
      >
        {centerValue}
      </text>
    </svg>
  );
}

export { donutGaugeVariants };
