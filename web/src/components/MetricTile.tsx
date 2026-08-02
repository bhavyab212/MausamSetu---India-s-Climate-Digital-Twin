import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const metricTileVariants = cva(
  "flex min-h-32 rounded-card border border-cardBorder p-card-lg shadow-elevation-card transition-opacity duration-fast",
  {
    variants: {
      variant: {
        default: "flex-col justify-between bg-card",
        "with-gauge": "flex-row items-center justify-between gap-card-lg bg-card-solid",
        "with-delta": "flex-col justify-between border-l-4 bg-card-muted",
      },
      disabled: {
        true: "cursor-not-allowed opacity-50",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      disabled: false,
    },
  },
);

const deltaVariants = cva("text-caption font-semibold tabular-nums", {
  variants: {
    tone: {
      positive: "text-positive",
      warning: "text-warning",
      critical: "text-critical",
      info: "text-primary",
      neutral: "text-textSecondary",
    },
  },
  defaultVariants: {
    tone: "neutral",
  },
});

export interface MetricTileProps
  extends Omit<HTMLAttributes<HTMLElement>, "children">,
    VariantProps<typeof metricTileVariants> {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  delta?: ReactNode;
  deltaTone?: VariantProps<typeof deltaVariants>["tone"];
  visual?: ReactNode;
  loading?: boolean;
}

export function MetricTile({
  label,
  value,
  unit,
  delta,
  deltaTone,
  visual,
  variant,
  disabled = false,
  loading = false,
  className,
  ...props
}: MetricTileProps) {
  return (
    <article
      className={cn(metricTileVariants({ variant, disabled }), className)}
      aria-busy={loading || undefined}
      {...props}
    >
      <div className="flex min-w-0 flex-1 flex-col justify-between gap-row-gap">
        <p className="text-label text-textSecondary">{label}</p>
        {loading ? (
          <div className="space-y-2" aria-hidden="true">
            <div className="h-8 w-3/4 animate-pulse rounded-control bg-missing" />
            <div className="h-3 w-1/2 animate-pulse rounded-control bg-missing" />
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="font-mono text-hero text-textPrimary tabular-nums">
                {value}
              </span>
              {unit ? (
                <span className="text-label text-textSecondary">{unit}</span>
              ) : null}
            </div>
            {delta ? (
              <p className={deltaVariants({ tone: deltaTone })}>{delta}</p>
            ) : null}
          </div>
        )}
      </div>
      {visual ? (
        <div className={cn("shrink-0", loading && "opacity-50")} aria-hidden={loading || undefined}>
          {visual}
        </div>
      ) : null}
    </article>
  );
}

export { metricTileVariants };
