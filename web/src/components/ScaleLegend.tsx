import type { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const scaleLegendVariants = cva("h-2 w-full rounded-pill", {
  variants: {
    palette: {
      rainfall: "bg-gradient-to-r from-rain-0 via-rain-50 to-rain-100",
      anomaly: "bg-gradient-to-r from-anomaly-negative via-anomaly-neutral to-anomaly-positive",
      temperature: "bg-gradient-to-r from-temperature-low to-temperature-high",
    },
  },
  defaultVariants: { palette: "rainfall" },
});

export interface ScaleLegendProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "title">,
    VariantProps<typeof scaleLegendVariants> {
  label: string;
  ticks: readonly string[];
}

export function ScaleLegend({ label, ticks, palette, className, ...props }: ScaleLegendProps) {
  return (
    <div className={cn("w-40 rounded-control border border-cardBorder bg-card p-2 shadow-elevation-card", className)} {...props}>
      <p className="mb-1 text-caption text-textSecondary">{label}</p>
      <div className={scaleLegendVariants({ palette })} aria-hidden="true" />
      <div className="mt-1 flex justify-between font-mono text-caption text-textSecondary tabular-nums">
        {ticks.map((tick) => <span key={tick}>{tick}</span>)}
      </div>
    </div>
  );
}

export { scaleLegendVariants };
