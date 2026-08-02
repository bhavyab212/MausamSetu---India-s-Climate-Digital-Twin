import { Check } from "lucide-react";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

export type RichCycleStatus = "completed" | "active" | "pending" | "critical";

const cycleMarkerVariants = cva(
  "relative z-10 flex h-5 w-5 shrink-0 items-center justify-center rounded-pill border-2 bg-card-solid",
  {
    variants: {
      status: {
        completed: "border-positive text-positive",
        active: "border-primary text-primary ring-4 ring-primary-soft",
        pending: "border-textTertiary text-textTertiary",
        critical: "border-critical text-critical ring-4 ring-critical-soft",
      },
    },
  },
);

export interface RichCycleStage {
  label: string;
  status: RichCycleStatus;
}

export interface RichCycleStripProps {
  cycleLabel: string;
  stages: readonly RichCycleStage[];
  className?: string;
}

export function RichCycleStrip({ cycleLabel, stages, className }: RichCycleStripProps) {
  return (
    <div className={cn("flex min-h-cycle items-center rounded-card border border-cardBorder bg-card px-card shadow-elevation-card", className)}>
      <p className="w-28 shrink-0 text-label text-textPrimary">{cycleLabel}</p>
      <ol className="flex min-w-0 flex-1 items-center" aria-label={`${cycleLabel} processing stages`}>
        {stages.map((stage, index) => (
          <li key={stage.label} className="relative flex flex-1 items-center gap-2">
            {index > 0 ? <span className="absolute right-full top-1/2 h-px w-full bg-divider" aria-hidden="true" /> : null}
            <span className={cycleMarkerVariants({ status: stage.status })}>
              {stage.status === "completed" ? <Check className="h-3 w-3" aria-hidden="true" /> : <span className="h-1.5 w-1.5 rounded-pill bg-current" aria-hidden="true" />}
            </span>
            <span className={cn("relative z-10 bg-card px-1 text-eyebrow", stage.status === "active" ? "text-primary" : stage.status === "critical" ? "text-critical" : stage.status === "completed" ? "text-textSecondary" : "text-textTertiary")}>{stage.label}</span>
            <span className="sr-only">{stage.status}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export { cycleMarkerVariants };
