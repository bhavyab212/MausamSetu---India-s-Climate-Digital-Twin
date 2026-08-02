import type { HTMLAttributes, ReactNode } from "react";

import { DonutGauge } from "@/components/DonutGauge";
import type { SemanticTone } from "@/components/InlineLegend";
import { MiniSparkline } from "@/components/MiniSparkline";
import { cn } from "@/lib/utils";

export interface OverviewMetricCardProps extends HTMLAttributes<HTMLElement> {
  label: string;
  value: number | string;
  unit: string;
  detail: string;
  tone: SemanticTone;
  icon: ReactNode;
  visual: "sparkline" | "gauge";
  series?: readonly number[];
}

const toneText: Record<SemanticTone, string> = {
  positive: "text-positive",
  warning: "text-warning",
  critical: "text-critical",
  info: "text-primary",
  neutral: "text-textSecondary",
};

export function OverviewMetricCard({
  label,
  value,
  unit,
  detail,
  tone,
  icon,
  visual,
  series = [],
  className,
  ...props
}: OverviewMetricCardProps) {
  return (
    <article
      className={cn(
        "flex min-h-32 flex-col justify-between rounded-card border border-cardBorder bg-card p-card shadow-elevation-card backdrop-blur-card",
        visual === "gauge" && "border-positive",
        className,
      )}
      {...props}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-eyebrow text-textPrimary">{label}</p>
        <span className={toneText[tone]}>{icon}</span>
      </div>
      {visual === "gauge" ? (
        <div className="flex min-h-20 items-center justify-center gap-2">
          <DonutGauge className="h-20 w-20" value={Number(value)} ariaLabel={`${label} ${value} ${unit}`} tone={tone} />
          <p className={cn("text-caption font-semibold", toneText[tone])}>↑ {detail}</p>
        </div>
      ) : (
        <>
          <div>
            <div className="flex items-baseline gap-2">
              <span className={cn("font-mono text-hero tabular-nums", tone === "warning" || tone === "positive" ? toneText[tone] : "text-textPrimary")}>{value}</span>
              <span className={cn("text-label", toneText[tone])}>{unit}</span>
            </div>
            <p className={cn("mt-1 truncate text-caption font-semibold", toneText[tone])}>{tone === "positive" ? "↑" : "→"} {detail}</p>
          </div>
          <MiniSparkline className="h-6" ariaLabel={`${label} trend`} points={series} tone={tone} />
        </>
      )}
    </article>
  );
}
