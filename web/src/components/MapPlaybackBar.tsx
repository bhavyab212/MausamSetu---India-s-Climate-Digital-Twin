import { Play } from "lucide-react";

import { cn } from "@/lib/utils";

export interface MapPlaybackBarProps {
  startLabel: string;
  currentLabel: string;
  endLabel: string;
  className?: string;
}

export function MapPlaybackBar({ startLabel, currentLabel, endLabel, className }: MapPlaybackBarProps) {
  return (
    <div className={cn("flex h-cycle items-center gap-card border-t border-divider bg-card px-card", className)}>
      <button type="button" aria-label="Play climate timeline" className="flex h-control w-control shrink-0 items-center justify-center rounded-pill border border-positive text-positive hover:bg-positive-soft">
        <Play className="h-4 w-4 fill-current" aria-hidden="true" />
      </button>
      <span className="shrink-0 text-caption text-textSecondary">{startLabel}</span>
      <div className="relative h-px flex-1 bg-divider" aria-hidden="true">
        <span className="absolute left-0 top-0 h-px w-1/2 bg-positive" />
        <span className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-pill border-2 border-card-solid bg-positive shadow-elevation-card" />
        <span className="absolute left-1/2 top-2 -translate-x-1/2 whitespace-nowrap text-caption text-textPrimary">{currentLabel}</span>
      </div>
      <span className="shrink-0 text-caption text-textSecondary">{endLabel}</span>
    </div>
  );
}
