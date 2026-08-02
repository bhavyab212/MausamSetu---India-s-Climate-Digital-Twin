import { cva } from "class-variance-authority"

import { cn } from "@/lib/utils"

export const DEFAULT_CYCLE_PHASES = [
  "INGEST",
  "REGRID",
  "ASSIMILATE",
  "FORECAST",
  "IMPACT",
  "RENDER",
] as const

export type CyclePhase = (typeof DEFAULT_CYCLE_PHASES)[number]
export type CyclePhaseStatus = "completed" | "active" | "pending"

const phaseVariants = cva(
  "flex min-h-cycle flex-1 items-center justify-center rounded-control border px-3 text-center text-eyebrow transition-colors duration-fast",
  {
    variants: {
      status: {
        completed: "border-positive bg-positive-soft text-positive",
        active: "border-primary bg-primary-soft text-primary-strong",
        pending: "border-cardBorder bg-card text-textTertiary",
      },
    },
  }
)

export interface CycleStripProps {
  activePhase: CyclePhase
  phases?: readonly CyclePhase[]
  className?: string
  "aria-label"?: string
}

export function CycleStrip({
  activePhase,
  phases = DEFAULT_CYCLE_PHASES,
  className,
  "aria-label": ariaLabel = "Processing cycle",
}: CycleStripProps) {
  const activeIndex = phases.indexOf(activePhase)

  return (
    <ol aria-label={ariaLabel} className={cn("flex flex-wrap gap-2", className)}>
      {phases.map((phase, index) => {
        const status: CyclePhaseStatus =
          index < activeIndex ? "completed" : index === activeIndex ? "active" : "pending"
        return (
          <li
            key={phase}
            aria-current={status === "active" ? "step" : undefined}
            className={phaseVariants({ status })}
          >
            <span>{phase}</span>
            <span className="sr-only">, {status}</span>
          </li>
        )
      })}
    </ol>
  )
}
