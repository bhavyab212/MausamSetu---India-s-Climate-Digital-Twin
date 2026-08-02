import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

export interface CountdownPillProps {
  value?: ReactNode
  children?: ReactNode
  label?: string
  className?: string
}

export function CountdownPill({
  value,
  children,
  label = "Time remaining",
  className,
}: CountdownPillProps) {
  return (
    <span
      aria-label={label}
      className={cn(
        "inline-flex min-h-control items-center rounded-pill border border-cardBorder bg-card px-3 font-mono text-caption text-textPrimary",
        className
      )}
    >
      {value ?? children}
    </span>
  )
}
