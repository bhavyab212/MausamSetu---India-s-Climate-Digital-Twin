import { cn } from "@/lib/utils"

import { Skeleton } from "./Skeleton"

export interface CardSkeletonProps {
  lines?: number
  className?: string
  label?: string
}

export function CardSkeleton({
  lines = 3,
  className,
  label = "Loading card",
}: CardSkeletonProps) {
  const lineCount = Math.max(0, Math.floor(lines))

  return (
    <section
      role="status"
      aria-busy="true"
      aria-label={label}
      className={cn(
        "rounded-card border border-cardBorder bg-card p-card shadow-elevation-card",
        className,
      )}
    >
      <Skeleton variant="title" aria-hidden label="Loading card title" />
      <div className="mt-4 space-y-3" aria-hidden="true">
        {Array.from({ length: lineCount }, (_, index) => (
          <Skeleton
            key={index}
            aria-hidden
            className={index === lineCount - 1 ? "w-2/3" : undefined}
            label="Loading card content"
          />
        ))}
      </div>
    </section>
  )
}
