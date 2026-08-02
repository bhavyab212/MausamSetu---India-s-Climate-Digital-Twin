import { cn } from "@/lib/utils"

import { Skeleton } from "./Skeleton"

export interface ChartSkeletonProps {
  className?: string
  label?: string
}

export function ChartSkeleton({
  className,
  label = "Loading chart",
}: ChartSkeletonProps) {
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
      <Skeleton variant="title" aria-hidden label="Loading chart title" />
      <div className="mt-5 flex h-48 items-end gap-2" aria-hidden="true">
        {["h-1/3", "h-1/2", "h-2/5", "h-3/4", "h-3/5", "h-full", "h-4/5", "h-1/2"].map(
          (height, index) => (
            <Skeleton
              key={index}
              aria-hidden
              className={cn("flex-1 rounded-t-sm", height)}
              label="Loading chart data"
            />
          ),
        )}
      </div>
      <Skeleton aria-hidden className="mt-4 h-3 w-1/2" label="Loading chart legend" />
    </section>
  )
}
