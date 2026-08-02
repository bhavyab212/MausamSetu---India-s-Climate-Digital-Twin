import { cn } from "@/lib/utils"

import { Skeleton } from "./Skeleton"

export interface TableSkeletonProps {
  rows?: number
  columns?: number
  className?: string
  label?: string
}

export function TableSkeleton({
  rows = 5,
  columns = 4,
  className,
  label = "Loading table",
}: TableSkeletonProps) {
  const rowCount = Math.max(0, Math.floor(rows))
  const columnCount = Math.max(1, Math.floor(columns))

  return (
    <section
      role="status"
      aria-busy="true"
      aria-label={label}
      className={cn(
        "overflow-hidden rounded-card border border-cardBorder bg-card shadow-elevation-card",
        className,
      )}
    >
      <div className="grid gap-column-gap border-b border-divider bg-card-muted p-card" style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }} aria-hidden="true">
        {Array.from({ length: columnCount }, (_, index) => (
          <Skeleton key={index} aria-hidden className="h-3 w-2/3" label="Loading table header" />
        ))}
      </div>
      <div className="divide-y divide-divider" aria-hidden="true">
        {Array.from({ length: rowCount }, (_, rowIndex) => (
          <div
            key={rowIndex}
            className="grid gap-column-gap p-card"
            style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
          >
            {Array.from({ length: columnCount }, (_, columnIndex) => (
              <Skeleton
                key={columnIndex}
                aria-hidden
                className={cn("h-4", columnIndex === 0 ? "w-4/5" : "w-3/5")}
                label="Loading table cell"
              />
            ))}
          </div>
        ))}
      </div>
    </section>
  )
}
