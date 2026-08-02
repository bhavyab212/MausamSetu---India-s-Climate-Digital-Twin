import type { ReactNode } from "react"
import { Inbox } from "lucide-react"

import { cn } from "@/lib/utils"

export interface EmptyStateProps {
  title: string
  description?: string
  action?: ReactNode
  icon?: ReactNode
  className?: string
}

export function EmptyState({
  title,
  description,
  action,
  icon,
  className,
}: EmptyStateProps) {
  return (
    <section
      aria-label={title}
      className={cn(
        "flex flex-col items-center rounded-card border border-cardBorder bg-card p-8 text-center",
        className,
      )}
    >
      <div className="mb-3 text-textTertiary" aria-hidden="true">
        {icon ?? <Inbox className="h-8 w-8" />}
      </div>
      <h2 className="text-title text-textPrimary">{title}</h2>
      {description ? (
        <p className="mt-1 max-w-md text-body text-textSecondary">{description}</p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </section>
  )
}
