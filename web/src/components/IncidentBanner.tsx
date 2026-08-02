import type { ReactNode } from "react"
import { AlertTriangle } from "lucide-react"

import { cn } from "@/lib/utils"

export interface IncidentBannerProps {
  title: string
  description: string
  actions?: ReactNode
  className?: string
}

export function IncidentBanner({
  title,
  description,
  actions,
  className,
}: IncidentBannerProps) {
  return (
    <section
      role="alert"
      aria-label="Critical incident"
      className={cn(
        "flex flex-col gap-3 rounded-card border border-critical bg-critical-soft p-card text-textPrimary sm:flex-row sm:items-start",
        className
      )}
    >
      <AlertTriangle className="h-5 w-5 shrink-0 text-critical" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="text-eyebrow uppercase text-critical">Critical incident</p>
        <h2 className="mt-1 text-title">{title}</h2>
        <p className="mt-1 text-body text-textSecondary">{description}</p>
      </div>
      {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
    </section>
  )
}
