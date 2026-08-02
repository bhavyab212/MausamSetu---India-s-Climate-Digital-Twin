import type { ReactNode } from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { AlertCircle, CheckCircle, Info, TriangleAlert } from "lucide-react"

import { cn } from "@/lib/utils"

const alertRowVariants = cva(
  "flex items-start gap-row-gap rounded-card border bg-card p-card",
  {
    variants: {
      severity: {
        info: "border-primary",
        positive: "border-positive",
        warning: "border-warning",
        critical: "border-critical",
      },
    },
    defaultVariants: { severity: "info" },
  }
)

const severityIcons = {
  info: Info,
  positive: CheckCircle,
  warning: TriangleAlert,
  critical: AlertCircle,
} as const

export interface AlertRowProps extends VariantProps<typeof alertRowVariants> {
  timestamp: string
  title: string
  description: string
  action?: ReactNode
  className?: string
}

export function AlertRow({
  timestamp,
  severity = "info",
  title,
  description,
  action,
  className,
}: AlertRowProps) {
  const resolvedSeverity = severity ?? "info"
  const Icon = severityIcons[resolvedSeverity]
  const iconColor = {
    info: "text-primary",
    positive: "text-positive",
    warning: "text-warning",
    critical: "text-critical",
  }[resolvedSeverity]

  return (
    <article className={cn(alertRowVariants({ severity: resolvedSeverity }), className)}>
      <Icon className={cn("mt-1 h-4 w-4", iconColor)} aria-hidden="true" />
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-eyebrow uppercase text-textSecondary">{resolvedSeverity}</span>
          <time className="font-mono text-caption text-textTertiary">{timestamp}</time>
        </div>
        <h3 className="mt-1 text-title text-textPrimary">{title}</h3>
        <p className="mt-1 text-body text-textSecondary">{description}</p>
      </div>
      {action ? <div className="ml-auto shrink-0">{action}</div> : null}
    </article>
  )
}
