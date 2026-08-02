"use client"

import { AlertCircle, RefreshCw } from "lucide-react"

import { cn } from "@/lib/utils"

export interface ErrorStateProps {
  title?: string
  description: string
  onRetry?: () => void
  retryLabel?: string
  className?: string
}

export function ErrorState({
  title = "Something went wrong",
  description,
  onRetry,
  retryLabel = "Try again",
  className,
}: ErrorStateProps) {
  return (
    <section
      role="alert"
      aria-label={title}
      className={cn(
        "flex flex-col items-center rounded-card border border-critical bg-critical-soft p-8 text-center",
        className,
      )}
    >
      <AlertCircle className="mb-3 h-8 w-8 text-critical" aria-hidden="true" />
      <h2 className="text-title text-textPrimary">{title}</h2>
      <p className="mt-1 max-w-md text-body text-textSecondary">{description}</p>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 inline-flex min-h-target items-center gap-2 rounded-button bg-critical px-4 text-label text-textInverse transition-colors duration-fast hover:bg-critical-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-critical focus-visible:ring-offset-2"
        >
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          {retryLabel}
        </button>
      ) : null}
    </section>
  )
}
