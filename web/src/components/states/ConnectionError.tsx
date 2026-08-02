"use client"

import { WifiOff } from "lucide-react"

import { cn } from "@/lib/utils"

import type { ErrorStateProps } from "./ErrorState"

export interface ConnectionErrorProps
  extends Omit<ErrorStateProps, "description" | "title"> {
  title?: string
  description?: string
}

export function ConnectionError({
  title = "Connection unavailable",
  description = "We couldn't reach the data service. Check your connection and try again.",
  className,
  ...props
}: ConnectionErrorProps) {
  return (
    <section
      role="alert"
      aria-label={title}
      className={cn(
        "flex flex-col items-center rounded-card border border-warning bg-warning-soft p-8 text-center",
        className,
      )}
    >
      <WifiOff className="mb-3 h-8 w-8 text-warning" aria-hidden="true" />
      <h2 className="text-title text-textPrimary">{title}</h2>
      <p className="mt-1 max-w-md text-body text-textSecondary">{description}</p>
      {props.onRetry ? (
        <button
          type="button"
          onClick={props.onRetry}
          className="mt-4 inline-flex min-h-target items-center rounded-button border border-warning px-4 text-label text-textPrimary transition-colors duration-fast hover:bg-warning-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-warning focus-visible:ring-offset-2"
        >
          {props.retryLabel ?? "Try again"}
        </button>
      ) : null}
    </section>
  )
}
