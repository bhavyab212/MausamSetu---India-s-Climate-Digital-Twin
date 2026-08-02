"use client"

import * as React from "react"
import { Search, X } from "lucide-react"

import { cn } from "@/lib/utils"

export interface CommandKProps {
  label?: string
  title?: string
  placeholder?: string
  emptyTitle?: string
  emptyDescription?: string
  enableShortcut?: boolean
  className?: string
}

export function CommandK({
  label = "Open command menu",
  title = "Command menu",
  placeholder = "Search commands",
  emptyTitle = "No commands available",
  emptyDescription = "Commands will appear here when they are available.",
  enableShortcut = true,
  className,
}: CommandKProps) {
  const [open, setOpen] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)
  const titleId = React.useId()

  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        enableShortcut &&
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === "k"
      ) {
        event.preventDefault()
        setOpen((current) => !current)
      }
      if (event.key === "Escape") setOpen(false)
    }
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [enableShortcut])

  React.useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  return (
    <>
      <button
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-keyshortcuts={enableShortcut ? "Meta+K Control+K" : undefined}
        onClick={() => setOpen(true)}
        className={cn(
          "inline-flex min-h-target items-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary transition-colors duration-fast hover:border-primary hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          className
        )}
      >
        <Search className="h-4 w-4" aria-hidden="true" />
        <span>{label}</span>
        {enableShortcut ? (
          <kbd className="rounded-sm border border-divider bg-muted px-2 py-1 font-mono text-caption" aria-hidden="true">
            ⌘ K
          </kbd>
        ) : null}
      </button>
      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-foreground/50 p-4 pt-16"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) setOpen(false)
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            className="w-full max-w-lg rounded-panel border border-cardBorder bg-popover p-card shadow-elevation-floating"
          >
            <div className="flex items-center justify-between gap-3">
              <h2 id={titleId} className="text-title text-textPrimary">{title}</h2>
              <button
                type="button"
                aria-label="Close command menu"
                onClick={() => setOpen(false)}
                className="inline-flex h-target w-target items-center justify-center rounded-button text-textSecondary hover:bg-muted hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            <div className="mt-4 flex items-center gap-2 rounded-control border border-input px-3 focus-within:ring-2 focus-within:ring-ring">
              <Search className="h-4 w-4 text-textTertiary" aria-hidden="true" />
              <input
                ref={inputRef}
                type="search"
                aria-label={placeholder}
                placeholder={placeholder}
                className="h-control w-full bg-transparent text-body text-textPrimary outline-none placeholder:text-textTertiary"
              />
            </div>
            <div className="py-8 text-center" role="status">
              <p className="text-label text-textPrimary">{emptyTitle}</p>
              <p className="mt-1 text-body text-textSecondary">{emptyDescription}</p>
            </div>
          </section>
        </div>
      ) : null}
    </>
  )
}
