import type { HTMLAttributes, ReactNode } from "react";
import { Bell, ChevronDown, Download, Share2 } from "lucide-react";

import { cn } from "@/lib/utils";

const headerActionClass =
  "inline-flex min-h-control items-center justify-center gap-2 rounded-button border border-cardBorder bg-card px-3 text-label text-textSecondary shadow-elevation-card transition-colors duration-fast hover:border-primary hover:text-primary";

export interface DashboardHeaderProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  title: ReactNode;
  metadata?: ReactNode;
  actions?: ReactNode;
  notificationCount?: number;
}

export function DashboardHeader({
  title,
  metadata,
  actions,
  notificationCount = 2,
  className,
  ...props
}: DashboardHeaderProps) {
  return (
    <header
      className={cn("flex min-h-topbar items-center justify-between gap-card-lg", className)}
      {...props}
    >
      <div className="min-w-0">
        <h1 className="truncate text-page text-textPrimary">{title}</h1>
        {metadata ? <p className="mt-1 truncate text-caption text-textSecondary">{metadata}</p> : null}
      </div>
      <div className="flex shrink-0 items-center gap-2" aria-label="Page actions">
        {actions ?? (
          <>
            <button type="button" className={headerActionClass}>
              <Download className="h-4 w-4 text-positive" aria-hidden="true" />
              Export
            </button>
            <button type="button" className={headerActionClass}>
              <Share2 className="h-4 w-4 text-positive" aria-hidden="true" />
              Share
            </button>
            <button type="button" className={cn(headerActionClass, "relative px-3")}>
              <Bell className="h-4 w-4" aria-hidden="true" />
              Notifications
              {notificationCount > 0 ? (
                <span className="rounded-pill bg-primary px-2 py-1 text-caption text-textInverse">
                  {notificationCount}
                </span>
              ) : null}
            </button>
            <button
              type="button"
              className="inline-flex min-h-control items-center gap-2 rounded-button px-2 text-label text-textSecondary hover:bg-card-muted hover:text-textPrimary"
              aria-label="Open user menu"
            >
              <span className="flex h-control w-control items-center justify-center rounded-pill bg-primary-soft text-primary">
                BC
              </span>
              <ChevronDown className="h-4 w-4" aria-hidden="true" />
            </button>
          </>
        )}
      </div>
    </header>
  );
}

export { headerActionClass };
