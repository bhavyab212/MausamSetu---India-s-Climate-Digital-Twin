"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Bell,
  CheckCircle2,
  ChevronRight,
  CloudRain,
  Droplet,
  Gauge,
  Globe2,
  House,
  Layers3,
  Settings,
  Sprout,
  type LucideIcon,
} from "lucide-react";
import { cva } from "class-variance-authority";

import { CommandK } from "@/components/CommandK";
import { cn } from "@/lib/utils";

const dashboardNavigationItemVariants = cva(
  "group relative flex min-h-target items-center gap-3 rounded-control px-3 text-label transition-colors duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  {
    variants: {
      active: {
        true: "bg-primary-soft text-primary",
        false: "text-textSecondary hover:bg-card-muted hover:text-textPrimary",
      },
    },
    defaultVariants: {
      active: false,
    },
  },
);

export interface DashboardNavigationItem {
  href: string;
  label: string;
  icon: LucideIcon;
  badge?: string;
}

export const dashboardNavigationItems: readonly DashboardNavigationItem[] = [
  { href: "/overview", label: "Overview", icon: House },
  { href: "/map", label: "Live Map", icon: Globe2 },
  { href: "/forecast", label: "Forecast", icon: CloudRain },
  { href: "/scenarios", label: "Scenarios", icon: Layers3 },
  { href: "/validation", label: "Validation", icon: CheckCircle2 },
  { href: "/sectors", label: "Sectors", icon: Gauge },
  { href: "/alerts", label: "Alerts", icon: Bell, badge: "3" },
];

export interface DashboardSidebarProps {
  className?: string;
  navigationItems?: readonly DashboardNavigationItem[];
}

const pilots = [
  { id: "cauvery", label: "Cauvery Basin", active: true },
  { id: "krishna", label: "Krishna Basin", active: false },
  { id: "godavari", label: "Godavari Basin", active: false },
] as const;

export function DashboardSidebar({
  className,
  navigationItems = dashboardNavigationItems,
}: DashboardSidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "flex h-full w-sidebar shrink-0 flex-col overflow-hidden rounded-shell border border-cardBorder bg-workspace p-card shadow-elevation-floating backdrop-blur-shell",
        className,
      )}
      aria-label="Primary navigation"
    >
      <div className="flex items-center gap-3 px-2 py-2">
        <div
          className="flex h-control w-control shrink-0 items-center justify-center rounded-control bg-primary-soft text-primary"
          aria-hidden="true"
        >
          <Droplet className="h-5 w-5 fill-current" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-title text-textPrimary">MausamSetu</p>
          <p className="text-caption leading-tight text-textSecondary">
            AI-Powered Climate
            <br />
            Digital Twin of India
          </p>
        </div>
      </div>

      <CommandK
        className="mt-card w-full justify-between bg-card-solid"
        label="Search regions, metrics…"
        title="Search MausamSetu"
        placeholder="Search regions, metrics, or screens"
      />

      <div className="mt-card min-h-0 flex-1 overflow-hidden pr-1">
        <nav aria-labelledby="dashboard-workspace-heading">
          <h2
            id="dashboard-workspace-heading"
            className="mb-2 px-3 text-eyebrow uppercase text-textTertiary"
          >
            Workspace
          </h2>
          <ul className="space-y-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);

              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={dashboardNavigationItemVariants({ active })}
                    aria-current={active ? "page" : undefined}
                  >
                    {active ? (
                      <span
                        className="absolute inset-y-2 left-0 w-1 rounded-pill bg-primary"
                        aria-hidden="true"
                      />
                    ) : null}
                    <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                    <span className="min-w-0 flex-1 truncate">{item.label}</span>
                    {item.badge ? (
                      <span className="rounded-pill bg-critical px-2 py-1 text-caption text-textInverse">
                        {item.badge}
                      </span>
                    ) : null}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <section className="mt-card-lg border-t border-divider pt-card" aria-labelledby="dashboard-pilots-heading">
          <h2
            id="dashboard-pilots-heading"
            className="mb-2 px-3 text-eyebrow uppercase text-textTertiary"
          >
            Pilots
          </h2>
          <ul className="space-y-1">
            {pilots.map((pilot) => (
              <li key={pilot.id}>
                <button
                  type="button"
                  className="flex min-h-control w-full items-center gap-3 rounded-control px-3 text-left text-label text-textSecondary transition-colors duration-fast hover:bg-card-muted hover:text-textPrimary"
                  aria-pressed={pilot.active}
                >
                  <span
                    className={cn(
                      "h-2 w-2 shrink-0 rounded-pill",
                      pilot.active ? "bg-primary" : "bg-divider",
                    )}
                    aria-hidden="true"
                  />
                  <span className={pilot.active ? "text-textPrimary" : undefined}>{pilot.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="mt-card border-t border-divider pt-card">
        <div className="flex items-center gap-3 rounded-card border border-cardBorder bg-card p-3 shadow-elevation-card">
          <div
            className="flex h-control w-control shrink-0 items-center justify-center rounded-pill bg-primary-soft text-label text-primary"
            aria-hidden="true"
          >
            BC
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-label text-textPrimary">Bhavya Chaudhary</p>
            <p className="truncate text-caption text-textSecondary">Climate Ops</p>
          </div>
          <button
            type="button"
            className="flex h-control w-control shrink-0 items-center justify-center rounded-control text-textSecondary transition-colors duration-fast hover:bg-card-muted hover:text-textPrimary"
            aria-label="Open settings"
          >
            <Settings className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="mt-2 flex items-center justify-between rounded-card px-3 py-2 text-positive">
          <span className="inline-flex items-center gap-2 text-label">
            <Sprout className="h-4 w-4" aria-hidden="true" />
            मौसम सेतु
          </span>
          <ChevronRight className="h-4 w-4 text-textTertiary" aria-hidden="true" />
        </div>
      </div>
    </aside>
  );
}

export { dashboardNavigationItemVariants };
