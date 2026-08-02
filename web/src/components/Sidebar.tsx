import type { HTMLAttributes, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import {
  BarChart3,
  Beaker,
  CloudSun,
  LayoutDashboard,
  Radio,
  User,
} from "lucide-react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const sidebarVariants = cva(
  "flex h-screen w-sidebar shrink-0 flex-col border-r border-divider bg-workspace p-card backdrop-blur-shell",
  {
    variants: {
      appearance: {
        default: "shadow-elevation-card",
        flat: "shadow-none",
      },
    },
    defaultVariants: {
      appearance: "default",
    },
  },
);

const navigationItemVariants = cva(
  "flex w-full items-center gap-3 rounded-control px-3 py-2 text-left text-label transition-colors duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  {
    variants: {
      selected: {
        true: "bg-primary-soft text-primary",
        false: "text-textSecondary hover:bg-card-muted hover:text-textPrimary",
      },
      disabled: {
        true: "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-textSecondary",
        false: "cursor-pointer",
      },
    },
    compoundVariants: [
      {
        selected: true,
        disabled: true,
        className: "hover:bg-primary-soft hover:text-primary",
      },
    ],
    defaultVariants: {
      selected: false,
      disabled: false,
    },
  },
);

const pilotStatusVariants = cva("ml-auto rounded-pill px-2 py-1 text-caption", {
  variants: {
    status: {
      active: "bg-positive-soft text-positive",
      draft: "bg-warning-soft text-warning",
      paused: "bg-card-muted text-textTertiary",
    },
  },
  defaultVariants: {
    status: "active",
  },
});

export interface SidebarNavigationItem {
  id: string;
  label: ReactNode;
  icon?: LucideIcon;
  selected?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
}

export interface SidebarPilotItem extends SidebarNavigationItem {
  status?: "active" | "draft" | "paused";
}

export interface SidebarUser {
  name: string;
  role: string;
  initials?: string;
}

export interface SidebarProps
  extends Omit<HTMLAttributes<HTMLElement>, "children">,
    VariantProps<typeof sidebarVariants> {
  workspaceItems?: readonly SidebarNavigationItem[];
  pilots?: readonly SidebarPilotItem[];
  user?: SidebarUser;
  footerSlot?: ReactNode;
  workspaceLabel?: string;
  pilotsLabel?: string;
}

const defaultWorkspaceItems: readonly SidebarNavigationItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, selected: true },
  { id: "forecast", label: "Forecast workspace", icon: CloudSun },
  { id: "observations", label: "Observation network", icon: Radio },
  { id: "analytics", label: "Climate analytics", icon: BarChart3, disabled: true },
];

const defaultPilots: readonly SidebarPilotItem[] = [
  { id: "monsoon", label: "Monsoon readiness", icon: Beaker, status: "active" },
  { id: "district", label: "District forecast", icon: Beaker, status: "draft" },
];

const defaultUser: SidebarUser = {
  name: "Workspace User",
  role: "MausamSetu Team",
  initials: "MS",
};

function NavigationItem({ item, pilot = false }: { item: SidebarNavigationItem | SidebarPilotItem; pilot?: boolean }) {
  const Icon = item.icon;
  const status = pilot ? (item as SidebarPilotItem).status : undefined;

  return (
    <button
      type="button"
      className={navigationItemVariants({ selected: item.selected, disabled: item.disabled })}
      disabled={item.disabled}
      aria-current={item.selected ? "page" : undefined}
      aria-label={item.ariaLabel}
    >
      {Icon ? <Icon aria-hidden="true" className="h-4 w-4 shrink-0" /> : null}
      <span className="min-w-0 truncate">{item.label}</span>
      {status ? <span className={pilotStatusVariants({ status })}>{status}</span> : null}
    </button>
  );
}

export function Sidebar({
  workspaceItems = defaultWorkspaceItems,
  pilots = defaultPilots,
  user = defaultUser,
  footerSlot,
  workspaceLabel = "Workspace",
  pilotsLabel = "Pilots",
  appearance,
  className,
  ...props
}: SidebarProps) {
  const initials = user.initials ?? user.name.slice(0, 2).toUpperCase();

  return (
    <aside className={cn(sidebarVariants({ appearance }), className)} aria-label="Primary" {...props}>
      <div className="flex h-topbar items-center gap-3 border-b border-divider px-2">
        <div
          className="flex h-control w-control shrink-0 items-center justify-center rounded-control bg-primary text-label text-primary-foreground"
          aria-hidden="true"
        >
          MS
        </div>
        <div className="min-w-0">
          <p className="truncate text-title text-textPrimary">MausamSetu</p>
          <p className="truncate text-caption text-textSecondary">मौसम सेतु</p>
        </div>
      </div>

      <nav className="mt-card flex min-h-0 flex-1 flex-col gap-card overflow-y-auto" aria-label="Workspace navigation">
        <section aria-labelledby="workspace-navigation-heading">
          <h2 id="workspace-navigation-heading" className="mb-2 px-3 text-eyebrow uppercase text-textTertiary">
            {workspaceLabel}
          </h2>
          <ul className="space-y-1">
            {workspaceItems.map((item) => (
              <li key={item.id}>
                <NavigationItem item={item} />
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="pilots-navigation-heading">
          <h2 id="pilots-navigation-heading" className="mb-2 px-3 text-eyebrow uppercase text-textTertiary">
            {pilotsLabel}
          </h2>
          <ul className="space-y-1">
            {pilots.map((pilot) => (
              <li key={pilot.id}>
                <NavigationItem item={pilot} pilot />
              </li>
            ))}
          </ul>
        </section>
      </nav>

      <div className="mt-card border-t border-divider pt-card">
        {footerSlot ? <div className="mb-card">{footerSlot}</div> : null}
        <div className="flex items-center gap-3 rounded-card border border-cardBorder bg-card p-3 shadow-elevation-card">
          <div
            className="flex h-control w-control shrink-0 items-center justify-center rounded-pill bg-primary-soft text-label text-primary"
            aria-hidden="true"
          >
            {initials || <User className="h-4 w-4" />}
          </div>
          <div className="min-w-0">
            <p className="truncate text-label text-textPrimary">{user.name}</p>
            <p className="truncate text-caption text-textSecondary">{user.role}</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

export { navigationItemVariants, pilotStatusVariants, sidebarVariants };
