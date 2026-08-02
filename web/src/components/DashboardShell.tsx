import type { ReactNode } from "react";

import { DashboardSidebar } from "@/components/DashboardSidebar";
import { cn } from "@/lib/utils";

export interface DashboardShellProps {
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  sidebarSlot?: ReactNode;
}

export function DashboardShell({
  children,
  className,
  contentClassName,
  sidebarSlot,
}: DashboardShellProps) {
  return (
    <div
      className={cn(
        "dashboard-canvas flex h-dvh min-h-0 gap-6 overflow-hidden bg-card-muted bg-cover bg-center p-workspace text-textPrimary",
        className,
      )}
      style={{ backgroundImage: "url('/assets/cauvery-landscape.webp')" }}
    >
      {sidebarSlot ?? <DashboardSidebar />}
      <main
        className={cn(
          "min-w-0 min-h-0 flex-1 overflow-hidden rounded-shell border border-cardBorder bg-workspace p-card-lg shadow-elevation-floating backdrop-blur-shell",
          contentClassName,
        )}
        aria-label="Dashboard workspace"
      >
        {children}
      </main>
    </div>
  );
}
