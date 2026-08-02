import type { HTMLAttributes, ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const topBarVariants = cva(
  "flex w-full items-center justify-between border-b border-divider bg-workspace backdrop-blur-shell",
  {
    variants: {
      density: {
        compact: "h-cycle px-card",
        default: "h-topbar px-card-lg",
      },
      position: {
        static: "relative",
        sticky: "sticky top-0 z-20",
      },
    },
    defaultVariants: {
      density: "default",
      position: "sticky",
    },
  },
);

export interface TopBarBreadcrumbItem {
  id: string;
  label: ReactNode;
}

export interface TopBarProps
  extends Omit<HTMLAttributes<HTMLElement>, "children">,
    VariantProps<typeof topBarVariants> {
  breadcrumbs?: readonly TopBarBreadcrumbItem[];
  actions?: ReactNode;
  breadcrumbLabel?: string;
}

export function TopBar({
  breadcrumbs = [],
  actions,
  breadcrumbLabel = "Breadcrumb",
  density,
  position,
  className,
  ...props
}: TopBarProps) {
  return (
    <header className={cn(topBarVariants({ density, position }), className)} {...props}>
      <nav aria-label={breadcrumbLabel} className="min-w-0">
        <ol className="flex min-w-0 items-center gap-2">
          {breadcrumbs.map((item, index) => {
            const isCurrent = index === breadcrumbs.length - 1;

            return (
              <li key={item.id} className="flex min-w-0 items-center gap-2">
                {index > 0 ? (
                  <ChevronRight
                    aria-hidden="true"
                    className="h-4 w-4 shrink-0 text-textTertiary"
                  />
                ) : null}
                <span
                  className={cn(
                    "truncate text-label",
                    isCurrent ? "text-textPrimary" : "text-textSecondary",
                  )}
                  aria-current={isCurrent ? "page" : undefined}
                >
                  {item.label}
                </span>
              </li>
            );
          })}
        </ol>
      </nav>
      {actions ? (
        <div className="ml-card-lg flex shrink-0 items-center gap-2" aria-label="Page actions">
          {actions}
        </div>
      ) : null}
    </header>
  );
}

export { topBarVariants };
