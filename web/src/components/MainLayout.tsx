import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";
import { Sidebar, type SidebarProps } from "@/components/Sidebar";
import { TopBar, type TopBarProps } from "@/components/TopBar";

const mainLayoutVariants = cva("flex min-h-screen bg-background text-textPrimary", {
  variants: {
    shell: {
      default: "",
      workspace: "p-workspace",
    },
  },
  defaultVariants: {
    shell: "default",
  },
});

const mainContentVariants = cva("min-w-0 flex-1", {
  variants: {
    density: {
      compact: "p-card",
      default: "p-card-lg",
      spacious: "p-workspace",
    },
  },
  defaultVariants: {
    density: "default",
  },
});

export interface MainLayoutProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "children">,
    VariantProps<typeof mainLayoutVariants>,
    VariantProps<typeof mainContentVariants> {
  children: ReactNode;
  sidebarProps?: SidebarProps;
  topBarProps?: TopBarProps;
  sidebarSlot?: ReactNode;
  topBarSlot?: ReactNode;
  contentClassName?: string;
  mainAriaLabel?: string;
}

export function MainLayout({
  children,
  sidebarProps,
  topBarProps,
  sidebarSlot,
  topBarSlot,
  shell,
  density,
  contentClassName,
  mainAriaLabel = "Main workspace",
  className,
  ...props
}: MainLayoutProps) {
  return (
    <div className={cn(mainLayoutVariants({ shell }), className)} {...props}>
      <div className="sticky top-0 h-screen shrink-0">
        {sidebarSlot ?? <Sidebar {...sidebarProps} />}
      </div>
      <div className="min-w-0 flex-1">
        {topBarSlot ?? <TopBar {...topBarProps} />}
        <main className={cn(mainContentVariants({ density }), contentClassName)} aria-label={mainAriaLabel}>
          {children}
        </main>
      </div>
    </div>
  );
}

export { mainContentVariants, mainLayoutVariants };
