import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const kpiHeaderStripVariants = cva(
  "grid grid-flow-col auto-cols-fr rounded-panel border border-cardBorder bg-workspace shadow-elevation-card backdrop-blur-card",
  {
    variants: {
      density: {
        compact: "gap-2 p-2",
        default: "gap-column-gap p-card",
        spacious: "gap-card-lg p-card-lg",
      },
    },
    defaultVariants: {
      density: "default",
    },
  },
);

export interface KPIHeaderStripProps
  extends Omit<HTMLAttributes<HTMLElement>, "children">,
    VariantProps<typeof kpiHeaderStripVariants> {
  children: ReactNode;
  ariaLabel?: string;
}

export function KPIHeaderStrip({
  children,
  density,
  ariaLabel = "Key performance indicators",
  className,
  ...props
}: KPIHeaderStripProps) {
  return (
    <section
      aria-label={ariaLabel}
      className={cn(kpiHeaderStripVariants({ density }), className)}
      {...props}
    >
      {children}
    </section>
  );
}

export { kpiHeaderStripVariants };
