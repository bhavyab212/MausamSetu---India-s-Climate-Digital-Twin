import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const tabPillVariants = cva(
  "inline-flex min-h-target items-center justify-center rounded-pill border px-4 text-label transition-colors duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      active: {
        true: "border-primary bg-primary text-primary-foreground",
        false:
          "border-cardBorder bg-card text-textSecondary hover:border-primary hover:text-primary",
      },
      size: {
        sm: "min-h-control px-3",
        md: "min-h-target px-4",
      },
    },
    defaultVariants: { active: false, size: "md" },
  }
)

export interface TabPillProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof tabPillVariants> {
  active?: boolean
}

export const TabPill = React.forwardRef<HTMLButtonElement, TabPillProps>(
  ({ active = false, size, className, type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      role="tab"
      aria-selected={active}
      className={cn(tabPillVariants({ active, size }), className)}
      {...props}
    />
  )
)

TabPill.displayName = "TabPill"
