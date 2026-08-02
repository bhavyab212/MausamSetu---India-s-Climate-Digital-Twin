import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const timelineDotVariants = cva(
  "inline-flex h-4 w-4 shrink-0 rounded-full border-2",
  {
    variants: {
      status: {
        completed: "border-positive bg-positive",
        active: "border-primary bg-primary ring-2 ring-primary-soft",
        pending: "border-textTertiary bg-card",
        critical: "border-critical bg-critical",
      },
    },
    defaultVariants: { status: "pending" },
  }
)

export interface TimelineDotProps extends VariantProps<typeof timelineDotVariants> {
  label: string
  className?: string
}

export function TimelineDot({ status = "pending", label, className }: TimelineDotProps) {
  return (
    <span
      role="img"
      aria-label={`${label}: ${status ?? "pending"}`}
      className={cn(timelineDotVariants({ status }), className)}
    />
  )
}
