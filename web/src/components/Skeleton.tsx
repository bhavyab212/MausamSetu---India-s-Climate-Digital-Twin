import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const skeletonVariants = cva(
  "animate-pulse bg-muted motion-reduce:animate-none",
  {
    variants: {
      variant: {
        text: "h-4 w-full rounded-sm",
        title: "h-6 w-2/3 rounded-sm",
        avatar: "h-target w-target rounded-full",
        card: "h-32 w-full rounded-card",
        block: "h-20 w-full rounded-control",
      },
    },
    defaultVariants: { variant: "text" },
  }
)

export interface SkeletonProps extends VariantProps<typeof skeletonVariants> {
  label?: string
  className?: string
}

export function Skeleton({ variant, label = "Loading", className }: SkeletonProps) {
  return (
    <div
      role="status"
      aria-busy="true"
      aria-label={label}
      className={cn(skeletonVariants({ variant }), className)}
    >
      <span className="sr-only">{label}</span>
    </div>
  )
}
