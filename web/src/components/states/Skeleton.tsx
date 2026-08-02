import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const skeletonVariants = cva("animate-pulse bg-muted motion-reduce:animate-none", {
  variants: {
    variant: {
      text: "h-4 w-full rounded-sm",
      title: "h-6 w-2/3 rounded-sm",
      avatar: "h-control w-control rounded-full",
      card: "h-32 w-full rounded-card",
      block: "h-20 w-full rounded-control",
    },
  },
  defaultVariants: { variant: "text" },
})

export interface SkeletonProps
  extends Omit<React.ComponentPropsWithoutRef<"div">, "children">,
    VariantProps<typeof skeletonVariants> {
  label?: string
}

export function Skeleton({
  variant,
  label = "Loading",
  className,
  "aria-hidden": ariaHidden,
  ...props
}: SkeletonProps) {
  return (
    <div
      role={ariaHidden ? undefined : "status"}
      aria-busy={ariaHidden ? undefined : "true"}
      aria-label={ariaHidden ? undefined : label}
      aria-hidden={ariaHidden}
      className={cn(skeletonVariants({ variant }), className)}
      {...props}
    >
      {!ariaHidden ? <span className="sr-only">{label}</span> : null}
    </div>
  )
}
