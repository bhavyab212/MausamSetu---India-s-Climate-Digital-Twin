import type { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const dividerVariants = cva("shrink-0 border-0 bg-divider", {
  variants: {
    orientation: {
      horizontal: "h-px w-full",
      vertical: "h-full min-h-6 w-px self-stretch",
    },
    emphasis: {
      default: "opacity-100",
      subtle: "opacity-50",
    },
  },
  defaultVariants: {
    orientation: "horizontal",
    emphasis: "default",
  },
});

export interface DividerProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "children">,
    VariantProps<typeof dividerVariants> {
  decorative?: boolean;
}

export function Divider({
  orientation = "horizontal",
  emphasis,
  decorative = false,
  className,
  ...props
}: DividerProps) {
  return (
    <div
      className={cn(dividerVariants({ orientation, emphasis }), className)}
      role={decorative ? "presentation" : "separator"}
      aria-orientation={decorative ? undefined : orientation ?? "horizontal"}
      {...props}
    />
  );
}

export { dividerVariants };
