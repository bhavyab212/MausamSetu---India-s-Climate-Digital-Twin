import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const statusPillVariants = cva(
  "inline-flex w-fit items-center rounded-pill font-semibold transition-opacity duration-fast",
  {
    variants: {
      tone: {
        positive: "bg-positive-soft text-positive",
        warning: "bg-warning-soft text-warning",
        critical: "bg-critical-soft text-critical",
        info: "bg-primary-soft text-primary",
        neutral: "bg-card-muted text-textSecondary",
      },
      size: {
        sm: "gap-1 px-2 py-1 text-caption",
        md: "gap-2 px-3 py-1.5 text-label",
        lg: "gap-2 px-4 py-2 text-body",
      },
      disabled: {
        true: "cursor-not-allowed opacity-50",
        false: "",
      },
    },
    defaultVariants: {
      tone: "neutral",
      size: "md",
      disabled: false,
    },
  },
);

const statusDotVariants = cva("block shrink-0 rounded-pill", {
  variants: {
    tone: {
      positive: "bg-positive",
      warning: "bg-warning",
      critical: "bg-critical",
      info: "bg-primary",
      neutral: "bg-textTertiary",
    },
    size: {
      sm: "h-1.5 w-1.5",
      md: "h-2 w-2",
      lg: "h-2.5 w-2.5",
    },
  },
  defaultVariants: {
    tone: "neutral",
    size: "md",
  },
});

export interface StatusPillProps
  extends Omit<HTMLAttributes<HTMLSpanElement>, "children">,
    VariantProps<typeof statusPillVariants> {
  label: ReactNode;
}

export function StatusPill({
  label,
  tone,
  size,
  disabled = false,
  className,
  ...props
}: StatusPillProps) {
  return (
    <span
      className={cn(statusPillVariants({ tone, size, disabled }), className)}
      aria-disabled={disabled || undefined}
      {...props}
    >
      <span className={statusDotVariants({ tone, size })} aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export { statusPillVariants };
