import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const glassCardVariants = cva(
  "rounded-card border border-cardBorder transition-all duration-fast",
  {
    variants: {
      variant: {
        default: "bg-card shadow-elevation-card backdrop-blur-card",
        muted: "bg-card-muted",
        floating: "bg-card shadow-elevation-floating backdrop-blur-card",
        interactive:
          "bg-card shadow-elevation-card backdrop-blur-card hover:border-primary focus-within:border-primary focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2",
      },
      padding: {
        none: "p-0",
        default: "p-card",
        spacious: "p-card-lg",
      },
      disabled: {
        true: "pointer-events-none cursor-not-allowed opacity-50",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      padding: "default",
      disabled: false,
    },
  },
);

export interface GlassCardProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "children">,
    VariantProps<typeof glassCardVariants> {
  children: ReactNode;
}

export function GlassCard({
  children,
  variant,
  padding,
  disabled = false,
  className,
  ...props
}: GlassCardProps) {
  return (
    <div
      className={cn(glassCardVariants({ variant, padding, disabled }), className)}
      aria-disabled={disabled || undefined}
      {...props}
    >
      {children}
    </div>
  );
}

export { glassCardVariants };
