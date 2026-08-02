import type { HTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const sectionHeaderVariants = cva(
  "flex min-w-0 items-start justify-between gap-card-lg",
  {
    variants: {
      size: {
        compact: "py-2",
        default: "py-card",
        spacious: "py-card-lg",
      },
      alignment: {
        start: "items-start",
        center: "items-center",
      },
    },
    defaultVariants: {
      size: "default",
      alignment: "start",
    },
  },
);

const sectionTitleVariants = cva("text-textPrimary", {
  variants: {
    size: {
      compact: "text-label",
      default: "text-title",
      spacious: "text-page",
    },
  },
  defaultVariants: {
    size: "default",
  },
});

export interface SectionHeaderProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "title">,
    VariantProps<typeof sectionHeaderVariants> {
  title: ReactNode;
  subtitle?: ReactNode;
  rightSlot?: ReactNode;
  headingLevel?: 1 | 2 | 3;
}

export function SectionHeader({
  title,
  subtitle,
  rightSlot,
  headingLevel = 2,
  size,
  alignment,
  className,
  ...props
}: SectionHeaderProps) {
  const Heading = `h${headingLevel}` as const;

  return (
    <div className={cn(sectionHeaderVariants({ size, alignment }), className)} {...props}>
      <div className="min-w-0">
        <Heading className={sectionTitleVariants({ size })}>{title}</Heading>
        {subtitle ? (
          <p className="mt-1 text-body text-textSecondary">{subtitle}</p>
        ) : null}
      </div>
      {rightSlot ? <div className="shrink-0">{rightSlot}</div> : null}
    </div>
  );
}

export { sectionHeaderVariants, sectionTitleVariants };
