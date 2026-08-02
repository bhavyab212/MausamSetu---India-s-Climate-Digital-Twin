import type { HTMLAttributes, ReactNode } from "react";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

export type SemanticTone =
  | "positive"
  | "warning"
  | "critical"
  | "info"
  | "neutral";

export interface InlineLegendItem {
  id: string;
  label: ReactNode;
  tone: SemanticTone;
  value?: ReactNode;
}

const legendMarkerVariants = cva("h-2 w-2 shrink-0 rounded-pill", {
  variants: {
    tone: {
      positive: "bg-positive",
      warning: "bg-warning",
      critical: "bg-critical",
      info: "bg-primary",
      neutral: "bg-textTertiary",
    },
  },
});

export interface InlineLegendProps
  extends Omit<HTMLAttributes<HTMLUListElement>, "children"> {
  items: readonly InlineLegendItem[];
}

export function InlineLegend({ items, className, ...props }: InlineLegendProps) {
  return (
    <ul
      className={cn("flex flex-wrap items-center gap-column-gap", className)}
      {...props}
    >
      {items.map((item) => (
        <li key={item.id} className="inline-flex items-center gap-2 text-caption">
          <span
            className={legendMarkerVariants({ tone: item.tone })}
            aria-hidden="true"
          />
          <span className="text-textSecondary">{item.label}</span>
          {item.value ? (
            <span className="font-mono text-textPrimary tabular-nums">
              {item.value}
            </span>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

export { legendMarkerVariants };
