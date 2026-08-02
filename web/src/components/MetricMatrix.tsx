import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export interface MetricMatrixItem {
  label: string;
  value: number | string;
  unit: string;
}

export interface MetricMatrixProps extends HTMLAttributes<HTMLDListElement> {
  items: readonly MetricMatrixItem[];
  columns?: 2 | 3;
}

export function MetricMatrix({ items, columns = 3, className, ...props }: MetricMatrixProps) {
  return (
    <dl className={cn("grid gap-x-card gap-y-row-gap", columns === 2 ? "grid-cols-2" : "grid-cols-3", className)} {...props}>
      {items.map((item) => (
        <div key={item.label}>
          <dt className="text-caption text-textSecondary">{item.label}</dt>
          <dd className="mt-1 flex items-baseline gap-1 font-mono text-title text-textPrimary tabular-nums">
            {item.value}
            <span className="font-sans text-caption text-textSecondary">{item.unit}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
