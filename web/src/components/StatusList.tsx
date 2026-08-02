import type { HTMLAttributes, ReactNode } from "react";

import type { SemanticTone } from "@/components/InlineLegend";
import { cn } from "@/lib/utils";

const toneClasses: Record<SemanticTone, string> = {
  positive: "bg-positive text-positive",
  warning: "bg-warning text-warning",
  critical: "bg-critical text-critical",
  info: "bg-primary text-primary",
  neutral: "bg-textTertiary text-textSecondary",
};

export interface StatusListItem {
  id: string;
  label: ReactNode;
  value: ReactNode;
  tone: SemanticTone;
  icon?: ReactNode;
  detail?: ReactNode;
}

export interface StatusListProps extends HTMLAttributes<HTMLUListElement> {
  items: readonly StatusListItem[];
  divided?: boolean;
  density?: "compact" | "default";
  markerPosition?: "start" | "end";
}

export function StatusList({
  items,
  divided = true,
  density = "default",
  markerPosition = "end",
  className,
  ...props
}: StatusListProps) {
  return (
    <ul className={cn(divided && "divide-y divide-divider", className)} {...props}>
      {items.map((item) => (
        <li
          key={item.id}
          className={cn(
            "flex items-center text-label",
            density === "compact" ? "min-h-0 gap-2 py-0.5" : "min-h-9 gap-3 py-1",
          )}
        >
          {markerPosition === "start" ? (
            <span className={cn("h-2 w-2 shrink-0 rounded-pill", toneClasses[item.tone].split(" ")[0])} aria-hidden="true" />
          ) : null}
          {item.icon ? <span className="text-textSecondary">{item.icon}</span> : null}
          <span className="min-w-0 flex-1 truncate text-textPrimary">{item.label}</span>
          {item.detail ? <span className="truncate text-caption text-textTertiary">{item.detail}</span> : null}
          <span className={cn("shrink-0 text-caption", toneClasses[item.tone].split(" ")[1])}>{item.value}</span>
          {markerPosition === "end" ? (
            <span className={cn("h-2 w-2 shrink-0 rounded-pill", toneClasses[item.tone].split(" ")[0])} aria-hidden="true" />
          ) : null}
        </li>
      ))}
    </ul>
  );
}
