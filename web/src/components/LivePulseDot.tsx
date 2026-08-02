import { cn } from "@/lib/utils"

export interface LivePulseDotProps {
  label?: string
  className?: string
}

export function LivePulseDot({ label = "Live", className }: LivePulseDotProps) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span
        className="h-2 w-2 rounded-full bg-positive motion-safe:animate-pulse motion-reduce:animate-none"
        aria-hidden="true"
      />
      <span className="sr-only">{label}</span>
    </span>
  )
}
