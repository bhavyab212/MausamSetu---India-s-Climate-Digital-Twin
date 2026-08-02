"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const togglePillVariants = cva(
  "inline-flex min-h-target items-center justify-center gap-2 rounded-pill border px-4 text-label transition-colors duration-fast focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      pressed: {
        true: "border-primary bg-primary-soft text-primary-strong",
        false:
          "border-cardBorder bg-card text-textSecondary hover:border-primary hover:text-primary",
      },
      size: {
        sm: "min-h-control px-3",
        md: "min-h-target px-4",
      },
    },
    defaultVariants: { pressed: false, size: "md" },
  }
)

export interface TogglePillProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onChange">,
    Omit<VariantProps<typeof togglePillVariants>, "pressed"> {
  pressed?: boolean
  defaultPressed?: boolean
  onPressedChange?: (pressed: boolean) => void
}

export const TogglePill = React.forwardRef<HTMLButtonElement, TogglePillProps>(
  (
    {
      pressed,
      defaultPressed = false,
      onPressedChange,
      onClick,
      size,
      className,
      type = "button",
      ...props
    },
    ref
  ) => {
    const [internalPressed, setInternalPressed] = React.useState(defaultPressed)
    const isPressed = pressed ?? internalPressed

    return (
      <button
        ref={ref}
        type={type}
        aria-pressed={isPressed}
        className={cn(togglePillVariants({ pressed: isPressed, size }), className)}
        onClick={(event) => {
          onClick?.(event)
          if (event.defaultPrevented) return
          const nextPressed = !isPressed
          if (pressed === undefined) setInternalPressed(nextPressed)
          onPressedChange?.(nextPressed)
        }}
        {...props}
      />
    )
  }
)

TogglePill.displayName = "TogglePill"
