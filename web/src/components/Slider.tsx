"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

export interface SliderProps
  extends Omit<
    React.InputHTMLAttributes<HTMLInputElement>,
    "type" | "value" | "defaultValue" | "onChange" | "min" | "max" | "step"
  > {
  label: string
  value?: number
  defaultValue?: number
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: number) => void
  formatValue?: (value: number) => string
}

export const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      id,
      label,
      value,
      defaultValue,
      min = 0,
      max = 100,
      step = 1,
      onValueChange,
      formatValue = String,
      className,
      disabled,
      ...props
    },
    ref
  ) => {
    const generatedId = React.useId()
    const inputId = id ?? generatedId
    const initialValue = defaultValue ?? min
    const [internalValue, setInternalValue] = React.useState(initialValue)
    const currentValue = value ?? internalValue

    return (
      <div className={cn("grid gap-2", className)}>
        <div className="flex items-center justify-between gap-3">
          <label htmlFor={inputId} className="text-label text-textPrimary">
            {label}
          </label>
          <output htmlFor={inputId} className="font-mono text-caption text-textSecondary">
            {formatValue(currentValue)}
          </output>
        </div>
        <input
          ref={ref}
          id={inputId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={currentValue}
          disabled={disabled}
          aria-valuetext={formatValue(currentValue)}
          className="h-control w-full cursor-pointer accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          onChange={(event) => {
            const nextValue = event.currentTarget.valueAsNumber
            if (value === undefined) setInternalValue(nextValue)
            onValueChange?.(nextValue)
          }}
          {...props}
        />
      </div>
    )
  }
)

Slider.displayName = "Slider"
