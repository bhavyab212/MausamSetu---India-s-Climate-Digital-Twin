"use client"

import * as React from "react"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

export interface DropdownOption {
  value: string
  label: string
  disabled?: boolean
}

export interface DropdownProps {
  label: string
  options: readonly DropdownOption[]
  value?: string
  defaultValue?: string
  placeholder?: string
  onValueChange?: (value: string) => void
  disabled?: boolean
  required?: boolean
  name?: string
  id?: string
  className?: string
}

export function Dropdown({
  label,
  options,
  value,
  defaultValue,
  placeholder = "Select an option",
  onValueChange,
  disabled,
  required,
  name,
  id,
  className,
}: DropdownProps) {
  const generatedId = React.useId()
  const triggerId = id ?? generatedId
  const labelId = `${triggerId}-label`

  return (
    <div className={cn("grid gap-2", className)}>
      <label id={labelId} htmlFor={triggerId} className="text-label text-textPrimary">
        {label}
        {required ? <span className="ml-1 text-critical" aria-hidden="true">*</span> : null}
      </label>
      <Select
        value={value}
        defaultValue={defaultValue}
        onValueChange={onValueChange}
        disabled={disabled}
        required={required}
        name={name}
      >
        <SelectTrigger id={triggerId} aria-labelledby={labelId}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
