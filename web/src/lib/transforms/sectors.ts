import type { Heat, Hydrology, Rupee } from "../api/schemas"
import type { Availability, SemanticTone } from "./overview"

export interface SectorTab {
  id: string
  label: string
  summary: string
  tone: SemanticTone
  selected: boolean
}

export interface SupplyDemandPoint {
  date: string
  supply_mcft_per_day: number | null
  demand_mcft_per_day: number | null
  available: boolean
  unavailableLabel: string | null
}

export interface SectorView {
  hydrology: Hydrology | null
  heat: Heat | null
  rupee: Rupee | null
  tabs: SectorTab[]
  supplyDemand: SupplyDemandPoint[]
  reservoirNetwork: Availability
  waterBalance: Availability
  advisories: Availability
  decisionLog: Availability
}

export function transformSectorHydrology(response: Hydrology | null | undefined): { dates: string[]; inflow_m3_per_day: Array<number | null>; total_m3: number | null; available: boolean; unavailableLabel: string | null } {
  if (!response) return { dates: [], inflow_m3_per_day: [], total_m3: null, available: false, unavailableLabel: "Hydrology unavailable" }
  return { dates: response.dates, inflow_m3_per_day: response.inflow_m3_per_day, total_m3: response.total_m3, available: true, unavailableLabel: null }
}

export function transformSectors(
  hydrology: Hydrology | null | undefined,
  heat: Heat | null | undefined,
  rupee: Rupee | null | undefined,
): SectorView {
  const hydrologyValue = hydrology ?? null
  const heatValue = heat ?? null
  const rupeeValue = rupee ?? null
  const dates = hydrologyValue?.dates ?? heatValue?.dates ?? rupeeValue?.dates ?? []
  const supplyDemand: SupplyDemandPoint[] = dates.map((date, index) => {
    const supply = hydrologyValue?.inflow_m3_per_day[index] ?? null
    return {
      date,
      supply_mcft_per_day: supply,
      demand_mcft_per_day: null,
      available: supply !== null,
      unavailableLabel: supply === null ? "Unavailable" : null,
    }
  })
  const tabs: SectorTab[] = [
    { id: "water", label: "Water security", summary: hydrologyValue ? "Hydrology available" : "Hydrology unavailable", tone: hydrologyValue ? "positive" : "neutral", selected: true },
    { id: "agriculture", label: "Agriculture", summary: "Sector data unavailable", tone: "neutral", selected: false },
    { id: "heat", label: "Heat stress", summary: heatValue ? `${heatValue.heat_stress_days} stress days` : "Heat data unavailable", tone: heatValue && heatValue.heat_stress_days > 0 ? "warning" : "neutral", selected: false },
    { id: "flood", label: "Flood risk", summary: "Flood data unavailable", tone: "neutral", selected: false },
  ]
  return {
    hydrology: hydrologyValue,
    heat: heatValue,
    rupee: rupeeValue,
    tabs,
    supplyDemand,
    reservoirNetwork: { available: false, unavailableLabel: "Reservoir data unavailable from API" },
    waterBalance: { available: false, unavailableLabel: "Water balance unavailable from API" },
    advisories: { available: false, unavailableLabel: "Stakeholder advisories unavailable from API" },
    decisionLog: { available: false, unavailableLabel: "Decision log unavailable from API" },
  }
}
