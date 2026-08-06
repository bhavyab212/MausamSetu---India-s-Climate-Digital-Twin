"""Abstract Reservoir Environment loader & configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import numpy as np

ENVS_DIR = Path(__file__).resolve().parent


@dataclass
class ReservoirConfig:
    name: str = "Generic Reservoir"
    region: str = "india"
    capacity_tcm: float = 100000.0
    dead_pool_tcm: float = 8000.0
    flood_threshold_tcm: float = 90000.0
    max_discharge_tcm_per_week: float = 20000.0
    demand_peak_week: int = 28
    base_demand_tcm_per_week: float = 1500.0
    peak_demand_tcm_per_week: float = 5000.0
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "demand_met": 1.0,
        "flood_penalty": 5.0,
        "drought_penalty": 3.0,
        "dead_pool_penalty": 10.0,
        "spill_penalty": 10.0,
    })

    def get_weekly_demand(self, week: int) -> float:
        """Sinusoidal weekly demand curve peaking at demand_peak_week."""
        w = week % 52
        phase = (w - self.demand_peak_week) * (2 * 3.14159 / 52)
        factor = 0.5 * (1.0 + np.cos(phase))
        return self.base_demand_tcm_per_week + factor * (self.peak_demand_tcm_per_week - self.base_demand_tcm_per_week)


def load_env_template(template_name: str = "mettur") -> ReservoirConfig:
    """Load reservoir config from yaml template (e.g., 'mettur', 'generic_india', 'template')."""
    path = ENVS_DIR / f"{template_name}.yaml"
    if not path.exists():
        path = ENVS_DIR / "mettur.yaml"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ReservoirConfig(
        name=data.get("name", "Reservoir"),
        region=data.get("region", "india"),
        capacity_tcm=float(data.get("capacity_tcm", 100000.0)),
        dead_pool_tcm=float(data.get("dead_pool_tcm", 8000.0)),
        flood_threshold_tcm=float(data.get("flood_threshold_tcm", 90000.0)),
        max_discharge_tcm_per_week=float(data.get("max_discharge_tcm_per_week", 20000.0)),
        demand_peak_week=int(data.get("demand_peak_week", 28)),
        base_demand_tcm_per_week=float(data.get("base_demand_tcm_per_week", 1500.0)),
        peak_demand_tcm_per_week=float(data.get("peak_demand_tcm_per_week", 5000.0)),
        reward_weights=data.get("reward_weights", {}),
    )


def list_env_templates() -> list[str]:
    """List available template names (stem of .yaml files)."""
    return [p.stem for p in sorted(ENVS_DIR.glob("*.yaml"))]
