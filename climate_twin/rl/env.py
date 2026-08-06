"""Gymnasium Reservoir Decision Environment.

State:
  [current_storage_pct, recent_inflow, forecast_p10, forecast_p50, forecast_p90, season, downstream_demand]

Action:
  release_volume_ratio in [0.0, 1.0]

Reward:
  + meeting demand - flood_penalty - drought_penalty - dead_pool_penalty - spill_penalty
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .envs.base_env import ReservoirConfig, load_env_template


class ReservoirEnv(gym.Env):
    """Gym-style Reservoir decision environment operating under forecast uncertainty."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: ReservoirConfig | str = "mettur",
        inflow_series: np.ndarray | None = None,
        forecast_model: Any | None = None,
        scenario: str = "normal",  # 'normal', 'dry', 'flood'
    ):
        super().__init__()

        if isinstance(config, str):
            self.config = load_env_template(config)
        else:
            self.config = config

        self.scenario = scenario
        self.forecast_model = forecast_model

        # Generate synthetic or realistic weekly inflow series (52 weeks)
        if inflow_series is not None:
            self.inflow_series = inflow_series
        else:
            self.inflow_series = self._generate_inflow_series(scenario)

        # Observation space: 7 continuous signals in [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        # Action space: continuous release ratio [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Environment state
        self.current_week = 0
        self.storage_tcm = self.config.capacity_tcm * 0.5  # Start at 50% capacity
        self.recent_inflows = []

    def _generate_inflow_series(self, scenario: str) -> np.ndarray:
        """Generate weekly inflow series (TCM/week) for 52 weeks."""
        weeks = np.arange(52)
        peak_w = self.config.demand_peak_week
        # Monsoon peak inflow
        base_inflow = self.config.capacity_tcm * 0.02
        peak_inflow = self.config.capacity_tcm * 0.15

        phase = (weeks - peak_w) * (2 * math.pi / 52)
        inflow = base_inflow + 0.5 * (1.0 + np.cos(phase)) * (peak_inflow - base_inflow)

        # Scenario adjustments
        if scenario == "dry":
            inflow *= 0.55
        elif scenario == "flood":
            inflow *= 1.65

        # Add stochastic noise
        np.random.seed(42 if scenario == "normal" else (101 if scenario == "dry" else 202))
        noise = np.random.normal(1.0, 0.15, size=52)
        return np.maximum(100.0, inflow * noise)

    def _get_forecast(self, week: int) -> tuple[float, float, float]:
        """Generate p10, p50, p90 inflow forecast for next 4-week horizon."""
        horizon_inflow = np.mean(self.inflow_series[week:min(52, week + 4)])
        max_inf = self.config.max_discharge_tcm_per_week * 2.0

        p50 = min(1.0, horizon_inflow / max_inf)
        p10 = max(0.0, p50 - 0.1)
        p90 = min(1.0, p50 + 0.1)
        return float(p10), float(p50), float(p90)

    def _get_obs(self) -> np.ndarray:
        cap = self.config.capacity_tcm
        storage_pct = self.storage_tcm / cap
        
        recent = np.mean(self.recent_inflows[-4:]) if self.recent_inflows else self.inflow_series[self.current_week]
        recent_norm = min(1.0, recent / (self.config.max_discharge_tcm_per_week * 2.0))

        p10, p50, p90 = self._get_forecast(self.current_week)
        season_norm = self.current_week / 52.0

        demand = self.config.get_weekly_demand(self.current_week)
        demand_norm = min(1.0, demand / self.config.max_discharge_tcm_per_week)

        return np.array([storage_pct, recent_norm, p10, p50, p90, season_norm, demand_norm], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_week = 0
        self.storage_tcm = self.config.capacity_tcm * 0.5
        self.recent_inflows = [self.inflow_series[0]]

        if options and "scenario" in options:
            self.scenario = options["scenario"]
            self.inflow_series = self._generate_inflow_series(self.scenario)

        return self._get_obs(), {"week": self.current_week, "storage_tcm": self.storage_tcm}

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act_val = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        act_val = np.clip(act_val, 0.0, 1.0)

        release_tcm = act_val * self.config.max_discharge_tcm_per_week
        inflow_tcm = float(self.inflow_series[self.current_week])
        self.recent_inflows.append(inflow_tcm)

        demand_tcm = self.config.get_weekly_demand(self.current_week)
        cap = self.config.capacity_tcm
        w = self.config.reward_weights

        # Balance equation
        new_storage = self.storage_tcm + inflow_tcm - release_tcm

        overflow_tcm = max(0.0, new_storage - cap)
        spill_tcm = overflow_tcm
        new_storage = min(cap, new_storage)

        dead_pool_deficit = max(0.0, self.config.dead_pool_tcm - new_storage)
        self.storage_tcm = max(0.0, new_storage)

        # Rewards & Penalties
        demand_met_ratio = min(1.0, release_tcm / max(1.0, demand_tcm))
        r_demand = w.get("demand_met", 1.0) * demand_met_ratio

        flood_penalty = w.get("flood_penalty", 5.0) * (max(0.0, self.storage_tcm - self.config.flood_threshold_tcm) / cap)
        drought_penalty = w.get("drought_penalty", 3.0) * (dead_pool_deficit / cap)
        spill_penalty = w.get("spill_penalty", 10.0) * (spill_tcm / max(1.0, self.config.max_discharge_tcm_per_week))

        reward = r_demand - flood_penalty - drought_penalty - spill_penalty

        self.current_week += 1
        terminated = self.current_week >= 52
        truncated = False

        info = {
            "week": self.current_week,
            "storage_tcm": self.storage_tcm,
            "inflow_tcm": inflow_tcm,
            "release_tcm": release_tcm,
            "demand_tcm": demand_tcm,
            "demand_met_pct": demand_met_ratio * 100.0,
            "is_flood": self.storage_tcm > self.config.flood_threshold_tcm,
            "is_dead_pool": self.storage_tcm < self.config.dead_pool_tcm,
            "spill_tcm": spill_tcm,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
