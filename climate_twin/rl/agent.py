"""RL Agent training, baseline comparison, and policy persistence (PPO via Stable-Baselines3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

from .env import ReservoirEnv
from .envs.base_env import load_env_template, ReservoirConfig

POLICIES_DIR = Path(__file__).resolve().parent / "policies"


def run_baseline_policy(
    env: ReservoirEnv,
    policy_type: str = "inflow",  # 'inflow', 'fixed_demand'
) -> dict[str, Any]:
    """Evaluates a heuristic baseline policy on the environment."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    demands_met = []
    flood_events = 0
    dead_pool_events = 0
    history = []

    while not done:
        curr_week = info["week"]
        inflow = float(env.inflow_series[min(51, curr_week)])
        demand = float(env.config.get_weekly_demand(curr_week))
        max_d = env.config.max_discharge_tcm_per_week

        if policy_type == "inflow":
            # Naive: release = inflow
            action = min(1.0, inflow / max_d)
        else:
            # Fixed demand: release = current demand
            action = min(1.0, demand / max_d)

        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        done = terminated or truncated

        total_reward += reward
        demands_met.append(info["demand_met_pct"])
        if info["is_flood"]:
            flood_events += 1
        if info["is_dead_pool"]:
            dead_pool_events += 1

        history.append({
            "week": curr_week,
            "inflow": info["inflow_tcm"],
            "release": info["release_tcm"],
            "demand": info["demand_tcm"],
            "storage": info["storage_tcm"],
            "reward": reward,
        })

    return {
        "policy": f"Baseline ({policy_type})",
        "total_reward": total_reward,
        "mean_demand_met_pct": float(np.mean(demands_met)),
        "flood_events": flood_events,
        "dead_pool_events": dead_pool_events,
        "history": history,
    }


def train_rl_agent(
    env_name: str = "mettur",
    agent_name: str = "ppo_mettur_v1",
    total_timesteps: int = 10000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    parent_name: str = "",
    on_step_callback: Callable[[int, float], None] | None = None,
) -> tuple[PPO, dict[str, Any]]:
    """Train PPO RL agent on specified reservoir environment."""
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    env = ReservoirEnv(env_name)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=512,
        batch_size=64,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps)

    # Evaluate trained RL agent
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    demands_met = []
    flood_events = 0
    dead_pool_events = 0
    history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        demands_met.append(info["demand_met_pct"])
        if info["is_flood"]:
            flood_events += 1
        if info["is_dead_pool"]:
            dead_pool_events += 1

        history.append({
            "week": info["week"],
            "inflow": info["inflow_tcm"],
            "release": info["release_tcm"],
            "demand": info["demand_tcm"],
            "storage": info["storage_tcm"],
            "reward": reward,
        })

    metrics = {
        "policy": f"PPO Agent ({agent_name})",
        "total_reward": total_reward,
        "mean_demand_met_pct": float(np.mean(demands_met)) if demands_met else 0.0,
        "flood_events": flood_events,
        "dead_pool_events": dead_pool_events,
        "history": history,
        "timesteps": total_timesteps,
    }

    # Save policy + meta lineage
    save_dir = POLICIES_DIR / agent_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "ppo_policy"))

    meta = {
        "agent_name": agent_name,
        "env_name": env_name,
        "parent_name": parent_name,
        "total_timesteps": total_timesteps,
        "total_reward": total_reward,
        "demand_met_pct": float(np.mean(demands_met)),
        "flood_events": flood_events,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return model, metrics


def list_saved_policies() -> list[dict[str, Any]]:
    """List saved RL policies and metadata."""
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(POLICIES_DIR.iterdir()):
        if d.is_dir():
            meta_p = d / "meta.json"
            if meta_p.exists():
                try:
                    out.append(json.loads(meta_p.read_text()))
                except Exception:
                    pass
    return out
