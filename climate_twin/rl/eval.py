"""Evaluation across multiple environments & SSP stress testing (dry-year / flood-year)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .agent import POLICIES_DIR, run_baseline_policy
from .env import ReservoirEnv
from .envs.base_env import list_env_templates


def evaluate_agent_across_envs(
    policy_name: str = "ppo_mettur_v1",
    env_names: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluates a trained policy across all environment templates & scenarios (generalization test)."""
    from stable_baselines3 import PPO

    policy_path = POLICIES_DIR / policy_name / "ppo_policy.zip"
    use_model = policy_path.exists()
    model = PPO.load(str(policy_path)) if use_model else None

    envs = env_names or list_env_templates()
    scens = scenarios or ["normal", "dry", "flood"]

    results = []
    for env_name in envs:
        for scen in scens:
            env = ReservoirEnv(env_name, scenario=scen)

            if model is not None:
                obs, info = env.reset()
                done = False
                tot_reward = 0.0
                demands_met = []
                floods = 0
                dead_pools = 0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    tot_reward += reward
                    demands_met.append(info["demand_met_pct"])
                    if info["is_flood"]:
                        floods += 1
                    if info["is_dead_pool"]:
                        dead_pools += 1

                results.append({
                    "Policy": policy_name,
                    "Environment": env_name,
                    "Scenario": scen,
                    "Reward": round(tot_reward, 2),
                    "Demand Met (%)": round(float(np.mean(demands_met)) if demands_met else 0.0, 1),
                    "Flood Events": floods,
                    "Dead Pool Events": dead_pools,
                })

            # Also evaluate Naive Baseline for direct contrast
            env_base = ReservoirEnv(env_name, scenario=scen)
            base_res = run_baseline_policy(env_base, policy_type="inflow")
            results.append({
                "Policy": "Naive (release=inflow)",
                "Environment": env_name,
                "Scenario": scen,
                "Reward": round(base_res["total_reward"], 2),
                "Demand Met (%)": round(base_res["mean_demand_met_pct"], 1),
                "Flood Events": base_res["flood_events"],
                "Dead Pool Events": base_res["dead_pool_events"],
            })

    return pd.DataFrame(results)
