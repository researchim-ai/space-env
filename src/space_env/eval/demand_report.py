"""Две разные заявки. Ярлык «выучено» не ставится."""
from __future__ import annotations

import gymnasium as gym

from space_env.envs.demand_env import packaged_demand
from space_env.rl.expert import expert_action


def _episode(path: str) -> dict:
    env = gym.make("space_env/DemandShift-v0", scenario=path, max_episode_steps=40)
    try:
        obs, info = env.reset(seed=0)
        done = False
        success = False
        while not done:
            obs, _reward, terminated, truncated, info = env.step(
                expert_action("space_env/DemandShift-v0", obs)
            )
            success = bool(info.get("success", False))
            done = bool(terminated or truncated)
        return {
            "scenario": path,
            "demand": info["demand"],
            "deadline_sec": info["deadline_sec"],
            "delivered_bits": info["delivered_bits"],
            "success": success,
        }
    finally:
        env.close()


def demand_shift_report() -> dict:
    """Эвристика на двух файлах заявки. Поле learned отсутствует."""
    rows = []
    for name in ("demand_deliver.yaml", "demand_late.yaml"):
        rows.append(_episode(str(packaged_demand(name))))
    return {"rows": rows}
