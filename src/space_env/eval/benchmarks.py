"""Прогон случайной политики и сводка по средам."""
from __future__ import annotations

import gymnasium as gym
import pandas as pd

# Потолок шагов в общем прогоне. Длиннее этого горизонта оценка не идёт,
# даже если вызывающий передал больший max_episode_steps.
_SUITE_STEPS = {
    "space_env/MissionPlanner-v0": 4,
    "space_env/ResourceManager-v0": 4,
    "space_env/AnomalyResponse-v0": 4,
    "space_env/ConjunctionAssessment-v0": 2,
    "space_env/Constellation-v0": 2,
    "space_env/ConstellationOps-v0": 4,
    "space_env/SentinelAcquisition-v0": 2,
    "space_env/LandsatAcquisition-v0": 2,
    "space_env/SentinelPhasing-v0": 2,
    "space_env/WorkPlan-v0": 2,
    "space_env/ConjunctionQueue-v0": 1,
    "space_env/DecayWatch-v0": 1,
    "space_env/DecayWatchHigh-v0": 1,
    "space_env/PhaseBurn-v0": 1,
    "space_env/Reroute-v0": 1,
    "space_env/FleetShift-v0": 4,
    "space_env/RendezvousHold-v0": 1,
    "space_env/PriorityCoverage-v0": 1,
    "space_env/PassFault-v0": 6,
    "space_env/DebrisAvoid-v0": 1,
    "space_env/Disposal-v0": 1,
    "space_env/StationContention-v0": 1,
    "space_env/OrderBook-v0": 1,
    "space_env/SentinelPassFault-v0": 8,
    "space_env/CubesatAcquisition-v0": 2,
    "space_env/DayMission-v0": 1,
}


def run_benchmark(
    env_id: str,
    *,
    episodes: int = 3,
    max_episode_steps: int = 20,
    seed: int = 0,
) -> dict[str, float | str | int]:
    """Случайный агент. success берётся из info последнего шага эпизода."""
    rows: list[dict[str, float | int | bool | str]] = []
    for episode in range(episodes):
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        try:
            _obs, _info = env.reset(seed=seed + episode)
            done = False
            total = 0.0
            steps = 0
            success = False
            while not done:
                _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                total += float(reward)
                steps += 1
                success = bool(info.get("success", False))
                done = bool(terminated or truncated)
        finally:
            env.close()
        rows.append({"env": env_id, "episode": episode, "reward": total, "steps": steps, "success": success})
    frame = pd.DataFrame(rows)
    return {
        "env": env_id,
        "episodes": episodes,
        "mean_reward": float(frame["reward"].mean()),
        "success_rate": float(frame["success"].mean()),
        "mean_steps": float(frame["steps"].mean()),
    }


def run_registered_benchmarks(
    *,
    episodes: int = 1,
    max_episode_steps: int = 12,
    seed: int = 0,
) -> list[dict[str, float | str | int]]:
    """Все идентификаторы space_env. Шаг эпизода не длиннее потолка этой среды."""
    from space_env.envs import registered_env_ids

    rows = []
    for env_id in registered_env_ids():
        cap = _SUITE_STEPS.get(env_id, max_episode_steps)
        rows.append(
            run_benchmark(
                env_id,
                episodes=episodes,
                max_episode_steps=min(max_episode_steps, cap),
                seed=seed,
            )
        )
    return rows
