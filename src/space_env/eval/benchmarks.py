"""Прогон случайной политики и сводка по средам."""
from __future__ import annotations

import gymnasium as gym
import pandas as pd


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
    episodes: int = 2,
    max_episode_steps: int = 12,
    seed: int = 0,
) -> list[dict[str, float | str | int]]:
    """Четыре зарегистрированные среды подряд."""
    env_ids = [
        "space_env/ResourceManager-v0",
        "space_env/AnomalyResponse-v0",
        "space_env/ConjunctionAssessment-v0",
        "space_env/MissionPlanner-v0",
    ]
    return [
        run_benchmark(env_id, episodes=episodes, max_episode_steps=max_episode_steps, seed=seed)
        for env_id in env_ids
    ]
