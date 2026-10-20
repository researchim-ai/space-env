"""Случайные роллауты зарегистрированных сред в JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from space_env.rl.curriculum import load_curriculum


def rollout_episode(env: gym.Env, seed: int, policy: Callable[[np.ndarray], int] | None = None) -> list[dict[str, Any]]:
    """Один эпизод. policy(obs) → действие; по умолчанию равномерная выборка."""
    obs, info = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    done = False
    step_index = 0
    while not done:
        action = int(policy(obs)) if policy is not None else int(env.action_space.sample())
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": step_index,
                "observation": np.asarray(obs, dtype=float).ravel().tolist(),
                "action": action,
                "reward": float(reward),
                "next_observation": np.asarray(next_obs, dtype=float).ravel().tolist(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(info.get("success", False)),
            }
        )
        obs = next_obs
        done = bool(terminated or truncated)
        step_index += 1
    return rows


def write_trajectories(
    output_path: str | Path,
    *,
    stages: list[dict[str, Any]] | None = None,
    seed: int = 0,
) -> int:
    """Пишет по строке на шаг. Возвращает число шагов."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = stages if stages is not None else load_curriculum()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for stage_index, stage in enumerate(plan):
            episodes = int(stage.get("episodes", 1))
            for episode in range(episodes):
                env = gym.make(stage["env"], max_episode_steps=int(stage.get("max_episode_steps", 20)))
                try:
                    rows = rollout_episode(env, seed=seed + stage_index * 1000 + episode)
                finally:
                    env.close()
                for row in rows:
                    row["stage"] = stage["name"]
                    row["env"] = stage["env"]
                    row["episode"] = episode
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count
