"""Случайные роллауты зарегистрированных сред в JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from space_env.rl.curriculum import load_curriculum


def _json_action(action: Any) -> int | list[int]:
    values = np.atleast_1d(np.asarray(action, dtype=int)).ravel()
    if values.size == 1:
        return int(values[0])
    return [int(item) for item in values]


def rollout_episode(env: gym.Env, seed: int, policy: Callable[[np.ndarray], Any] | None = None) -> list[dict[str, Any]]:
    """Один эпизод. policy(obs) → действие; по умолчанию равномерная выборка."""
    obs, info = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    done = False
    step_index = 0
    while not done:
        action = policy(obs) if policy is not None else env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": step_index,
                "observation": np.asarray(obs, dtype=float).ravel().tolist(),
                "action": _json_action(action),
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
    use_expert: bool = False,
) -> int:
    """Пишет по строке на шаг. Возвращает число шагов. use_expert берёт эвристику вместо равномерной выборки."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = stages if stages is not None else load_curriculum()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for stage_index, stage in enumerate(plan):
            episodes = int(stage.get("episodes", 1))
            for episode in range(episodes):
                reserved = {"name", "env", "episodes", "max_episode_steps"}
                env = gym.make(
                    stage["env"],
                    max_episode_steps=int(stage.get("max_episode_steps", 20)),
                    **{key: value for key, value in stage.items() if key not in reserved},
                )
                env_id = env.spec.id if env.spec is not None else str(stage["env"])

                def _expert(obs: np.ndarray, env_name: str = env_id) -> int:
                    from space_env.rl.expert import expert_action

                    return expert_action(env_name, obs)

                try:
                    rows = rollout_episode(
                        env,
                        seed=seed + stage_index * 1000 + episode,
                        policy=_expert if use_expert else None,
                    )
                finally:
                    env.close()
                for row in rows:
                    row["stage"] = stage["name"]
                    row["env"] = stage["env"]
                    row["episode"] = episode
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count
