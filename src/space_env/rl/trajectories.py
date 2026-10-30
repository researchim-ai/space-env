"""Случайные роллауты зарегистрированных сред в JSONL."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from space_env.rl.curriculum import load_curriculum


def _json_value(value: Any) -> Any:
    """Preserve structured observations rather than flattening entity dimensions."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"неподдерживаемое значение наблюдения: {type(value).__name__}")


def rollout_episode(env: gym.Env, seed: int, policy: Callable[[Any], Any] | None = None) -> list[dict[str, Any]]:
    """Один эпизод. policy(obs) → действие; по умолчанию равномерная выборка."""
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    rows: list[dict[str, Any]] = []
    done = False
    step_index = 0
    while not done:
        action = policy(obs) if policy is not None else env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "schema": 2,
                "seed": int(seed),
                "step": step_index,
                "observation": _json_value(obs),
                "action": _json_value(action),
                "reward": float(reward),
                "next_observation": _json_value(next_obs),
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
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
    return count
