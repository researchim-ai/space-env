"""Два файла заявок. Ярлык «выучено» не ставится: здесь нет отложенного обучения."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.envs.tasking_env import packaged_tasking
from space_env.rl.expert import expert_action


def _episode(path: str, *, expert: bool, seed: int = 0) -> bool:
    env = gym.make("space_env/TaskingShift-v0", scenario=path, max_episode_steps=40)
    try:
        obs, info = env.reset(seed=seed)
        assert "correct_action" not in info
        rng = np.random.default_rng(seed)
        done = False
        success = False
        while not done:
            action = expert_action("space_env/TaskingShift-v0", obs) if expert else int(rng.integers(0, 3))
            obs, _reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("success", False))
            done = bool(terminated or truncated)
        return success
    finally:
        env.close()


def tasking_shift_report() -> dict:
    """Жадный план и случайные действия на двух файлах. Поля learned нет."""
    rows = []
    for name in ("tasking_deliver.yaml", "tasking_late.yaml"):
        path = str(packaged_tasking(name))
        random_hits = sum(_episode(path, expert=False, seed=seed) for seed in range(4))
        rows.append(
            {
                "scenario": name,
                "greedy_success": _episode(path, expert=True),
                "random_success_rate": random_hits / 4.0,
            }
        )
    return {"rows": rows}
