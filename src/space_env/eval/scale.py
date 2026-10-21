"""Успех эвристики против числа аппаратов в уже написанных задачах."""
from __future__ import annotations

import gymnasium as gym

from space_env.rl.expert import expert_action


def _success(env_id: str, *, steps: int, **kwargs) -> bool:
    env = gym.make(env_id, max_episode_steps=steps, **kwargs)
    try:
        obs, _info = env.reset(seed=0)
        done = False
        last = {}
        while not done:
            obs, _reward, terminated, truncated, last = env.step(expert_action(env_id, obs))
            done = bool(terminated or truncated)
        return bool(last.get("success", False))
    finally:
        env.close()


def scale_report() -> list[dict]:
    """Спор за станцию, пакет заявок и смена при нескольких размерах группировки."""
    rows: list[dict] = []
    for count in (2, 4, 6):
        rows.append({"task": "station", "count": count, "success": _success("space_env/StationContention-v0", steps=1, count=count)})
    for count in (2, 3, 4):
        rows.append({"task": "orders", "count": count, "success": _success("space_env/OrderBook-v0", steps=1, count=count)})
    for count in (4, 6):
        rows.append(
            {
                "task": "shift",
                "count": count,
                "success": _success("space_env/FleetShift-v0", steps=count + 2, count=count),
            }
        )
    return rows
