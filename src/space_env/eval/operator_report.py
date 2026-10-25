"""Сравнение случайной политики и эвристики на полных сутках. Ярлык «выучено» не ставится."""
from __future__ import annotations

import gymnasium as gym

from space_env.rl.expert import expert_action


def _episode(seed: int, use_expert: bool) -> tuple[float, bool]:
    env = gym.make("space_env/OperatorShift-v0", until="full", max_episode_steps=2000)
    try:
        obs, _info = env.reset(seed=seed)
        done = False
        total = 0.0
        success = False
        env_id = env.spec.id if env.spec is not None else "space_env/OperatorShift-v0"
        while not done:
            action = expert_action(env_id, obs) if use_expert else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            success = bool(info.get("success", False))
            done = bool(terminated or truncated)
        return total, success
    finally:
        env.close()


def operator_shift_report(*, train_seed: int = 0, holdout_seed: int = 1) -> dict:
    """
    Полные сутки на сиде обучения и на отложенном сиде: раздача зеркал и все дела.
    Поле learned отсутствует: один прогон не доказывает навык.
    """
    rows = []
    for split, seed in (("train", train_seed), ("holdout", holdout_seed)):
        for policy, use_expert in (("random", False), ("heuristic", True)):
            reward, success = _episode(seed, use_expert)
            rows.append({"split": split, "policy": policy, "seed": seed, "reward": reward, "success": success})
    return {"until": "full", "rows": rows}
