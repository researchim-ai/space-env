"""Сравнение случайной политики и эвристики на одном наборе сидов."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.rl.expert import expert_action


def _episode(env: gym.Env, seed: int, use_expert: bool) -> tuple[float, bool]:
    obs, _info = env.reset(seed=seed)
    done = False
    total = 0.0
    success = False
    env_id = env.spec.id if env.spec is not None else ""
    while not done:
        action = expert_action(env_id, obs) if use_expert else int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        success = bool(info.get("success", False))
        done = bool(terminated or truncated)
    return total, success


def compare_policies(
    env_id: str,
    *,
    episodes: int = 5,
    max_episode_steps: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Доля успехов и средняя награда случайного агента и эвристики."""
    random_success = []
    expert_success = []
    random_reward = []
    expert_reward = []
    for episode in range(episodes):
        for use_expert, success_box, reward_box in (
            (False, random_success, random_reward),
            (True, expert_success, expert_reward),
        ):
            env = gym.make(env_id, max_episode_steps=max_episode_steps)
            try:
                reward, success = _episode(env, seed + episode, use_expert)
            finally:
                env.close()
            success_box.append(success)
            reward_box.append(reward)
    return {
        "random_success_rate": float(np.mean(random_success)),
        "expert_success_rate": float(np.mean(expert_success)),
        "random_mean_reward": float(np.mean(random_reward)),
        "expert_mean_reward": float(np.mean(expert_reward)),
    }
