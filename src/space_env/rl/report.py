"""Короткий отчёт: эвристика, случайная политика и обученный softmax на одном семени."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.eval.holdout import compare_policies
from space_env.rl.reinforce import train_multidiscrete_reinforce


def _greedy_return(env_id: str, weights: list[np.ndarray], bias: list[np.ndarray], seed: int, steps: int) -> tuple[float, bool]:
    env = gym.make(env_id, max_episode_steps=steps)
    try:
        obs, _info = env.reset(seed=seed)
        done = False
        total = 0.0
        success = False
        while not done:
            obs_v = np.asarray(obs, dtype=float).ravel()
            action = np.zeros(len(weights), dtype=int)
            for index, (matrix, offset) in enumerate(zip(weights, bias)):
                logits = matrix @ obs_v + offset
                action[index] = int(np.argmax(logits))
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            success = bool(info.get("success", False))
            done = bool(terminated or truncated)
        return total, success
    finally:
        env.close()


def training_report(
    env_id: str,
    *,
    episodes: int = 8,
    max_episode_steps: int = 8,
    seed: int = 0,
    eval_episodes: int = 2,
) -> dict[str, float | bool]:
    """
    Учит политику и кладёт рядом числа эвристики и случайной политики.
    learned сравнивает жадный успех на отложенных эпизодах с успехом эвристики.
    Хвост обучения со случайной наградой не сравнивается.
    """
    trained = train_multidiscrete_reinforce(
        env_id,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    held = compare_policies(env_id, episodes=eval_episodes, max_episode_steps=max_episode_steps, seed=seed)
    greedy_rewards = []
    greedy_success = []
    for episode in range(eval_episodes):
        reward, success = _greedy_return(
            env_id,
            trained["weights"],
            trained["bias"],
            seed + 100 + episode,
            max_episode_steps,
        )
        greedy_rewards.append(reward)
        greedy_success.append(success)
    random_mean = float(held["random_mean_reward"])
    trained_tail = float(trained["mean_last"])
    held_out = float(np.mean(greedy_success))
    return {
        "expert_success_rate": float(held["expert_success_rate"]),
        "random_mean_reward": random_mean,
        "trained_mean_last": trained_tail,
        "trained_success_rate": held_out,
        "held_out_success_rate": held_out,
        "learned": held_out >= float(held["expert_success_rate"]),
    }
