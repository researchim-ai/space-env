"""Линейный REINFORCE для дискретных сред Gymnasium. Без внешних RL-библиотек."""
from __future__ import annotations

import gymnasium as gym
import numpy as np


def train_linear_reinforce(
    env_id: str,
    *,
    episodes: int = 40,
    max_episode_steps: int = 20,
    seed: int = 0,
    lr: float = 0.05,
    gamma: float = 0.99,
) -> dict[str, float | list[float]]:
    """
    Политика softmax(W obs + b). Обновление — REINFORCE со средним возвратом шага как baseline.
    Возвращает сумму наград по эпизодам и среднее по последним пяти.
    """
    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    obs_n = int(np.prod(env.observation_space.shape))
    n_act = int(env.action_space.n)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(n_act, obs_n))
    bias = np.zeros(n_act)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + episode)
            grad_w: list[np.ndarray] = []
            grad_b: list[np.ndarray] = []
            rewards: list[float] = []
            done = False
            while not done:
                obs_v = np.asarray(obs, dtype=float).ravel()
                logits = weights @ obs_v + bias
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / np.sum(probs)
                action = int(rng.choice(n_act, p=probs))
                one_hot = np.zeros(n_act)
                one_hot[action] = 1.0
                advantage_direction = one_hot - probs
                grad_w.append(np.outer(advantage_direction, obs_v))
                grad_b.append(advantage_direction)
                obs, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                done = bool(terminated or truncated)
            gain = 0.0
            gains: list[float] = []
            for reward in reversed(rewards):
                gain = reward + gamma * gain
                gains.append(gain)
            gains.reverse()
            baseline = float(np.mean(gains)) if gains else 0.0
            for step_gain, step_gw, step_gb in zip(gains, grad_w, grad_b):
                advantage = step_gain - baseline
                weights += lr * advantage * step_gw
                bias += lr * advantage * step_gb
            returns.append(float(sum(rewards)))
    finally:
        env.close()
    tail = returns[-5:] if returns else [0.0]
    return {"returns": returns, "mean_last": float(np.mean(tail))}
