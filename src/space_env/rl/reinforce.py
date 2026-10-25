"""Линейный REINFORCE для дискретных сред Gymnasium. Без внешних RL-библиотек."""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    return probs / np.sum(probs)


def _artifact(
    *,
    returns: list[float],
    weights,
    bias,
    baseline: float,
    baseline_count: int,
    obs_n: int,
    seed: int,
    lr: float,
    gamma: float,
    rng: np.random.Generator,
) -> dict:
    tail = returns[-5:] if returns else [0.0]
    return {
        "returns": returns,
        "mean_last": float(np.mean(tail)),
        "weights": weights,
        "bias": bias,
        "baseline": float(baseline),
        "baseline_count": int(baseline_count),
        "obs_mean": np.zeros(obs_n),
        "obs_scale": np.ones(obs_n),
        "seed": seed,
        "lr": lr,
        "gamma": gamma,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
    }


def train_linear_reinforce(
    env_id: str,
    *,
    episodes: int = 40,
    max_episode_steps: int = 20,
    seed: int = 0,
    lr: float = 0.05,
    gamma: float = 0.99,
    env: gym.Env | None = None,
    weights: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    baseline: float = 0.0,
    baseline_count: int = 0,
    rng_state: dict | None = None,
) -> dict:
    """
    Политика softmax(W obs + b). Baseline — скользящее среднее прошлых возвратов.
    Он не зависит от действия текущего шага: у эпизода из одного шага преимущество не обнуляется.
    """
    owns_env = env is None
    if env is None:
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
    if not hasattr(env.action_space, "n"):
        kind = type(env.action_space).__name__
        if owns_env:
            env.close()
        raise TypeError(
            "линейный REINFORCE читает одно дискретное действие. "
            f"У {env_id} пространство {kind}."
        )
    obs_n = int(np.prod(env.observation_space.shape))
    n_act = int(env.action_space.n)
    rng = np.random.default_rng(seed)
    if rng_state is not None:
        rng.bit_generator.state = copy.deepcopy(rng_state)
    if weights is None:
        weights = rng.normal(0.0, 0.01, size=(n_act, obs_n))
        bias = np.zeros(n_act)
    else:
        weights = np.array(weights, dtype=float, copy=True)
        bias = np.zeros(n_act) if bias is None else np.array(bias, dtype=float, copy=True)
    running = float(baseline)
    seen = int(baseline_count)
    trained_before = seen
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + trained_before + episode)
            grad_w: list[np.ndarray] = []
            grad_b: list[np.ndarray] = []
            rewards: list[float] = []
            done = False
            while not done:
                obs_v = np.asarray(obs, dtype=float).ravel()
                probs = _softmax(weights @ obs_v + bias)
                action = int(rng.choice(n_act, p=probs))
                one_hot = np.zeros(n_act)
                one_hot[action] = 1.0
                grad_w.append(np.outer(one_hot - probs, obs_v))
                grad_b.append(one_hot - probs)
                obs, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                done = bool(terminated or truncated)
            gain = 0.0
            gains: list[float] = []
            for reward in reversed(rewards):
                gain = reward + gamma * gain
                gains.append(gain)
            gains.reverse()
            for step_gain, step_gw, step_gb in zip(gains, grad_w, grad_b):
                advantage = step_gain - running
                weights += lr * advantage * step_gw
                bias += lr * advantage * step_gb
            episode_return = float(sum(rewards))
            returns.append(episode_return)
            seen += 1
            running += (episode_return - running) / seen
    finally:
        if owns_env:
            env.close()
    return _artifact(
        returns=returns,
        weights=weights,
        bias=bias,
        baseline=running,
        baseline_count=seen,
        obs_n=obs_n,
        seed=seed,
        lr=lr,
        gamma=gamma,
        rng=rng,
    )


def evaluate_linear_policy(
    env_id: str,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    episodes: int = 1,
    max_episode_steps: int = 20,
    seed: int = 0,
    env: gym.Env | None = None,
) -> dict[str, float]:
    """Жадная оценка сохранённых весов. Веса и baseline не меняет."""
    owns_env = env is None
    if env is None:
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
    total = 0.0
    success = 0.0
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + episode)
            done = False
            episode_success = False
            while not done:
                obs_v = np.asarray(obs, dtype=float).ravel()
                action = int(np.argmax(weights @ obs_v + bias))
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                episode_success = bool(info.get("success", False))
                done = bool(terminated or truncated)
            success += float(episode_success)
    finally:
        if owns_env:
            env.close()
    return {
        "mean_reward": total / max(episodes, 1),
        "success_rate": success / max(episodes, 1),
    }


def train_multidiscrete_reinforce(
    env_id: str,
    *,
    episodes: int = 40,
    max_episode_steps: int = 20,
    seed: int = 0,
    lr: float = 0.05,
    gamma: float = 0.99,
    weights: list[np.ndarray] | None = None,
    bias: list[np.ndarray] | None = None,
    baseline: float = 0.0,
    baseline_count: int = 0,
    rng_state: dict | None = None,
) -> dict:
    """
    Независимые softmax на каждую компоненту MultiDiscrete.
    Baseline тот же: среднее прошлых возвратов, без вычитания среднего текущего эпизода.
    """
    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
        kind = type(env.action_space).__name__
        env.close()
        raise TypeError(f"тренер нескольких чисел ждёт MultiDiscrete, у {env_id} пространство {kind}.")
    obs_n = int(np.prod(env.observation_space.shape))
    factors = [int(item) for item in np.asarray(env.action_space.nvec).ravel()]
    rng = np.random.default_rng(seed)
    if rng_state is not None:
        rng.bit_generator.state = copy.deepcopy(rng_state)
    if weights is None:
        weights = [rng.normal(0.0, 0.01, size=(count, obs_n)) for count in factors]
        bias = [np.zeros(count) for count in factors]
    else:
        weights = [np.array(item, dtype=float, copy=True) for item in weights]
        bias = [np.zeros(count) for count in factors] if bias is None else [np.array(item, dtype=float, copy=True) for item in bias]
    running = float(baseline)
    seen = int(baseline_count)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + episode)
            step_grads: list[list[tuple[np.ndarray, np.ndarray]]] = []
            rewards: list[float] = []
            done = False
            while not done:
                obs_v = np.asarray(obs, dtype=float).ravel()
                action = np.zeros(len(factors), dtype=int)
                grads: list[tuple[np.ndarray, np.ndarray]] = []
                for index, count in enumerate(factors):
                    probs = _softmax(weights[index] @ obs_v + bias[index])
                    choice = int(rng.choice(count, p=probs))
                    one_hot = np.zeros(count)
                    one_hot[choice] = 1.0
                    action[index] = choice
                    grads.append((one_hot - probs, obs_v))
                step_grads.append(grads)
                obs, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                done = bool(terminated or truncated)
            gain = 0.0
            gains: list[float] = []
            for reward in reversed(rewards):
                gain = reward + gamma * gain
                gains.append(gain)
            gains.reverse()
            for step_gain, grads in zip(gains, step_grads):
                advantage = step_gain - running
                for index, (direction, obs_v) in enumerate(grads):
                    weights[index] += lr * advantage * np.outer(direction, obs_v)
                    bias[index] += lr * advantage * direction
            episode_return = float(sum(rewards))
            returns.append(episode_return)
            seen += 1
            running += (episode_return - running) / seen
    finally:
        env.close()
    return _artifact(
        returns=returns,
        weights=weights,
        bias=bias,
        baseline=running,
        baseline_count=seen,
        obs_n=obs_n,
        seed=seed,
        lr=lr,
        gamma=gamma,
        rng=rng,
    )
