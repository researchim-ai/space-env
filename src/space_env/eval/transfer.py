"""Одна политика сдачи на трёх бортах. Дообучения между ними нет."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.envs.cards import observation_index

_VEHICLES = (
    ("sentinel-2a", "space_env/SentinelAcquisition-v0"),
    ("landsat-9", "space_env/LandsatAcquisition-v0"),
    ("cubesat-6u", "space_env/CubesatAcquisition-v0"),
)


def acquisition_action(obs: np.ndarray, env_id: str = "space_env/SentinelAcquisition-v0") -> int:
    """Контакт — сдача, свет и цель у надира — съёмка. Индексы из карточки Sentinel, раскладка у трёх бортов одна."""
    contact = observation_index(env_id, "контакт")
    eclipse = observation_index(env_id, "тень")
    memory = observation_index(env_id, "заполнение памяти")
    off_nadir = observation_index(env_id, "угол от надира")
    vector = np.asarray(obs, dtype=float).ravel()
    if vector[contact] > 0.5 and vector[memory] > 0.0:
        return 2
    if vector[eclipse] < 0.5 and vector[off_nadir] <= 1.0:
        return 1
    if vector[contact] > 0.5:
        return 2
    return 0


def transfer_dump_report(*, max_episode_steps: int = 8, seed: int = 0) -> list[dict]:
    """Три числа сдачи: Sentinel, Landsat, опорный 6U. Политика одна и та же."""
    rows = []
    for vehicle, env_id in _VEHICLES:
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        try:
            obs, _info = env.reset(seed=seed)
            # В памяти уже есть данные: сравнивается сдача, а не удача съёмки до сеанса.
            env.unwrapped._buffer_bits = 1.0e9
            obs = env.unwrapped._obs()
            done = False
            last = {}
            while not done:
                obs, _reward, terminated, truncated, last = env.step(acquisition_action(obs))
                done = bool(terminated or truncated)
        finally:
            env.close()
        rows.append(
            {
                "vehicle": vehicle,
                "env": env_id,
                "delivered_bits": float(last.get("delivered_bits", 0.0)),
                "success": bool(last.get("success", False)),
            }
        )
    return rows
