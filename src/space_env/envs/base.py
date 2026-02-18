"""
Базовая обёртка и утилиты для RL-сред.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


def make_box_obs(shape: tuple[int, ...], low: float = 0.0, high: float = 1.0) -> gym.Space:
    return gym.spaces.Box(
        low=np.float32(low),
        high=np.float32(high),
        shape=shape,
        dtype=np.float32,
    )


def normalize_altitude_km(alt_km: float, alt_min: float = 200.0, alt_max: float = 1000.0) -> float:
    return float(np.clip((alt_km - alt_min) / (alt_max - alt_min), 0.0, 1.0))


def normalize_temp(temp_C: float, t_min: float = -20.0, t_max: float = 50.0) -> float:
    return float(np.clip((temp_C - t_min) / (t_max - t_min), 0.0, 1.0))
