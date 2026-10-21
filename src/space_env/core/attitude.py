"""
Кватернион борт → инерциальная СК и разворот с постоянной угловой скоростью.
Ось прибора — +Z тела. Крен вокруг неё не задаётся: берётся кратчайший поворот.
"""
from __future__ import annotations

import numpy as np


def _normalize(quaternion: np.ndarray) -> np.ndarray:
    vector = np.asarray(quaternion, dtype=float)
    return vector / np.linalg.norm(vector)


def align_z(direction: np.ndarray) -> np.ndarray:
    """Кватернион (w, x, y, z), который ставит +Z тела вдоль direction."""
    target = np.asarray(direction, dtype=float)
    target = target / np.linalg.norm(target)
    body_z = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(body_z, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if cosine < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(body_z, target)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * float(np.arccos(cosine))
    sine = float(np.sin(half))
    return np.array([float(np.cos(half)), axis[0] * sine, axis[1] * sine, axis[2] * sine])


def boresight(quaternion: np.ndarray) -> np.ndarray:
    """Куда смотрит +Z после поворота кватернионом."""
    w, x, y, z = _normalize(quaternion)
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def _slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    left = _normalize(start)
    right = _normalize(end)
    if float(np.dot(left, right)) < 0.0:
        right = -right
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return right
    weight_left = np.sin((1.0 - fraction) * angle) / np.sin(angle)
    weight_right = np.sin(fraction * angle) / np.sin(angle)
    return _normalize(weight_left * left + weight_right * right)


def slew_step(
    quaternion: np.ndarray,
    target: np.ndarray,
    rate_deg_s: float,
    dt_sec: float,
) -> tuple[np.ndarray, float]:
    """
    Шаг к target со скоростью rate_deg_s.
    Возвращает новый кватернион и оставшийся угол, градусы.
    """
    start = _normalize(quaternion)
    goal = _normalize(target)
    if float(np.dot(start, goal)) < 0.0:
        goal = -goal
    cosine = float(np.clip(np.dot(start, goal), -1.0, 1.0))
    angle_rad = float(np.arccos(cosine)) * 2.0
    if angle_rad < 1e-8 or dt_sec <= 0.0 or rate_deg_s <= 0.0:
        return goal, 0.0
    fraction = min(1.0, np.deg2rad(rate_deg_s) * dt_sec / angle_rad)
    updated = goal if fraction >= 1.0 else _slerp(start, goal, fraction)
    return updated, float(np.rad2deg(angle_rad * (1.0 - fraction)))
