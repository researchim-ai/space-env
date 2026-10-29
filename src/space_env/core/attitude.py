"""
Кватернион борт → инерциальная СК и разворот.
Ось прибора — +Z тела. Крен вокруг неё не задаётся: берётся кратчайший поворот.
Нулевое ускорение набирает предельную скорость сразу. Положительное ускорение
разгоняет, держит предел и тормозит так, чтобы остановиться на цели.
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


def slew_limited(
    quaternion: np.ndarray,
    target: np.ndarray,
    rate_deg_s: float,
    rate_max_deg_s: float,
    accel_deg_s2: float,
    dt_sec: float,
) -> tuple[np.ndarray, float, float]:
    """
    Разворот с пределом скорости и ускорения.
    Возвращает кватернион, остаток в градусах и скорость в конце шага, °/с.
    Ускорение не больше нуля сюда не приходит: такой шаг считает slew_step.
    """
    start = _normalize(quaternion)
    goal = _normalize(target)
    if float(np.dot(start, goal)) < 0.0:
        goal = -goal
    cosine = float(np.clip(np.dot(start, goal), -1.0, 1.0))
    angle_deg = float(np.rad2deg(np.arccos(cosine) * 2.0))
    if angle_deg < 1e-6 or dt_sec <= 0.0 or rate_max_deg_s <= 0.0 or accel_deg_s2 <= 0.0:
        return goal if angle_deg < 1e-6 else start, 0.0 if angle_deg < 1e-6 else angle_deg, 0.0
    remaining = angle_deg
    omega = min(max(float(rate_deg_s), 0.0), float(rate_max_deg_s))
    traveled = 0.0
    left = float(dt_sec)
    while left > 1e-9 and remaining > 1e-6:
        step = min(0.01, left)
        stop = omega * omega / (2.0 * accel_deg_s2)
        if remaining <= stop + 1e-9:
            omega_next = max(0.0, omega - accel_deg_s2 * step)
        elif omega < rate_max_deg_s:
            sped = min(rate_max_deg_s, omega + accel_deg_s2 * step)
            move_try = 0.5 * (omega + sped) * step
            if move_try + sped * sped / (2.0 * accel_deg_s2) > remaining:
                omega_next = max(0.0, omega - accel_deg_s2 * step)
            else:
                omega_next = sped
        else:
            omega_next = omega if stop < remaining else max(0.0, omega - accel_deg_s2 * step)
        move = 0.5 * (omega + omega_next) * step
        if move >= remaining - 1e-9:
            traveled += remaining
            remaining = 0.0
            omega = 0.0
            break
        remaining -= move
        traveled += move
        omega = omega_next
        left -= step
    if traveled >= angle_deg - 1e-6:
        return goal, 0.0, 0.0
    updated = _slerp(start, goal, traveled / angle_deg)
    return updated, remaining, omega
