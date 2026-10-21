"""
Вероятность столкновения в плоскости промаха.
collision_probability — приближение Чена для кругового σ и малого тела.
collision_probability_integral — численный интеграл двумерной гауссианы по диску
твёрдого тела. Это та же величина, которую считает метод Альфано; здесь она
берётся квадратурой, а не рядом.
"""
from __future__ import annotations

import math

import numpy as np


def collision_probability(miss_m: float, hard_body_radius_m: float, sigma_m: float) -> float:
    """Pc в диапазоне 0..1. При σ → 0 вероятность скачком 1 внутри радиуса тела."""
    if hard_body_radius_m <= 0.0:
        return 0.0
    if sigma_m <= 1e-6:
        return 1.0 if miss_m <= hard_body_radius_m else 0.0
    scale = (hard_body_radius_m ** 2) / (2.0 * sigma_m ** 2)
    exponent = -0.5 * (miss_m / sigma_m) ** 2
    # Защита от переполнения: при большом промахе вероятность нулевая.
    if exponent < -80.0:
        return 0.0
    return float(min(1.0, scale * math.exp(exponent)))


def collision_probability_integral(
    mean_m: np.ndarray,
    covariance_m2: np.ndarray,
    hard_body_radius_m: float,
    *,
    n_radial: int = 28,
    n_angle: int = 48,
) -> float:
    """
    ∫∫_{x²+y²<R²} N(mean, cov) dx dy.
    Диск сидит в начале координат плоскости промаха, mean — прогнозируемый промах.
    """
    if hard_body_radius_m <= 0.0:
        return 0.0
    cov = 0.5 * (np.asarray(covariance_m2, dtype=float) + np.asarray(covariance_m2, dtype=float).T)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0:
        distance = float(np.linalg.norm(mean_m))
        return 1.0 if distance <= hard_body_radius_m else 0.0
    inv = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * math.pi * math.exp(0.5 * logdet))
    mean = np.asarray(mean_m, dtype=float)
    total = 0.0
    for i in range(n_radial):
        r0 = hard_body_radius_m * i / n_radial
        r1 = hard_body_radius_m * (i + 1) / n_radial
        radius = 0.5 * (r0 + r1)
        cell = math.pi * (r1 * r1 - r0 * r0) / n_angle
        for j in range(n_angle):
            theta = 2.0 * math.pi * (j + 0.5) / n_angle
            point = np.array([radius * math.cos(theta), radius * math.sin(theta)])
            delta = point - mean
            exponent = -0.5 * float(delta @ inv @ delta)
            if exponent < -80.0:
                continue
            total += norm * math.exp(exponent) * cell
    return float(min(1.0, max(0.0, total)))


def bplane_probability(
    dr_rtn_m: np.ndarray,
    dv_rtn_m_s: np.ndarray,
    covariance_rtn_m2: np.ndarray,
    hard_body_radius_m: float,
) -> float:
    """
    Проецирует позиционную ковариацию RTN на плоскость, перпендикулярную
    относительной скорости, и интегрирует по диску твёрдого тела.
    """
    position = np.asarray(covariance_rtn_m2, dtype=float)[:3, :3]
    velocity = np.asarray(dv_rtn_m_s, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = velocity / speed
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = np.cross(axis, helper)
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    basis = np.vstack((b1, b2))
    miss = basis @ np.asarray(dr_rtn_m, dtype=float)
    cov = basis @ position @ basis.T
    return collision_probability_integral(miss, cov, hard_body_radius_m)
