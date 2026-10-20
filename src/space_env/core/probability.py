"""
Вероятность столкновения в плоскости промаха.
Малое тело, круговая суммарная ковариация: приближение Чена
Pc ≈ (R² / (2 σ²)) exp(−d² / (2 σ²)).
R — радиус твёрдого тела, σ — суммарное СКО в плоскости встречи, d — промах.
Это не полный интеграл Альфано и не подгонка под заданное число.
"""
from __future__ import annotations

import math


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
