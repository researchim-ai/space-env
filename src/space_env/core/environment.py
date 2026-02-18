"""
Пространственная среда: Земля, тень (eclipse), константы.
Упрощённая модель — сфера Земли и цилиндрическая тень.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import R_EARTH_EQUATOR, SOLAR_CONSTANT


def in_eclipse(r_sat_m: np.ndarray, r_sun_m: np.ndarray) -> bool:
    """
    Находится ли КА в тени Земли (упрощённо: цилиндрическая тень).
    r_sat_m, r_sun_m — векторы положения КА и Солнца в инерциальной СК (м).
    """
    r_sat = np.asarray(r_sat_m, dtype=float)
    r_sun = np.asarray(r_sun_m, dtype=float)
    r_sat_norm = np.linalg.norm(r_sat)
    r_sun_norm = np.linalg.norm(r_sun)
    if r_sat_norm < 1e3 or r_sun_norm < 1e3:
        return False
    # Единичный вектор на Солнце от Земли
    s = r_sun / r_sun_norm
    # Проекция положения КА на направление на Солнце
    proj = np.dot(r_sat, s)
    if proj <= 0:
        # КА «за» Землёй по направлению к Солнцу
        return True
    # Расстояние от оси Солнце–Земля до КА
    r_perp = np.linalg.norm(r_sat - proj * s)
    return bool(r_perp < R_EARTH_EQUATOR)


def solar_flux_at_satellite_W_m2(r_sat_m: np.ndarray, r_sun_m: np.ndarray, in_umbra: bool) -> float:
    """
    Солнечный поток в месте КА, Вт/м².
    Упрощённо: SOLAR_CONSTANT если на свету, 0 в тени; расстояние до Солнца не учитываем.
    """
    if in_umbra:
        return 0.0
    return SOLAR_CONSTANT


def altitude_from_position_m(r_m: np.ndarray) -> float:
    """Высота над сферой R_EARTH_EQUATOR, м."""
    return float(np.linalg.norm(r_m) - R_EARTH_EQUATOR)
