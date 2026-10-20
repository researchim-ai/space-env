"""
Низкоточное положение Солнца в геоцентрической инерциальной СК.
Формулы — укороченный ряд Миуса (достаточно для тени и сезонов затмений).
Для ядер SPICE есть space_env.integrations.spice_ephem.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import AU
from space_env.core.time import Time


def sun_position_m(time: Time) -> np.ndarray:
    """Вектор Земля → Солнце, м, приближение экваториальных координат."""
    n = time.jd - 2451545.0
    mean_long = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
    mean_anom = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    ecl_long = (
        mean_long
        + np.deg2rad(1.915) * np.sin(mean_anom)
        + np.deg2rad(0.020) * np.sin(2.0 * mean_anom)
    )
    obliquity = np.deg2rad(23.439 - 0.0000004 * n)
    distance_au = 1.00014 - 0.01671 * np.cos(mean_anom) - 0.00014 * np.cos(2.0 * mean_anom)
    x = distance_au * np.cos(ecl_long)
    y = distance_au * np.cos(obliquity) * np.sin(ecl_long)
    z = distance_au * np.sin(obliquity) * np.sin(ecl_long)
    return np.array([x, y, z], dtype=float) * AU
