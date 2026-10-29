"""
Геоцентрическое Солнце.

Рабочий вектор — ERFA epv00. Его читают тень, световое давление, освещённость
и прилив. Укороченный ряд Миуса остаётся отдельной функцией для сравнения.
Для ядер SPICE есть space_env.integrations.spice_ephem.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import AU
from space_env.core.time import Time


def sun_meeus_m(time: Time) -> np.ndarray:
    """Вектор Земля → Солнце, м, укороченный ряд Миуса. В тень и давление не подставляется."""
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


def sun_position_m(time: Time) -> np.ndarray:
    """Солнце для тени, давления и освещённости. Совпадает с приливом, когда есть ERFA."""
    try:
        return sun_iau2000_m(time)
    except ImportError:
        return sun_meeus_m(time)


def sun_iau2000_m(time: Time, *, tai_minus_utc_s: float = 37.0) -> np.ndarray:
    """Геоцентрическое Солнце по ERFA epv00, метры. Тот же вектор читают тень и световое давление."""
    try:
        from erfa import epv00
    except ImportError as exc:
        raise ImportError("для ряда Солнца IAU нужен пакет erfa") from exc

    tt = float(time.jd) + (float(tai_minus_utc_s) + 32.184) / 86400.0
    heliocentric, _barycentric = epv00(2400000.5, tt - 2400000.5)
    return -np.asarray(heliocentric["p"], dtype=float) * AU
