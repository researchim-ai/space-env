"""
Положение Луны и приливное ускорение точечных тел.
В приливе Луна берётся из ERFA moon98, если пакет есть. Ряд Миуса остаётся для сравнения.
Солнце тени и светового давления — по-прежнему укороченный ряд Миуса.
Прилив: GM (r_отн/|r|³ − r_тела/|R|³).
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import AU
from space_env.core.time import Time

MU_MOON_M3_S2: float = 4.9048695e12
MU_SUN_M3_S2: float = 1.32712440018e20


def moon_meeus_m(time: Time) -> np.ndarray:
    """Геоцентрический вектор на Луну, м. Укороченный ряд Миуса, сотни километров."""
    day = time.jd - 2451545.0
    longitude = np.deg2rad((218.3164477 + 13.17639648 * day) % 360.0)
    anomaly = np.deg2rad((134.9633964 + 13.06499295 * day) % 360.0)
    sun_anomaly = np.deg2rad((357.5291092 + 0.98560028 * day) % 360.0)
    latitude_arg = np.deg2rad((93.2720950 + 13.22935024 * day) % 360.0)
    elongation = longitude - np.deg2rad((357.5291092 + 0.98560028 * day) % 360.0)
    ecliptic_lon = (
        longitude
        + np.deg2rad(6.289) * np.sin(anomaly)
        + np.deg2rad(1.274) * np.sin(2 * elongation - anomaly)
        + np.deg2rad(0.658) * np.sin(2 * elongation)
        + np.deg2rad(0.214) * np.sin(2 * anomaly)
        + np.deg2rad(0.186) * np.sin(sun_anomaly)
    )
    ecliptic_lat = (
        np.deg2rad(5.128) * np.sin(latitude_arg)
        + np.deg2rad(0.281) * np.sin(anomaly + latitude_arg)
    )
    distance_m = (
        385000.56e3
        - 20905.355e3 * np.cos(anomaly)
        - 3699.111e3 * np.cos(2 * elongation - anomaly)
        - 2955.968e3 * np.cos(2 * elongation)
    )
    obliquity = np.deg2rad(23.439291)
    cos_lon = np.cos(ecliptic_lon)
    sin_lon = np.sin(ecliptic_lon)
    cos_lat = np.cos(ecliptic_lat)
    sin_lat = np.sin(ecliptic_lat)
    x = distance_m * cos_lon * cos_lat
    y = distance_m * (sin_lon * cos_lat * np.cos(obliquity) - sin_lat * np.sin(obliquity))
    z = distance_m * (sin_lon * cos_lat * np.sin(obliquity) + sin_lat * np.cos(obliquity))
    return np.array([x, y, z], dtype=float)


def moon_iau2000_m(time: Time, *, tai_minus_utc_s: float = 37.0) -> np.ndarray:
    """Геоцентрическая Луна по ERFA moon98, метры. Ряд IAU, не Миус."""
    try:
        from erfa import moon98
    except ImportError as exc:
        raise ImportError("для ряда Луны IAU нужен пакет erfa") from exc

    tt = float(time.jd) + (float(tai_minus_utc_s) + 32.184) / 86400.0
    sample = moon98(2400000.5, tt - 2400000.5)
    return np.asarray(sample["p"], dtype=float) * AU


def moon_position_m(time: Time) -> np.ndarray:
    """Луна для прилива. ERFA moon98, если пакет установлен, иначе ряд Миуса."""
    try:
        return moon_iau2000_m(time)
    except ImportError:
        return moon_meeus_m(time)


def tidal_acceleration_m_s2(r_sat_m: np.ndarray, r_body_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """Приливное ускорение точечного тела на спутнике у Земли."""
    r_sat = np.asarray(r_sat_m, dtype=float)
    r_body = np.asarray(r_body_m, dtype=float)
    delta = r_body - r_sat
    distance = float(np.linalg.norm(delta))
    body = float(np.linalg.norm(r_body))
    if distance < 1.0 or body < 1.0:
        return np.zeros(3)
    return mu_m3_s2 * (delta / distance**3 - r_body / body**3)


def third_body_acceleration_m_s2(r_sat_m: np.ndarray, time: Time) -> np.ndarray:
    """Сумма приливов Луны и Солнца. Солнце прилива — ряд ERFA, не ряд тени."""
    from space_env.core.sun import sun_iau2000_m

    moon = tidal_acceleration_m_s2(r_sat_m, moon_position_m(time), MU_MOON_M3_S2)
    sun = tidal_acceleration_m_s2(r_sat_m, sun_iau2000_m(time), MU_SUN_M3_S2)
    return moon + sun
