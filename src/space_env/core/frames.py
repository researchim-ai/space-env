"""
Переходы ECI ↔ ECEF и угол места над станцией.
GMST — приближение USNO, достаточное для окон связи в учебной среде.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import R_EARTH_EQUATOR, R_EARTH_POLAR
from space_env.core.time import Time


def gmst_rad(time: Time) -> float:
    """Гринвичское среднее звёздное время, рад. UT1 принят равным UTC: бюллетеня IERS в репозитории нет."""
    d = time.jd - 2451545.0
    gmst_hours = (18.697374558 + 24.06570982441908 * d) % 24.0
    return float(gmst_hours / 24.0 * 2.0 * np.pi)


def gast_rad(time: Time) -> float:
    """
    Гринвичское истинное звёздное время: GMST плюс уравнение равноденствий.
    Уравнение — главный член нутации по долготе (±17″), не полный ряд IAU 2000A.
    """
    gmst = gmst_rad(time)
    centuries = (time.jd - 2451545.0) / 36525.0
    omega = np.deg2rad(125.044522 - 1934.136261 * centuries)
    obliquity = np.deg2rad(23.439291 - 0.0130042 * centuries)
    dpsi = np.deg2rad((-17.20 / 3600.0) * np.sin(omega))
    return float(gmst + dpsi * np.cos(obliquity))


def eci_to_ecef(r_eci_m: np.ndarray, time: Time) -> np.ndarray:
    """Поворот вокруг оси Z на GMST. Вектор в метрах."""
    theta = gast_rad(time)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = np.asarray(r_eci_m, dtype=float)
    return np.array([c * x + s * y, -s * x + c * y, z], dtype=float)


def ecef_to_eci(r_ecef_m: np.ndarray, time: Time) -> np.ndarray:
    """Обратный поворот ECEF → ECI."""
    theta = gast_rad(time)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = np.asarray(r_ecef_m, dtype=float)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Геодезические WGS84 → ECEF, м."""
    a = R_EARTH_EQUATOR
    b = R_EARTH_POLAR
    e2 = 1.0 - (b * b) / (a * a)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(r_ecef_m: np.ndarray) -> tuple[float, float, float]:
    """ECEF → широта °, долгота °, высота м. Метод Боуринга, WGS84."""
    x, y, z = np.asarray(r_ecef_m, dtype=float)
    a = R_EARTH_EQUATOR
    b = R_EARTH_POLAR
    e2 = 1.0 - (b * b) / (a * a)
    ep2 = e2 / (1.0 - e2)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    theta = float(np.arctan2(z * a, p * b))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = float(np.arctan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    if abs(cos_lat) < 1e-8:
        alt = float(z / sin_lat - n * (1.0 - e2))
    else:
        alt = float(p / cos_lat - n)
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), alt


def elevation_deg(
    r_sat_eci_m: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    time: Time,
) -> float:
    """Угол места КА над местным горизонтом станции, градусы."""
    r_sat = eci_to_ecef(r_sat_eci_m, time)
    r_sta = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    rho = r_sat - r_sta
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    east = -slon * rho[0] + clon * rho[1]
    north = -slat * clon * rho[0] - slat * slon * rho[1] + clat * rho[2]
    up = clat * clon * rho[0] + clat * slon * rho[1] + slat * rho[2]
    return float(np.rad2deg(np.arctan2(up, np.hypot(east, north))))
