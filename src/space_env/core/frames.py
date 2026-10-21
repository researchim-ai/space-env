"""
Переходы ECI ↔ ECEF и угол места над станцией.
GMST — приближение USNO, достаточное для окон связи в учебной среде.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import R_EARTH_EQUATOR, R_EARTH_POLAR
from space_env.core.dut1 import dut1_seconds, pole_arcsec
from space_env.core.time import Time


def gmst_rad(time: Time) -> float:
    """Гринвичское среднее звёздное время, рад. Аргумент — UT1: UTC плюс DUT1 снимка IERS, иначе UTC."""
    d = time.jd + dut1_seconds(time) / 86400.0 - 2451545.0
    gmst_hours = (18.697374558 + 24.06570982441908 * d) % 24.0
    return float(gmst_hours / 24.0 * 2.0 * np.pi)


# Десять крупнейших членов IAU 1980: (l, l', F, D, Ω, Aψ, Bψ, Aε, Bε).
# A и B в единицах 0.0001″. Это не полный ряд IAU 2000A (~1000 членов).
_IAU1980_TERMS: tuple[tuple[int, int, int, int, int, float, float, float, float], ...] = (
    (0, 0, 0, 0, 1, -171996.0, -174.2, 92025.0, 8.9),
    (0, 0, 0, 0, 2, 2062.0, 0.2, -895.0, 0.5),
    (0, 0, 2, -2, 2, -13187.0, -1.6, 5736.0, -3.1),
    (0, 1, 0, 0, 0, 1426.0, -3.4, 54.0, -0.1),
    (0, 0, 2, 0, 2, -2274.0, -0.2, 977.0, -0.5),
    (1, 0, 0, 0, 0, 712.0, 0.1, -7.0, 0.0),
    (0, 1, 2, -2, 2, -517.0, 1.2, 224.0, -0.6),
    (0, 0, 2, 0, 1, -386.0, -0.4, 200.0, 0.0),
    (1, 0, 2, 0, 2, -301.0, 0.0, 129.0, -0.1),
    (0, -1, 2, -2, 2, 217.0, -0.5, -95.0, 0.3),
)
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def _delaunay_rad(centuries: float) -> tuple[float, float, float, float, float]:
    """Фундаментальные аргументы IERS (l, l′, F, D, Ω), рад. T — юлианские столетия от J2000."""
    t = centuries
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    moon = 485868.249036 + 1717915923.2178 * t + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4
    sun = 1287104.79305 + 129596581.0481 * t - 0.5532 * t2 + 0.000136 * t3 - 0.00001149 * t4
    latitude = 335779.526232 + 1739527262.8478 * t - 12.7512 * t2 - 0.001037 * t3 + 0.00000417 * t4
    elongation = 1072260.70369 + 1602961601.2090 * t - 6.3706 * t2 + 0.006593 * t3 - 0.00003169 * t4
    node = 450160.398036 - 6962890.5431 * t + 7.4722 * t2 + 0.007702 * t3 - 0.00005939 * t4
    return tuple(item * _ARCSEC_TO_RAD for item in (moon, sun, latitude, elongation, node))  # type: ignore[return-value]


def nutation_rad(time: Time) -> tuple[float, float]:
    """Нутация в долготе и наклоне (Δψ, Δε), рад. Десять главных членов IAU 1980."""
    centuries = (time.jd - 2451545.0) / 36525.0
    arguments = _delaunay_rad(centuries)
    longitude = 0.0
    obliquity = 0.0
    for *indexes, long_a, long_b, obl_a, obl_b in _IAU1980_TERMS:
        angle = sum(index * arg for index, arg in zip(indexes, arguments))
        longitude += (long_a + long_b * centuries) * np.sin(angle)
        obliquity += (obl_a + obl_b * centuries) * np.cos(angle)
    scale = _ARCSEC_TO_RAD / 10000.0
    return float(longitude * scale), float(obliquity * scale)


def gast_rad(time: Time) -> float:
    """
    Гринвичское истинное звёздное время.
    GMST плюс уравнение равноденствий по десяти членам IAU 1980.
    UT1 берётся из снимка IERS, когда дата внутри него. Полный ряд IAU 2000A не используется.
    """
    centuries = (time.jd - 2451545.0) / 36525.0
    dpsi, _deps = nutation_rad(time)
    mean_obliquity = np.deg2rad(23.439291 - 0.0130042 * centuries)
    _moon, _sun, _lat, _elong, node = _delaunay_rad(centuries)
    complementary = (0.00264 * np.sin(node) + 0.000063 * np.sin(2.0 * node)) * _ARCSEC_TO_RAD
    return float(gmst_rad(time) + dpsi * np.cos(mean_obliquity) + complementary)


def _rotate_z(vector: np.ndarray, theta: float) -> np.ndarray:
    """R3(θ) в соглашении IERS: ось Z, правый поворот."""
    cosine, sine = np.cos(theta), np.sin(theta)
    x, y, z = vector
    return np.array([cosine * x + sine * y, -sine * x + cosine * y, z], dtype=float)


def _polar_motion(vector: np.ndarray, xp_rad: float, yp_rad: float) -> np.ndarray:
    """
    W = R2(xp) R1(yp) из Соглашений IERS 2010, без локатора TIO s'.
    s' — микросекунды дуги, в этот поворот не входит.
    """
    cosine_y, sine_y = np.cos(yp_rad), np.sin(yp_rad)
    x, y, z = vector
    y1 = cosine_y * y + sine_y * z
    z1 = -sine_y * y + cosine_y * z
    cosine_x, sine_x = np.cos(xp_rad), np.sin(xp_rad)
    return np.array([cosine_x * x - sine_x * z1, y1, sine_x * x + cosine_x * z1], dtype=float)


def _undo_polar_motion(vector: np.ndarray, xp_rad: float, yp_rad: float) -> np.ndarray:
    """R1(−yp) R2(−xp), обратный к W."""
    cosine_x, sine_x = np.cos(xp_rad), np.sin(xp_rad)
    x, y, z = vector
    x1 = cosine_x * x + sine_x * z
    z1 = -sine_x * x + cosine_x * z
    cosine_y, sine_y = np.cos(yp_rad), np.sin(yp_rad)
    return np.array([x1, cosine_y * y - sine_y * z1, sine_y * y + cosine_y * z1], dtype=float)


def eci_to_ecef(r_eci_m: np.ndarray, time: Time) -> np.ndarray:
    """Сначала звёздное время, затем полярное движение. Вектор в метрах."""
    spun = _rotate_z(np.asarray(r_eci_m, dtype=float), gast_rad(time))
    xp, yp = pole_arcsec(time)
    arcsec = np.pi / (180.0 * 3600.0)
    return _polar_motion(spun, xp * arcsec, yp * arcsec)


def ecef_to_eci(r_ecef_m: np.ndarray, time: Time) -> np.ndarray:
    """Обратный переход: снять полярное движение, затем звёздное время."""
    xp, yp = pole_arcsec(time)
    arcsec = np.pi / (180.0 * 3600.0)
    undone = _undo_polar_motion(np.asarray(r_ecef_m, dtype=float), xp * arcsec, yp * arcsec)
    return _rotate_z(undone, -gast_rad(time))


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
