"""
Ускорение в ECI: центральное поле, J2 (EGM96) и экспоненциальная атмосфера.
Плотность — кусочно-логарифмическая таблица среднего Солнца, не NRLMSISE.
J2 — форма Кёртиса: a_J2 = −(3/2) J2 μ Re²/r⁵ · (x(1−5z²/r²), y(1−5z²/r²), z(3−5z²/r²)).
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH_EQUATOR

# Высота, км → плотность, кг/м³. Ориентир US Standard / средний солнечный цикл.
_DENSITY_TABLE_KM = np.array([150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800], dtype=float)
_DENSITY_TABLE_RHO = np.array(
    [2.07e-9, 2.54e-10, 6.07e-11, 1.92e-11, 7.0e-12, 2.8e-12, 1.2e-12, 5.2e-13, 1.1e-13, 3.0e-14, 1.1e-14],
    dtype=float,
)


def atmospheric_density_kg_m3(altitude_m: float) -> float:
    """Логарифмическая интерполяция таблицы. Выше 800 км и ниже 150 км — крайние узлы."""
    h_km = float(altitude_m) / 1000.0
    if h_km <= _DENSITY_TABLE_KM[0]:
        return float(_DENSITY_TABLE_RHO[0])
    if h_km >= _DENSITY_TABLE_KM[-1]:
        return float(_DENSITY_TABLE_RHO[-1])
    log_rho = np.interp(h_km, _DENSITY_TABLE_KM, np.log(_DENSITY_TABLE_RHO))
    return float(np.exp(log_rho))


def acceleration_m_s2(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    *,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
    include_j2: bool = True,
    include_drag: bool = True,
) -> np.ndarray:
    """Ускорение, м/с². Сопротивление считается относительно атмосферы, вращающейся с Землёй."""
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    radius = float(np.linalg.norm(r))
    if radius < 1.0:
        return np.zeros(3)
    a = -MU_EARTH * r / radius**3
    if include_j2:
        z2 = r[2] ** 2
        r2 = radius * radius
        factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH_EQUATOR**2 / radius**5
        a = a + factor * np.array(
            [
                r[0] * (5.0 * z2 / r2 - 1.0),
                r[1] * (5.0 * z2 / r2 - 1.0),
                r[2] * (5.0 * z2 / r2 - 3.0),
            ]
        )
        # factor * (5z²/r² − 1) = −factor * (1 − 5z²/r²), это и есть форма Кёртиса.
    if include_drag and drag_area_m2 > 0.0 and mass_kg > 0.0 and drag_coefficient > 0.0:
        omega = np.array([0.0, 0.0, OMEGA_EARTH])
        v_rel = v - np.cross(omega, r)
        speed = float(np.linalg.norm(v_rel))
        if speed > 0.0:
            rho = atmospheric_density_kg_m3(radius - R_EARTH_EQUATOR)
            a = a - 0.5 * drag_coefficient * drag_area_m2 / mass_kg * rho * speed * v_rel
    return a
