"""
Ускорение в ECI: центральное поле, J2 (EGM96) и атмосфера.
По умолчанию плотность — таблица среднего Солнца, умноженная на exp(0.012·(F10.7−150)).
Если atmosphere='nrlmsise' и установлен пакет nrlmsise00, плотность берётся из NRLMSISE-00
(полная масса, г/см³ → кг/м³). J2 — форма Кёртиса.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

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


def density_nrlmsise_kg_m3(
    altitude_m: float,
    when: datetime,
    lat_deg: float,
    lon_deg: float,
    *,
    f107: float = 150.0,
    f107a: float = 150.0,
    ap: float = 4.0,
) -> float:
    """Полная массовая плотность NRLMSISE-00, кг/м³. Без пакета — ImportError."""
    try:
        from nrlmsise00 import msise_flat
    except ImportError as exc:
        raise ImportError("Для NRLMSISE установите extra msis: pip install nrlmsise00") from exc
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    densities = msise_flat(when, float(altitude_m) / 1000.0, lat_deg, lon_deg, f107a, f107, ap)
    # Индекс 5 — полная масса в г/см³.
    return float(densities[5]) * 1000.0


def ambient_density_kg_m3(
    altitude_m: float,
    *,
    atmosphere: str = "table",
    f107: float = 150.0,
    f107a: float = 150.0,
    ap: float = 4.0,
    density_scale: float = 1.0,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    when: datetime | None = None,
) -> float:
    """Таблица среднего Солнца или NRLMSISE-00. density_scale — множитель поверх модели."""
    if atmosphere == "nrlmsise":
        moment = when or datetime(2024, 1, 1)
        rho = density_nrlmsise_kg_m3(
            altitude_m, moment, lat_deg, lon_deg, f107=f107, f107a=f107a, ap=ap
        )
    else:
        rho = atmospheric_density_kg_m3(altitude_m) * math.exp(0.012 * (float(f107) - 150.0))
    return float(rho * density_scale)


def acceleration_m_s2(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    *,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
    include_j2: bool = True,
    include_drag: bool = True,
    atmosphere: str = "table",
    f107: float = 150.0,
    f107a: float = 150.0,
    ap: float = 4.0,
    density_scale: float = 1.0,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    when: datetime | None = None,
    srp_area_m2: float = 0.0,
    srp_coefficient: float = 1.2,
    include_third_body: bool = False,
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
            rho = ambient_density_kg_m3(
                radius - R_EARTH_EQUATOR,
                atmosphere=atmosphere,
                f107=f107,
                f107a=f107a,
                ap=ap,
                density_scale=density_scale,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                when=when,
            )
            a = a - 0.5 * drag_coefficient * drag_area_m2 / mass_kg * rho * speed * v_rel
    if srp_area_m2 > 0.0 and mass_kg > 0.0 and srp_coefficient > 0.0 and when is not None:
        a = a + _srp_acceleration_m_s2(r, when, mass_kg, srp_area_m2, srp_coefficient)
    if include_third_body and when is not None:
        from space_env.core.bodies import third_body_acceleration_m_s2
        from space_env.core.time import Time

        a = a + third_body_acceleration_m_s2(r, Time(when))
    return a


def _srp_acceleration_m_s2(
    r_m: np.ndarray,
    when: datetime,
    mass_kg: float,
    area_m2: float,
    coefficient: float,
) -> np.ndarray:
    """
    Пушечное ядро: a = −Cr (A/m) P (1 а.е. / r)² ν · r̂_солнце.
    Направление и доля диска берутся из того же вектора Солнца, что тень и прилив.
    В umbra ускорение нулевое.
    """
    from space_env.core.environment import shadow_state
    from space_env.core.sun import sun_position_m
    from space_env.core.time import Time

    moment = Time(when)
    r_sun = sun_position_m(moment)
    fraction = shadow_state(r_m, r_sun).flux_fraction
    if fraction <= 0.0:
        return np.zeros(3)
    distance = float(np.linalg.norm(r_sun))
    if distance < 1.0:
        return np.zeros(3)
    from space_env.core.constants import AU, SOLAR_PRESSURE_1AU_N_M2

    sun_hat = r_sun / distance
    pressure = SOLAR_PRESSURE_1AU_N_M2 * (AU / distance) ** 2
    # Минус: фотоны летят от Солнца и толкают аппарат от Солнца.
    return -coefficient * (area_m2 / mass_kg) * pressure * fraction * sun_hat
