"""
Оценка срока круговой орбиты в экспоненциальной атмосфере.
Время порядка H / |da/dt| на начальной высоте: большую часть срока аппарат проводит там.
Это не интегрирование Кауэлла на 25 лет и не сертификат IADC.
"""
from __future__ import annotations

import math

from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.core.gravity import atmospheric_density_kg_m3

_SECONDS_PER_YEAR = 365.25 * 86400.0
_FALLBACK_SCALE_M = 50_000.0


def _density(altitude_m: float, f107: float) -> float:
    return atmospheric_density_kg_m3(altitude_m) * math.exp(0.012 * (float(f107) - 150.0))


def _scale_height_m(altitude_m: float, f107: float) -> float:
    rho_here = _density(altitude_m, f107)
    rho_above = _density(altitude_m + 20_000.0, f107)
    if rho_here <= 0.0 or rho_above <= 0.0 or rho_above >= rho_here:
        return _FALLBACK_SCALE_M
    return -20_000.0 / math.log(rho_above / rho_here)


def circular_lifetime_years(
    altitude_m: float,
    *,
    mass_kg: float,
    area_m2: float,
    drag_coefficient: float,
    f107: float = 150.0,
) -> dict[str, float | bool | str]:
    """
    Годы до схода в масштабе местной шкалы высот.
    da/dt = −(Cd A/m) ρ √(μ a) для круговой орбиты.
    """
    if mass_kg <= 0.0 or area_m2 <= 0.0 or drag_coefficient <= 0.0:
        raise ValueError("масса, площадь и Cd должны быть положительными")
    altitude = float(altitude_m)
    radius = R_EARTH_EQUATOR + altitude
    ballistic = drag_coefficient * area_m2 / mass_kg
    density = _density(altitude, f107)
    scale = _scale_height_m(altitude, f107)
    decay = ballistic * density * math.sqrt(MU_EARTH * radius)
    speed = math.sqrt(MU_EARTH / radius)
    if decay <= 0.0:
        years = math.inf
    else:
        years = (scale / decay) / _SECONDS_PER_YEAR
    return {
        "years": years,
        "within_25_years": years < 25.0,
        "scale_height_m": scale,
        "density_kg_m3": density,
        "decay_m_s": decay,
        "speed_m_s": speed,
        "model": "exponential_scale_height",
    }


def drag_makeup_m_s(altitude_m: float, *, mass_kg: float, area_m2: float, drag_coefficient: float, horizon_sec: float, f107: float = 150.0) -> float:
    """
    Импульс, который за горизонт компенсирует падение большой полуоси.
    Для круговой орбиты Δv = v |Δa| / (2a). Это не 25-летний Кауэлл.
    """
    report = circular_lifetime_years(
        altitude_m,
        mass_kg=mass_kg,
        area_m2=area_m2,
        drag_coefficient=drag_coefficient,
        f107=f107,
    )
    radius = R_EARTH_EQUATOR + float(altitude_m)
    drop = float(report["decay_m_s"]) * float(horizon_sec)
    return float(report["speed_m_s"]) * drop / (2.0 * radius)


_REENTRY_ALTITUDE_M = 120_000.0


def drag_ode_lifetime_years(
    altitude_m: float,
    *,
    mass_kg: float,
    area_m2: float,
    drag_coefficient: float,
    f107: float = 150.0,
    max_years: float = 25.0,
) -> dict[str, float | bool | str | int]:
    """
    Круговая орбита с da/dt = −(Cd A/m) ρ(h) √(μ a) и табличной плотностью на текущей высоте.
    Шаг ограничен падением на 1 км. Это не пошаговый Кауэлл и не сертификат IADC.
    """
    if mass_kg <= 0.0 or area_m2 <= 0.0 or drag_coefficient <= 0.0:
        raise ValueError("масса, площадь и Cd должны быть положительными")
    altitude = float(altitude_m)
    radius = R_EARTH_EQUATOR + altitude
    floor = R_EARTH_EQUATOR + _REENTRY_ALTITUDE_M
    ballistic = drag_coefficient * area_m2 / mass_kg
    elapsed = 0.0
    limit = float(max_years) * _SECONDS_PER_YEAR
    steps = 0
    while elapsed < limit and radius > floor:
        height = radius - R_EARTH_EQUATOR
        density = _density(height, f107)
        rate = ballistic * density * math.sqrt(MU_EARTH * radius)
        if rate <= 0.0:
            break
        step = min(1_000.0 / rate, 86_400.0, limit - elapsed)
        drop = rate * step
        if radius - drop < floor:
            step = (radius - floor) / rate
            drop = rate * step
        radius -= drop
        elapsed += step
        steps += 1
        if steps > 2_000_000:
            break
    reentered = radius <= floor + 1.0
    years = elapsed / _SECONDS_PER_YEAR
    return {
        "years": years,
        "within_25_years": bool(reentered and years < 25.0),
        "reentered": reentered,
        "final_altitude_m": radius - R_EARTH_EQUATOR,
        "steps": steps,
        "model": "drag_ode_table_density",
    }
