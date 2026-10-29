"""
Конечное включение: тяга, удельный импульс и убыль массы.
Импульсный бюджет Δv здесь не считается и этим модулем не заменяется.
"""
from __future__ import annotations

import math

# Стандартное ускорение для перевода удельного импульса в скорость истечения.
G0_M_S2: float = 9.80665


def exhaust_speed_m_s(isp_s: float) -> float:
    return float(isp_s) * G0_M_S2


def mass_after_thrust(
    mass_kg: float,
    dt_sec: float,
    thrust_n: float,
    isp_s: float,
    dry_mass_kg: float,
) -> tuple[float, float]:
    """
    Масса после интервала и импульс тяги, м/с.
    Расход ṁ = F / (Isp g0) постоянен, пока есть топливо над сухой массой.
    """
    if thrust_n <= 0.0 or isp_s <= 0.0 or dt_sec <= 0.0 or mass_kg <= dry_mass_kg:
        return float(mass_kg), 0.0
    ve = exhaust_speed_m_s(isp_s)
    m_dot = thrust_n / ve
    propellant = float(mass_kg) - max(float(dry_mass_kg), 0.0)
    used = min(float(dt_sec), propellant / m_dot)
    new_mass = float(mass_kg) - m_dot * used
    if new_mass <= 0.0 or new_mass >= mass_kg:
        return float(mass_kg), 0.0
    return new_mass, ve * math.log(float(mass_kg) / new_mass)


def burn_seconds(
    mass_kg: float,
    dv_m_s: float,
    thrust_n: float,
    isp_s: float,
    dry_mass_kg: float,
    dt_sec: float,
) -> float:
    """Сколько секунд этого шага гореть, чтобы набрать dv_m_s и не выйти за сухую массу."""
    if thrust_n <= 0.0 or isp_s <= 0.0 or dv_m_s <= 0.0 or dt_sec <= 0.0 or mass_kg <= dry_mass_kg:
        return 0.0
    ve = exhaust_speed_m_s(isp_s)
    m_dot = thrust_n / ve
    propellant_time = (float(mass_kg) - float(dry_mass_kg)) / m_dot
    dv_time = (float(mass_kg) * ve / thrust_n) * (1.0 - math.exp(-float(dv_m_s) / ve))
    return max(0.0, min(float(dt_sec), propellant_time, dv_time))
