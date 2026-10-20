"""Шаг аккумулятора и выработка солнечных батарей."""
from __future__ import annotations


def solar_generation_w(area_m2: float, flux_W_m2: float, efficiency: float, health_factor: float) -> float:
    """Мощность СБ, Вт. health_factor: 1 номинал, 0.5 деградация, 0 отказ."""
    return max(0.0, area_m2 * flux_W_m2 * efficiency * health_factor)


def step_battery_soc(
    soc: float,
    capacity_Wh: float,
    consumption_W: float,
    generation_W: float,
    dt_sec: float,
) -> float:
    """Новый уровень заряда 0..1. Положительный баланс заряжает батарею."""
    if capacity_Wh <= 0.0:
        return soc
    net_Wh = (generation_W - consumption_W) * (dt_sec / 3600.0)
    return float(min(1.0, max(0.0, soc + net_Wh / capacity_Wh)))
