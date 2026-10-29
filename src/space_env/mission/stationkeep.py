"""
Расписание компенсации торможения.
Импульс берётся из падения большой полуоси и делится на включения.
Отказ по бюджету не ставит включения. Это не полёт на 25 лет.
"""
from __future__ import annotations

from space_env.core.lifetime import drag_makeup_m_s


def drag_makeup_schedule(
    altitude_m: float,
    *,
    mass_kg: float,
    area_m2: float,
    drag_coefficient: float,
    horizon_sec: float,
    burns: int,
    dv_budget_m_s: float,
    mission_reserve_m_s: float,
    f107: float = 150.0,
) -> dict:
    if int(burns) < 1 or float(horizon_sec) <= 0.0:
        raise ValueError("нужны положительный горизонт и хотя бы одно включение")
    total = drag_makeup_m_s(
        altitude_m,
        mass_kg=mass_kg,
        area_m2=area_m2,
        drag_coefficient=drag_coefficient,
        horizon_sec=horizon_sec,
        f107=f107,
    )
    spare = float(dv_budget_m_s) - float(mission_reserve_m_s)
    if total > spare + 1e-12:
        return {
            "accepted": False,
            "reason": "бюджет миссии",
            "total_m_s": total,
            "burns": [],
            "model": "exponential_scale_height",
        }
    count = int(burns)
    each = total / count
    slots = [
        {
            "at_sec": float(horizon_sec) * (index + 1) / (count + 1),
            "dv_m_s": each,
            "direction": "prograde",
        }
        for index in range(count)
    ]
    return {
        "accepted": True,
        "reason": "",
        "total_m_s": total,
        "burns": slots,
        "model": "exponential_scale_height",
    }
