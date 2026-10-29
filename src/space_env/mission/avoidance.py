"""
Уклонение от сближения тратит бюджет импульса миссии.
Отказ оставляет промах и вероятность на месте.
"""
from __future__ import annotations

from space_env.core.probability import collision_probability


def avoidance_plan(
    miss_m: float,
    *,
    hard_body_radius_m: float,
    sigma_m: float,
    dv_budget_m_s: float,
    mission_reserve_m_s: float,
    target_miss_m: float,
    dv_per_m: float,
) -> dict:
    """
    Доводим промах до target_miss_m ценой dv_per_m * прирост.
    dv_per_m задаётся постановкой до прогона, это не подгонка под Pc.
    """
    if dv_per_m <= 0.0 or target_miss_m <= 0.0:
        raise ValueError("цена импульса и целевой промах должны быть положительными")
    before = collision_probability(miss_m, hard_body_radius_m, sigma_m)
    spare = float(dv_budget_m_s) - float(mission_reserve_m_s)
    if miss_m + 1e-6 >= target_miss_m:
        return {
            "accepted": False,
            "reason": "промах уже достаточен",
            "dv_m_s": 0.0,
            "miss_m": float(miss_m),
            "pc_before": before,
            "pc_after": before,
            "mission_reserve_m_s": float(mission_reserve_m_s),
        }
    needed = (float(target_miss_m) - float(miss_m)) * float(dv_per_m)
    if needed > spare + 1e-9:
        return {
            "accepted": False,
            "reason": "бюджет миссии",
            "dv_m_s": needed,
            "miss_m": float(miss_m),
            "pc_before": before,
            "pc_after": before,
            "mission_reserve_m_s": float(mission_reserve_m_s),
        }
    after = collision_probability(target_miss_m, hard_body_radius_m, sigma_m)
    return {
        "accepted": True,
        "reason": "",
        "dv_m_s": needed,
        "miss_m": float(target_miss_m),
        "pc_before": before,
        "pc_after": after,
        "mission_reserve_m_s": float(mission_reserve_m_s),
    }
