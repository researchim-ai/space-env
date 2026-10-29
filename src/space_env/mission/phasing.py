"""
Одно включение, которое задаёт вековой уход вдоль трека.
Модель Клохесси–Уилтшира: положительный импульс даёт уход Δy ≈ −3 Δv T.
Чтобы снять ошибку, Δv = ошибка / (3T). Колебательный член в момент T остаётся.
Дальше сотой радиуса и дольше пятой части витка план не строится.
"""
from __future__ import annotations

import math

from space_env.core.relative import mean_motion_rad_s


def phase_plan(
    along_error_m: float,
    *,
    radius_m: float,
    horizon_sec: float,
    dv_budget_m_s: float,
    mission_reserve_m_s: float,
) -> dict:
    motion = mean_motion_rad_s(radius_m)
    period = 2.0 * math.pi / motion
    if abs(float(along_error_m)) > 0.01 * float(radius_m):
        return {
            "accepted": False,
            "reason": "слишком далеко для линейной модели",
            "dv_m_s": 0.0,
            "direction": "prograde",
            "model": "clohessy_wiltshire_secular",
        }
    if float(horizon_sec) <= 0.0 or float(horizon_sec) > 0.2 * period:
        return {
            "accepted": False,
            "reason": "горизонт длиннее линейной модели",
            "dv_m_s": 0.0,
            "direction": "prograde",
            "model": "clohessy_wiltshire_secular",
        }
    dv = float(along_error_m) / (3.0 * float(horizon_sec))
    spare = float(dv_budget_m_s) - float(mission_reserve_m_s)
    if abs(dv) > spare + 1e-12:
        return {
            "accepted": False,
            "reason": "бюджет миссии",
            "dv_m_s": dv,
            "direction": "prograde" if dv >= 0.0 else "retrograde",
            "model": "clohessy_wiltshire_secular",
        }
    return {
        "accepted": True,
        "reason": "",
        "dv_m_s": dv,
        "direction": "prograde" if dv >= 0.0 else "retrograde",
        "model": "clohessy_wiltshire_secular",
    }
