"""
Короткие эталоны ядра. Допуск записан до сравнения и не подгоняется под ответ.

Двухтелая задача: относительный уход энергии и момента за один оборот при шаге 10 с
не больше 1e-8. Это запас на метод RK4, не измеренный остаток конкретного прогона.
J2: оскулирующий узел за 6 ч против вековой формулы Кёртиса, относительный допуск 0.25.
Короткопериодическая часть в вековой формуле отсутствует, поэтому допуск шире.
Сопротивление: с таблицей плотности радиус через час меньше, чем без неё.
Это знак эффекта, не эталон US Standard.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from space_env import __version__
from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.time import Time

_TWO_BODY_TOLERANCE = 1.0e-8
_J2_RELATIVE = 0.25


def _energy(r: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * float(np.dot(v, v)) - MU_EARTH / float(np.linalg.norm(r))


def _momentum(r: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(r, v)))


def _raan_deg(r: np.ndarray, v: np.ndarray) -> float:
    angular = np.cross(r, v)
    node = np.cross(np.array([0.0, 0.0, 1.0]), angular)
    return float(np.rad2deg(math.atan2(node[1], node[0])))


def _case(name: str, measured: float, tolerance: float, applicability: str, source: str) -> dict[str, Any]:
    return {
        "name": name,
        "measured": measured,
        "tolerance": tolerance,
        "passed": abs(measured) <= tolerance,
        "applicability": applicability,
        "source": source,
        "status": "verified" if abs(measured) <= tolerance else "experimental",
    }


def core_report() -> dict[str, Any]:
    """Считает эталоны ядра и возвращает отчёт с измеренной ошибкой."""
    a = 6_800_000.0
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(a, 0.001, math.radians(51.6), 0.2, 0.1, 0.3, epoch)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        period,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=False,
        include_drag=False,
        substep_sec=10.0,
    )
    energy_error = abs(_energy(r, v) - _energy(orbit.r, orbit.v)) / abs(_energy(orbit.r, orbit.v))
    momentum_error = abs(_momentum(r, v) - _momentum(orbit.r, orbit.v)) / _momentum(orbit.r, orbit.v)
    inclined = Orbit.from_keplerian(a, 0.0, math.radians(51.6), math.radians(40.0), 0.0, 0.0, epoch)
    hours = 6.0
    node_r, node_v, _samples = propagate_state(
        inclined.r,
        inclined.v,
        hours * 3600.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=True,
        include_drag=False,
        substep_sec=20.0,
    )
    delta = _raan_deg(node_r, node_v) - 40.0
    if delta > 180.0:
        delta -= 360.0
    if delta < -180.0:
        delta += 360.0
    mean_motion = math.sqrt(MU_EARTH / a**3)
    analytical = math.degrees(
        -1.5 * mean_motion * J2_EARTH * (R_EARTH_EQUATOR / a) ** 2 * math.cos(math.radians(51.6)) * hours * 3600.0
    )
    j2_error = abs(delta - analytical) / abs(analytical)
    coast = Orbit.from_keplerian(a, 0.0, math.radians(51.6), 0.0, 0.0, 0.0, epoch)
    common = dict(mass_kg=100.0, drag_area_m2=1.0, drag_coefficient=2.2, include_j2=True, substep_sec=30.0)
    dragged, _vd, _samples = propagate_state(coast.r, coast.v, 3600.0, include_drag=True, **common)
    clean, _vc, _samples = propagate_state(coast.r, coast.v, 3600.0, include_drag=False, **common)
    radius_drop_m = float(np.linalg.norm(clean) - np.linalg.norm(dragged))
    cases = [
        _case(
            "two_body_energy",
            energy_error,
            _TWO_BODY_TOLERANCE,
            "Один оборот, шаг 10 с, J2, сопротивление, давление и третье тело выключены.",
            "Удельная энергия v²/2 − μ/r сохраняется в задаче двух тел.",
        ),
        _case(
            "two_body_angular_momentum",
            momentum_error,
            _TWO_BODY_TOLERANCE,
            "Тот же оборот и тот же шаг.",
            "Модуль момента |r × v| сохраняется в задаче двух тел.",
        ),
        _case(
            "j2_node",
            j2_error,
            _J2_RELATIVE,
            "Круговая орбита 6800 км, 51.6°, 6 ч, без сопротивления. Оскулирующий узел против вековой формулы.",
            "Кёртис: dΩ/dt = −(3/2) n J2 (Re/a)² cos i.",
        ),
        {
            "name": "drag_lowers_radius",
            "measured": radius_drop_m,
            "tolerance": 0.0,
            "passed": radius_drop_m > 0.0,
            "applicability": "Таблица среднего Солнца, час, площадь 1 м², Cd 2.2. Не эталон плотности.",
            "source": "Сопротивление уменьшает большую полуось, поэтому радиус через час меньше, чем без него.",
            "status": "approximate" if radius_drop_m > 0.0 else "experimental",
        },
    ]
    return {
        "version": __version__,
        "profile": "core",
        "cases": cases,
        "passed": all(item["passed"] for item in cases),
    }


def applicability_limits() -> list[dict[str, str]]:
    """Граница профиля. Эти пункты не выдаются за проверенный эталон."""
    return [
        {
            "name": "pc",
            "status": "not_independently_validated",
            "note": "Pc считается в коде и не сверен с независимым эталоном. Это не эксплуатационная оценка.",
        },
        {
            "name": "rk4_stage_epoch",
            "status": "shared_substep_epoch",
            "note": "Четыре стадии RK4 внутри одного подынтервала делят один момент и одну широту.",
        },
        {
            "name": "teme_to_gcrf",
            "status": "not_implemented",
            "note": "Перехода TEME в численный кадр нет. Движок принимает только GCRF.",
        },
        {
            "name": "iau_2000a",
            "status": "not_implemented",
            "note": "Звёздное время — десять членов IAU 1980. Ряд IAU 2000A не считается.",
        },
    ]


def main() -> None:
    print(json.dumps(core_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
