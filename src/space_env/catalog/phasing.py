"""Вдольтрековая ошибка пары относительно антипода лидера. Для удержания 180°."""
from __future__ import annotations

import numpy as np

from space_env.core.orbit import Orbit
from space_env.simulation.engine import SimulationEngine


def phase_error_m(r_leader_m: np.ndarray, v_leader_m_s: np.ndarray, r_follower_m: np.ndarray) -> float:
    """
    Дуга от антипода ведущего до ведомого, метры.
    Ноль при фазе 180°. Знак берётся по моменту импульса ведущего, поэтому разница радиусов
    в ошибку не входит: её даёт только сдвиг по орбите.
    """
    leader = np.asarray(r_leader_m, dtype=float)
    follower = np.asarray(r_follower_m, dtype=float)
    radius = float(np.linalg.norm(leader))
    follower_radius = float(np.linalg.norm(follower))
    if radius < 1.0 or follower_radius < 1.0:
        return 0.0
    antipode = -leader / radius
    follower_hat = follower / follower_radius
    cosine = float(np.clip(np.dot(antipode, follower_hat), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    moment = np.cross(leader, np.asarray(v_leader_m_s, dtype=float))
    sign = 1.0 if float(np.dot(np.cross(antipode, follower_hat), moment)) >= 0.0 else -1.0
    return sign * angle * radius


def add_prograde_dv(engine: SimulationEngine, dv_m_s: float) -> None:
    """Меняет скорость, не списывая бюджет Δv. Так задаётся начальная ошибка большой полуоси."""
    velocity = np.asarray(engine.orbit.v, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        raise ValueError("скорость нулевая")
    updated = velocity + velocity / speed * float(dv_m_s)
    engine.orbit = Orbit.from_vectors(engine.orbit.r, updated, engine.orbit.epoch)
