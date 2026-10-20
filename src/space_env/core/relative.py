"""
Относительное движение Клохесси–Уилтшира и переход ECI ↔ RTN.
Матрица перехода — стандартная форма Валладо (x — радиус, y — вдоль трека, z — по нормали).
Линеаризация пригодна, пока расстояние много меньше радиуса орбиты.
"""
from __future__ import annotations

import numpy as np

from space_env.core.constants import MU_EARTH


def mean_motion_rad_s(radius_m: float) -> float:
    """Среднее движение круговой орбиты этого радиуса, рад/с."""
    return float(np.sqrt(MU_EARTH / max(radius_m, 1.0) ** 3))


def rtn_basis(r_m: np.ndarray, v_m_s: np.ndarray) -> np.ndarray:
    """Столбцы: радиальная, вдоль трека, нормаль. Умножение на вектор RTN даёт ECI."""
    radial = np.asarray(r_m, dtype=float)
    radial = radial / np.linalg.norm(radial)
    normal = np.cross(radial, np.asarray(v_m_s, dtype=float))
    normal = normal / np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return np.column_stack((radial, along, normal))


def eci_to_rtn(vec_eci: np.ndarray, r_m: np.ndarray, v_m_s: np.ndarray) -> np.ndarray:
    return rtn_basis(r_m, v_m_s).T @ np.asarray(vec_eci, dtype=float)


def cw_stm(n_rad_s: float, t_sec: float) -> np.ndarray:
    """6×6 матрица перехода состояния [x, y, z, vx, vy, vz]."""
    n = float(n_rad_s)
    t = float(t_sec)
    if abs(n) < 1e-12:
        stm = np.eye(6)
        stm[0, 3] = t
        stm[1, 4] = t
        stm[2, 5] = t
        return stm
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    stm = np.zeros((6, 6))
    stm[0, 0] = 4.0 - 3.0 * c
    stm[0, 3] = s / n
    stm[0, 4] = 2.0 * (1.0 - c) / n
    stm[1, 0] = 6.0 * (s - nt)
    stm[1, 1] = 1.0
    stm[1, 3] = 2.0 * (c - 1.0) / n
    stm[1, 4] = (4.0 * s - 3.0 * nt) / n
    stm[2, 2] = c
    stm[2, 5] = s / n
    stm[3, 0] = 3.0 * n * s
    stm[3, 3] = c
    stm[3, 4] = 2.0 * s
    stm[4, 0] = 6.0 * n * (c - 1.0)
    stm[4, 3] = -2.0 * s
    stm[4, 4] = 4.0 * c - 3.0
    stm[5, 2] = -n * s
    stm[5, 5] = c
    return stm


def cw_propagate(state: np.ndarray, n_rad_s: float, t_sec: float) -> np.ndarray:
    return cw_stm(n_rad_s, t_sec) @ np.asarray(state, dtype=float)


def closest_approach(
    state: np.ndarray,
    n_rad_s: float,
    horizon_sec: float,
    dt_sec: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    """
    Минимальное расстояние на интервале [0, horizon], секунда этого минимума и состояние в нём.
    Шаг 1 с: при скорости сближения сотни м/с ошибка промаха — сотни метров, для порога в километры
    грубо, поэтому шаг оставляем секундным.
    """
    cur = np.asarray(state, dtype=float).copy()
    if horizon_sec <= 0.0:
        return float(np.linalg.norm(cur[:3])), 0.0, cur
    stm = cw_stm(n_rad_s, dt_sec)
    steps = int(np.ceil(horizon_sec / dt_sec))
    best_d = float(np.linalg.norm(cur[:3]))
    best_t = 0.0
    best = cur.copy()
    for index in range(1, steps + 1):
        cur = stm @ cur
        distance = float(np.linalg.norm(cur[:3]))
        if distance < best_d:
            best_d = distance
            best_t = index * dt_sec
            best = cur.copy()
    return best_d, best_t, best
