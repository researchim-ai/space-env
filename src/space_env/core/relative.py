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


def _vector3(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def rtn_basis(r_m: np.ndarray, v_m_s: np.ndarray) -> np.ndarray:
    """Columns R,T,N map RTN vectors to inertial vectors; reject degenerate frames."""
    position = _vector3(r_m, "position")
    velocity = _vector3(v_m_s, "velocity")
    radius = np.linalg.norm(position)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("RTN reference radius must be nonzero and finite")
    radial = position / radius
    moment = np.cross(radial, velocity)
    norm = np.linalg.norm(moment)
    if not np.isfinite(norm) or norm <= 1e-12 * max(1.0, np.linalg.norm(velocity)):
        raise ValueError("RTN reference requires nonradial velocity")
    normal = moment / norm
    return np.column_stack((radial, np.cross(normal, radial), normal))


def _frame_rate(reference_r, reference_v, acceleration=None):
    """Instantaneous RTN angular velocity, expressed in RTN (rad/s).

    Without acceleration assumes zero normal acceleration. Supplying actual
    inertial acceleration also accounts for precession of the orbital plane.
    """
    r = _vector3(reference_r, "reference position")
    v = _vector3(reference_v, "reference velocity")
    basis = rtn_basis(r, v)
    radius = np.linalg.norm(r)
    h = np.linalg.norm(np.cross(r, v))
    normal_a = (
        0.0
        if acceleration is None
        else basis[:, 2] @ _vector3(acceleration, "reference acceleration")
    )
    return basis, np.array([radius * normal_a / h, 0.0, h / radius**2])


def relative_state_rtn(
    position, velocity, reference_r, reference_v, *, reference_acceleration=None
):
    """[rho, d(rho)/dt] in rotating RTN, not merely rotated inertial delta-v.

    Exact instantaneous coordinate transform; no CW linearization or propagation.
    Reference acceleration is needed when its orbital plane precesses.
    """
    basis, omega = _frame_rate(reference_r, reference_v, reference_acceleration)
    rho = basis.T @ (
        _vector3(position, "position") - _vector3(reference_r, "reference position")
    )
    rate = basis.T @ (
        _vector3(velocity, "velocity") - _vector3(reference_v, "reference velocity")
    ) - np.cross(omega, rho)
    return np.concatenate((rho, rate))


def rtn_state_to_eci(
    relative_state, reference_r, reference_v, *, reference_acceleration=None
):
    """Inverse of relative_state_rtn; return absolute inertial position/velocity."""
    state = np.asarray(relative_state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("relative state must be a finite six-vector")
    basis, omega = _frame_rate(reference_r, reference_v, reference_acceleration)
    return np.concatenate(
        (
            _vector3(reference_r, "reference position") + basis @ state[:3],
            _vector3(reference_v, "reference velocity")
            + basis @ (state[3:] + np.cross(omega, state[:3])),
        )
    )


def eci_to_rtn(vec_eci: np.ndarray, r_m: np.ndarray, v_m_s: np.ndarray) -> np.ndarray:
    return rtn_basis(r_m, v_m_s).T @ np.asarray(vec_eci, dtype=float)


def inertial_delta_v_m_s(
    position: np.ndarray, velocity: np.ndarray, dv_m_s: float, direction: str
) -> np.ndarray:
    """Приращение скорости в ECI. normal — вдоль удельного момента, как команда манёвра движка."""
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6 or dv_m_s == 0.0:
        return np.zeros(3)
    if direction == "retrograde":
        axis = -velocity / speed
    elif direction == "normal":
        moment = np.cross(position, velocity)
        norm = float(np.linalg.norm(moment))
        axis = moment / norm if norm > 0.0 else velocity / speed
    elif direction == "radial":
        axis = np.asarray(position, dtype=float) / np.linalg.norm(position)
    else:
        axis = velocity / speed
    return axis * float(dv_m_s)


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
    Дискретный минимум, не непрерывная гарантия безопасности. Последняя точка
    точно на horizon, без выхода за интервал. Шаг 1 с: при скорости сближения сотни м/с ошибка промаха — сотни метров, для порога в километры
    грубо, поэтому шаг оставляем секундным.
    """
    cur = np.asarray(state, dtype=float).copy()
    if cur.shape != (6,) or not np.all(np.isfinite(cur)):
        raise ValueError("relative state must be a finite six-vector")
    if not np.isfinite(n_rad_s) or not np.isfinite(horizon_sec) or horizon_sec < 0:
        raise ValueError("mean motion and nonnegative horizon must be finite")
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("sampling step must be positive and finite")
    steps = int(np.ceil(horizon_sec / dt_sec))
    if steps > 1_000_000:
        raise ValueError("closest-approach sampling exceeds one million intervals")
    best_d, best_t, best = float(np.linalg.norm(cur[:3])), 0.0, cur.copy()
    # Evaluate each sample from the initial state, ending exactly at the horizon.
    for index in range(1, steps + 1):
        clock = min(index * dt_sec, horizon_sec)
        candidate = cw_propagate(cur, n_rad_s, clock)
        distance = float(np.linalg.norm(candidate[:3]))
        if distance < best_d:
            best_d, best_t, best = distance, clock, candidate
    return best_d, best_t, best
