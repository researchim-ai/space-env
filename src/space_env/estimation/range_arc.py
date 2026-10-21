"""
Оценка по геометрическим дальностям одной станции.
Один параметр — сдвиг вдоль скорости. Три параметра — ещё радиус и скорость вдоль трека.
Истина и модель летят одним Кауэллом. Поперек трека дуга почти слепая, эта дисперсия задаётся отдельно.
Это не фильтр сети слежения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.core.frames import eci_to_ecef, geodetic_to_ecef
from space_env.core.propagate import propagate_state
from space_env.core.relative import rtn_basis
from space_env.core.time import Time


@dataclass(frozen=True)
class RangeFix:
    along_track_m: float
    rms_m: float
    iterations: int
    samples: int


@dataclass(frozen=True)
class ArcStateFix:
    """Радиус, вдоль трека, поперёк трека и скорость вдоль трека. Ковариация 6×6 в RTN на эпоху дуги."""

    radial_m: float
    along_m: float
    along_rate_m_s: float
    covariance_rtn_m2: np.ndarray
    rms_m: float
    iterations: int
    samples: int
    cross_m: float = 0.0


def geometric_range_m(
    r_eci_m: np.ndarray,
    epoch: Time,
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
) -> float:
    station = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return float(np.linalg.norm(eci_to_ecef(np.asarray(r_eci_m, dtype=float), epoch) - station))


def _shifted(r_m: np.ndarray, v_m_s: np.ndarray, along_m: float) -> np.ndarray:
    velocity = np.asarray(v_m_s, dtype=float)
    speed = float(np.linalg.norm(velocity))
    return np.asarray(r_m, dtype=float) + along_m * velocity / speed


def _range_at(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    along_m: float,
    dt_sec: float,
    epoch: Time,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
) -> float:
    position = _shifted(r_m, v_m_s, along_m)
    velocity = np.asarray(v_m_s, dtype=float)
    if dt_sec > 0.0:
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            dt_sec,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_drag=False,
            include_j2=True,
            include_third_body=False,
            substep_sec=10.0,
        )
    return geometric_range_m(position, epoch + dt_sec, lat_deg, lon_deg, alt_m)


def fit_along_track_m(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    epoch: Time,
    ranges_m: list[tuple[float, float]],
    lat_deg: float,
    lon_deg: float,
    *,
    alt_m: float = 0.0,
    start_m: float = 0.0,
    step_m: float = 50.0,
    max_iterations: int = 8,
) -> RangeFix:
    """
    Гаусс–Ньютон по скаляру вдоль начальной скорости.
    ranges_m — пары (секунды от эпохи, дальность, м). Якобиан — конечная разность step_m.
    """
    if len(ranges_m) < 3:
        raise ValueError("на дуге нужно хотя бы три дальности")
    estimate = float(start_m)
    residual = np.zeros(len(ranges_m))
    for iteration in range(1, max_iterations + 1):
        jacobian = np.zeros(len(ranges_m))
        for index, (offset, measured) in enumerate(ranges_m):
            predicted = _range_at(r_m, v_m_s, estimate, offset, epoch, lat_deg, lon_deg, alt_m)
            predicted_step = _range_at(r_m, v_m_s, estimate + step_m, offset, epoch, lat_deg, lon_deg, alt_m)
            residual[index] = measured - predicted
            jacobian[index] = (predicted_step - predicted) / step_m
        normal = float(np.dot(jacobian, jacobian))
        if normal < 1e-12:
            break
        delta = float(np.dot(jacobian, residual) / normal)
        estimate += delta
        if abs(delta) < 1.0:
            rms = float(np.sqrt(np.mean(residual**2)))
            return RangeFix(estimate, rms, iteration, len(ranges_m))
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return RangeFix(estimate, rms, max_iterations, len(ranges_m))


def _apply_rtn(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    radial_m: float,
    along_m: float,
    along_rate_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(r_m, dtype=float)
    velocity = np.asarray(v_m_s, dtype=float)
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    radial = position / radius
    along = velocity / speed
    return position + radial_m * radial + along_m * along, velocity + along_rate_m_s * along


def _range_rtn(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    radial_m: float,
    along_m: float,
    along_rate_m_s: float,
    dt_sec: float,
    epoch: Time,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
) -> float:
    position, velocity = _apply_rtn(r_m, v_m_s, radial_m, along_m, along_rate_m_s)
    if dt_sec > 0.0:
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            dt_sec,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_drag=False,
            include_j2=True,
            include_third_body=False,
            substep_sec=10.0,
        )
    return geometric_range_m(position, epoch + dt_sec, lat_deg, lon_deg, alt_m)


def fit_rtn_arc(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    epoch: Time,
    ranges_m: list[tuple[float, float]],
    lat_deg: float,
    lon_deg: float,
    *,
    alt_m: float = 0.0,
    range_sigma_m: float = 10.0,
    unobserved_m: float = 10_000.0,
    unobserved_m_s: float = 10.0,
    step_m: float = 50.0,
    step_m_s: float = 0.05,
    max_iterations: int = 8,
) -> ArcStateFix:
    """
    Гаусс–Ньютон: радиус, вдоль трека, скорость вдоль трека.
    Ковариация — range_sigma² (JᵀJ)⁻¹ на этих трёх осях.
    Поперек трека и остальные скорости не наблюдаются и получают unobserved_*.
    """
    if len(ranges_m) < 4:
        raise ValueError("для трёх параметров нужно хотя бы четыре дальности")
    if range_sigma_m <= 0.0:
        raise ValueError("сигма дальности должна быть положительной")
    estimate = np.zeros(3)
    steps = np.array([step_m, step_m, step_m_s])
    residual = np.zeros(len(ranges_m))
    jacobian = np.zeros((len(ranges_m), 3))
    iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        for index, (offset, measured) in enumerate(ranges_m):
            predicted = _range_rtn(r_m, v_m_s, *estimate, offset, epoch, lat_deg, lon_deg, alt_m)
            residual[index] = measured - predicted
            for axis in range(3):
                trial = estimate.copy()
                trial[axis] += steps[axis]
                nudged = _range_rtn(r_m, v_m_s, *trial, offset, epoch, lat_deg, lon_deg, alt_m)
                jacobian[index, axis] = (nudged - predicted) / steps[axis]
        normal = jacobian.T @ jacobian
        delta = np.linalg.lstsq(normal, jacobian.T @ residual, rcond=None)[0]
        estimate = estimate + delta
        iterations = iteration
        if float(np.linalg.norm(delta[:2])) < 1.0 and abs(float(delta[2])) < 1e-3:
            break
    parameter_cov = (range_sigma_m**2) * np.linalg.pinv(jacobian.T @ jacobian)
    covariance = np.diag(
        [unobserved_m**2, unobserved_m**2, unobserved_m**2, unobserved_m_s**2, unobserved_m_s**2, unobserved_m_s**2]
    ).astype(float)
    # Порядок RTN: радиус, вдоль, поперёк, скорости в том же порядке.
    # Скорость вдоль трека — индекс 4. Радиальная скорость дугой не оценена.
    index_map = (0, 1, 4)
    for row, row_axis in enumerate(index_map):
        for column, column_axis in enumerate(index_map):
            covariance[row_axis, column_axis] = parameter_cov[row, column]
    covariance = 0.5 * (covariance + covariance.T)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return ArcStateFix(
        float(estimate[0]),
        float(estimate[1]),
        float(estimate[2]),
        covariance,
        rms,
        iterations,
        len(ranges_m),
        0.0,
    )


def _apply_cross(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    radial_m: float,
    along_m: float,
    cross_m: float,
    along_rate_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(r_m, dtype=float)
    velocity = np.asarray(v_m_s, dtype=float)
    basis = rtn_basis(position, velocity)
    position = position + basis @ np.array([radial_m, along_m, cross_m])
    velocity = velocity + basis @ np.array([0.0, along_rate_m_s, 0.0])
    return position, velocity


def fit_two_station_arc(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    epoch: Time,
    ranges_m: list[tuple[float, float, float, float]],
    *,
    range_sigma_m: float = 20.0,
    unobserved_m_s: float = 10.0,
    step_m: float = 40.0,
    step_m_s: float = 0.04,
    max_iterations: int = 8,
) -> ArcStateFix:
    """
    Две станции: радиус, вдоль трека, поперёк трека и скорость вдоль трека.
    ranges_m — (секунды, дальность, широта, долгота). Шум уже в дальности.
    Ковариация этих четырёх осей — range_sigma² (JᵀJ)⁻¹.
    Радиальная скорость и скорость поперёк трека дугой не оценены.
    """
    if len(ranges_m) < 6:
        raise ValueError("для четырёх параметров нужно хотя бы шесть дальностей")
    if range_sigma_m <= 0.0:
        raise ValueError("сигма дальности должна быть положительной")
    estimate = np.zeros(4)
    steps = np.array([step_m, step_m, step_m, step_m_s])

    def predicted(state: np.ndarray, offset: float, lat: float, lon: float) -> float:
        position, velocity = _apply_cross(r_m, v_m_s, state[0], state[1], state[2], state[3])
        if offset > 0.0:
            position, velocity, _samples = propagate_state(
                position,
                velocity,
                offset,
                mass_kg=100.0,
                drag_area_m2=0.0,
                drag_coefficient=0.0,
                include_drag=False,
                include_j2=True,
                include_third_body=False,
                substep_sec=10.0,
            )
        return geometric_range_m(position, epoch + offset, lat, lon)

    residual = np.zeros(len(ranges_m))
    jacobian = np.zeros((len(ranges_m), 4))
    iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        for index, (offset, measured, lat, lon) in enumerate(ranges_m):
            base = predicted(estimate, offset, lat, lon)
            residual[index] = measured - base
            for axis in range(4):
                trial = estimate.copy()
                trial[axis] += steps[axis]
                jacobian[index, axis] = (predicted(trial, offset, lat, lon) - base) / steps[axis]
        delta = np.linalg.lstsq(jacobian.T @ jacobian, jacobian.T @ residual, rcond=None)[0]
        estimate = estimate + delta
        iterations = iteration
        if float(np.linalg.norm(delta[:3])) < 1.0 and abs(float(delta[3])) < 1e-3:
            break
    parameter_cov = (range_sigma_m**2) * np.linalg.pinv(jacobian.T @ jacobian)
    covariance = np.diag([0.0, 0.0, 0.0, unobserved_m_s**2, unobserved_m_s**2, unobserved_m_s**2]).astype(float)
    index_map = (0, 1, 2, 4)
    for row, row_axis in enumerate(index_map):
        for column, column_axis in enumerate(index_map):
            covariance[row_axis, column_axis] = parameter_cov[row, column]
    covariance = 0.5 * (covariance + covariance.T)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return ArcStateFix(
        float(estimate[0]),
        float(estimate[1]),
        float(estimate[3]),
        covariance,
        rms,
        iterations,
        len(ranges_m),
        float(estimate[2]),
    )
