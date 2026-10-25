"""
Второе тело интегрируется тем же Кауэллом, что и свой аппарат.
Ковариация относительного состояния в RTN едет матрицей Клохесси–Уилтшира
и шумом процесса: P ← Φ P Φᵀ + Q.
Вероятность — интеграл гауссианы по диску в плоскости, перпендикулярной скорости сближения.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np

from space_env.core.probability import bplane_probability
from space_env.core.propagate import propagate_state
from space_env.core.relative import closest_approach, cw_propagate, cw_stm, eci_to_rtn, mean_motion_rad_s, rtn_basis


@dataclass
class PropagationContext:
    mass_kg: float
    drag_area_m2: float
    drag_coefficient: float
    atmosphere: str = "table"
    f107: float = 150.0
    f107a: float = 150.0
    ap: float = 4.0
    density_scale: float = 1.0
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    when: object = None
    include_drag: bool = True
    srp_area_m2: float = 0.0
    srp_coefficient: float = 1.2
    include_third_body: bool = False


def _advanced(ctx: PropagationContext, dt_sec: float) -> PropagationContext:
    """Следующий вызов не повторяет исходный when."""
    if isinstance(ctx.when, datetime):
        return replace(ctx, when=ctx.when + timedelta(seconds=dt_sec))
    return ctx


def _step_state(r: np.ndarray, v: np.ndarray, dt: float, ctx: PropagationContext) -> tuple[np.ndarray, np.ndarray, PropagationContext]:
    r_end, v_end, _samples = propagate_state(
        r,
        v,
        dt,
        mass_kg=ctx.mass_kg,
        drag_area_m2=ctx.drag_area_m2,
        drag_coefficient=ctx.drag_coefficient,
        include_drag=ctx.include_drag,
        atmosphere=ctx.atmosphere,
        f107=ctx.f107,
        f107a=ctx.f107a,
        ap=ctx.ap,
        density_scale=ctx.density_scale,
        lat_deg=ctx.lat_deg,
        lon_deg=ctx.lon_deg,
        when=ctx.when,
        srp_area_m2=ctx.srp_area_m2,
        srp_coefficient=ctx.srp_coefficient,
        include_third_body=ctx.include_third_body,
        substep_sec=min(10.0, dt),
    )
    return r_end, v_end, _advanced(ctx, dt)


def process_noise(dt_sec: float) -> np.ndarray:
    """
    Дискретный шум белого ускорения.
    Вдоль трека 5·10⁻³ м/с², в радиусе и по нормали 1·10⁻³ м/с² —
    консервативный шум фильтра на дуге минут, не модель атмосферы.
    """
    q = np.zeros((6, 6))
    for index, sigma_a in ((0, 1e-3), (1, 5e-3), (2, 1e-3)):
        var = sigma_a ** 2
        q[index, index] = var * dt_sec**3 / 3.0
        q[index + 3, index + 3] = var * dt_sec
        q[index, index + 3] = var * dt_sec**2 / 2.0
        q[index + 3, index] = q[index, index + 3]
    return q


class NumericalEncounter:
    """Состояние второго объекта и ковариация на текущий момент."""

    def __init__(
        self,
        r_m: np.ndarray,
        v_m_s: np.ndarray,
        covariance: np.ndarray,
        tca_sec: float,
        hard_body_m: float,
        debris: PropagationContext,
    ) -> None:
        self.r = np.asarray(r_m, dtype=float).copy()
        self.v = np.asarray(v_m_s, dtype=float).copy()
        self.covariance = np.asarray(covariance, dtype=float).copy()
        self.tca_sec = tca_sec
        self.hard_body_m = hard_body_m
        self.debris = debris
        self.passed = False
        self.frozen_miss_m = 0.0
        self.frozen_pc = 0.0
        self.frozen_radial_m = 0.0
        self.frozen_cross_m = 0.0

    def advance(self, dt_sec: float, n_rad_s: float) -> None:
        self.r, self.v, self.debris = _step_state(self.r, self.v, dt_sec, self.debris)
        stm = cw_stm(n_rad_s, dt_sec)
        updated = stm @ self.covariance @ stm.T + process_noise(dt_sec)
        self.covariance = 0.5 * (updated + updated.T)

    def predict(
        self,
        r_primary: np.ndarray,
        v_primary: np.ndarray,
        primary: PropagationContext,
        horizon_sec: float,
    ) -> tuple[float, float, float, float]:
        """Промах, Pc, радиальная и боковая составляющие в точке наибольшего сближения."""
        if self.passed:
            return self.frozen_miss_m, self.frozen_pc, self.frozen_radial_m, self.frozen_cross_m
        r1 = np.asarray(r_primary, dtype=float).copy()
        v1 = np.asarray(v_primary, dtype=float).copy()
        r2 = self.r.copy()
        v2 = self.v.copy()
        horizon = max(horizon_sec, 1.0)
        best_d = float(np.linalg.norm(r2 - r1))
        best_t = 0.0
        best = (r1.copy(), v1.copy(), r2.copy(), v2.copy())
        coarse = 20.0
        t = 0.0
        primary_ctx = primary
        debris_ctx = self.debris
        while t < horizon - 1e-6:
            h = min(coarse, horizon - t)
            r1, v1, primary_ctx = _step_state(r1, v1, h, primary_ctx)
            r2, v2, debris_ctx = _step_state(r2, v2, h, debris_ctx)
            t += h
            distance = float(np.linalg.norm(r2 - r1))
            if distance < best_d:
                best_d = distance
                best_t = t
                best = (r1.copy(), v1.copy(), r2.copy(), v2.copy())
        # Уточнение вокруг грубого минимума шагом 0.5 с.
        r1, v1, r2, v2, primary_ctx, debris_ctx = self._state_at(
            np.asarray(r_primary), np.asarray(v_primary), primary, max(0.0, best_t - 20.0)
        )
        t = max(0.0, best_t - 20.0)
        end = min(horizon, best_t + 20.0)
        while t < end - 1e-6:
            h = min(0.5, end - t)
            r1, v1, primary_ctx = _step_state(r1, v1, h, primary_ctx)
            r2, v2, debris_ctx = _step_state(r2, v2, h, debris_ctx)
            t += h
            distance = float(np.linalg.norm(r2 - r1))
            if distance < best_d:
                best_d = distance
                best_t = t
                best = (r1.copy(), v1.copy(), r2.copy(), v2.copy())
        r1, v1, r2, v2 = best
        dr = eci_to_rtn(r2 - r1, r1, v1)
        dv = eci_to_rtn(v2 - v1, r1, v1)
        n = mean_motion_rad_s(float(np.linalg.norm(r1)))
        mapped = cw_stm(n, best_t) @ self.covariance @ cw_stm(n, best_t).T
        probability = bplane_probability(dr, dv, mapped, self.hard_body_m)
        return best_d, probability, float(dr[0]), float(dr[2])

    def _state_at(
        self,
        r1: np.ndarray,
        v1: np.ndarray,
        primary: PropagationContext,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r2, v2 = self.r.copy(), self.v.copy()
        debris_ctx = self.debris
        if dt <= 1e-6:
            return r1.copy(), v1.copy(), r2, v2, primary, debris_ctx
        r1, v1, primary = _step_state(r1, v1, dt, primary)
        r2, v2, debris_ctx = _step_state(r2, v2, dt, debris_ctx)
        return r1, v1, r2, v2, primary, debris_ctx


def secondary_from_geometry(
    r_primary: np.ndarray,
    v_primary: np.ndarray,
    radial_m: float,
    cross_m: float,
    approach_m_s: float,
    tca_in_sec: float,
    sigma_m: float,
    tca_sec: float,
    hard_body_m: float,
    debris: PropagationContext,
) -> NumericalEncounter:
    """Начальное ECI второго тела: геометрия TCA, откатанная матрицей CW, затем оба интегрируются численно."""
    n = mean_motion_rad_s(float(np.linalg.norm(r_primary)))
    at_tca = np.array([radial_m, 0.0, cross_m, 0.0, -approach_m_s, 0.0])
    rel = cw_propagate(at_tca, n, -tca_in_sec)
    basis = rtn_basis(r_primary, v_primary)
    r2 = np.asarray(r_primary, dtype=float) + basis @ rel[:3]
    v2 = np.asarray(v_primary, dtype=float) + basis @ rel[3:]
    along = max(1500.0, 8.0 * sigma_m)
    vel = 0.02
    covariance = np.diag([sigma_m**2, along**2, sigma_m**2, vel**2, (4.0 * vel) ** 2, vel**2])
    return NumericalEncounter(r2, v2, covariance, tca_sec, hard_body_m, debris)


def miss_of_relative_state(rel: np.ndarray, n: float, horizon: float) -> float:
    """Вспомогательная проверка CW-инициализации, не используется в шаге."""
    distance, _t, _state = closest_approach(rel, n, horizon, dt_sec=1.0)
    return distance
