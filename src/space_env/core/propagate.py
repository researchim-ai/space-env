"""Численная пропагация Кауэлла, RK4. Шаг по умолчанию 10 с."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from space_env.core.frames import ecef_to_geodetic, eci_to_ecef
from space_env.core.gravity import acceleration_m_s2
from space_env.core.time import Time


def _rk4(
    r: np.ndarray,
    v: np.ndarray,
    h: float,
    *,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
    include_j2: bool,
    include_drag: bool,
    atmosphere: str,
    f107: float,
    f107a: float,
    ap: float,
    density_scale: float,
    lat_deg: float,
    lon_deg: float,
    when,
    srp_area_m2: float,
    srp_coefficient: float,
    include_third_body: bool,
) -> tuple[np.ndarray, np.ndarray]:
    def deriv(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acc = acceleration_m_s2(
            pos,
            vel,
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            include_j2=include_j2,
            include_drag=include_drag,
            atmosphere=atmosphere,
            f107=f107,
            f107a=f107a,
            ap=ap,
            density_scale=density_scale,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            when=when,
            srp_area_m2=srp_area_m2,
            srp_coefficient=srp_coefficient,
            include_third_body=include_third_body,
        )
        return vel, acc

    v1, a1 = deriv(r, v)
    v2, a2 = deriv(r + 0.5 * h * v1, v + 0.5 * h * a1)
    v3, a3 = deriv(r + 0.5 * h * v2, v + 0.5 * h * a2)
    v4, a4 = deriv(r + h * v3, v + h * a3)
    r_new = r + (h / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
    v_new = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return r_new, v_new


def propagate_state(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    dt_sec: float,
    *,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
    include_j2: bool = True,
    include_drag: bool = True,
    substep_sec: float = 10.0,
    atmosphere: str = "table",
    f107: float = 150.0,
    f107a: float = 150.0,
    ap: float = 4.0,
    density_scale: float = 1.0,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    when=None,
    srp_area_m2: float = 0.0,
    srp_coefficient: float = 1.2,
    include_third_body: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    """
    Пропагирует состояние на dt_sec.
    Возвращает конечные r, v и отсчёты (смещение времени, r, v), включая 0 и dt_sec.
    """
    if dt_sec <= 0.0:
        raise ValueError("dt_sec должен быть положительным")
    r = np.asarray(r_m, dtype=float).copy()
    v = np.asarray(v_m_s, dtype=float).copy()
    samples: list[tuple[float, np.ndarray, np.ndarray]] = [(0.0, r.copy(), v.copy())]
    elapsed = 0.0
    step = max(substep_sec, 1e-3)
    moment = when
    while elapsed < dt_sec - 1e-9:
        h = min(step, dt_sec - elapsed)
        step_lat, step_lon = lat_deg, lon_deg
        if isinstance(moment, datetime):
            step_lat, step_lon, _alt = ecef_to_geodetic(eci_to_ecef(r, Time(moment)))
        r, v = _rk4(
            r,
            v,
            h,
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            include_j2=include_j2,
            include_drag=include_drag,
            atmosphere=atmosphere,
            f107=f107,
            f107a=f107a,
            ap=ap,
            density_scale=density_scale,
            lat_deg=step_lat,
            lon_deg=step_lon,
            when=moment,
            srp_area_m2=srp_area_m2,
            srp_coefficient=srp_coefficient,
            include_third_body=include_third_body,
        )
        elapsed += h
        if isinstance(moment, datetime):
            moment = moment + timedelta(seconds=h)
        samples.append((elapsed, r.copy(), v.copy()))
    return r, v, samples
