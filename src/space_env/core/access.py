"""Подспутниковая точка, угол от надира и точки трассы для сценария съёмки."""
from __future__ import annotations

import numpy as np

from space_env.core.frames import ecef_to_eci, ecef_to_geodetic, eci_to_ecef, geodetic_to_ecef
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.time import Time


def subsatellite_latlon(r_eci_m: np.ndarray, time: Time) -> tuple[float, float]:
    """Геодезические широта и долгота подспутниковой точки, градусы."""
    lat, lon, _alt = ecef_to_geodetic(eci_to_ecef(r_eci_m, time))
    return lat, lon


def off_nadir_deg(r_eci_m: np.ndarray, lat_deg: float, lon_deg: float, time: Time) -> float:
    """
    Угол между надиром и направлением на точку поверхности, градусы.
    Надирный прибор с половинным полем зрения видит цель, если угол не больше этого поля.
    """
    r = np.asarray(r_eci_m, dtype=float)
    target = ecef_to_eci(geodetic_to_ecef(lat_deg, lon_deg, 0.0), time)
    look = target - r
    nadir = -r
    denom = float(np.linalg.norm(look) * np.linalg.norm(nadir))
    if denom < 1.0:
        return 180.0
    cosine = float(np.clip(np.dot(look, nadir) / denom, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def ground_track_latlon(
    orbit: Orbit,
    offsets_sec: list[float],
    *,
    mass_kg: float = 100.0,
    drag_area_m2: float = 0.9,
    drag_coefficient: float = 2.2,
) -> list[tuple[float, float]]:
    """Широта и долгота подспутниковой точки в заданные секунды от эпохи орбиты. Та же пропагация, что у движка."""
    r = orbit.r.copy()
    v = orbit.v.copy()
    elapsed = 0.0
    points: list[tuple[float, float]] = []
    ordered = sorted(offsets_sec)
    for offset in ordered:
        dt = offset - elapsed
        if dt > 0.0:
            r, v, _samples = propagate_state(
                r,
                v,
                dt,
                mass_kg=mass_kg,
                drag_area_m2=drag_area_m2,
                drag_coefficient=drag_coefficient,
            )
            elapsed = offset
        points.append(subsatellite_latlon(r, orbit.epoch + elapsed))
    return points
