"""Подспутниковая точка, угол от надира и доступность цели.

off_nadir_deg — только угол между надиром и направлением на точку. У цели на
обратной стороне этот угол тоже нулевой: луч смотрит сквозь Землю. Продукт
создаёт target_access, где есть линия визирования.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.core.environment import in_eclipse
from space_env.core.frames import (
    NUMERICAL_FRAME,
    ecef_to_eci,
    ecef_to_geodetic,
    eci_to_ecef,
    elevation_deg,
    geodetic_to_ecef,
    require_numerical_frame,
)
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.sun import sun_position_m
from space_env.core.time import Time


def subsatellite_latlon(r_eci_m: np.ndarray, time: Time) -> tuple[float, float]:
    """Геодезические широта и долгота подспутниковой точки, градусы."""
    lat, lon, _alt = ecef_to_geodetic(eci_to_ecef(r_eci_m, time))
    return lat, lon


def footprint_diameter_m(altitude_m: float, half_fov_deg: float) -> float:
    """Диаметр кадра надирного прибора на плоской площадке, м. Угол — половина поля зрения."""
    return 2.0 * float(altitude_m) * float(np.tan(np.deg2rad(half_fov_deg)))


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


@dataclass(frozen=True)
class TargetAccess:
    """Геометрия цели. visible уже учитывает лимб и правило освещения прибора."""

    off_nadir_deg: float
    elevation_deg: float
    line_of_sight: bool
    in_fov: bool
    target_lit: bool
    satellite_in_umbra: bool
    illumination_required: bool

    @property
    def visible(self) -> bool:
        if not self.line_of_sight or not self.in_fov:
            return False
        if self.illumination_required and (not self.target_lit or self.satellite_in_umbra):
            return False
        return True


def target_access(
    r_eci_m: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    time: Time,
    *,
    half_fov_deg: float,
    alt_m: float = 0.0,
    frame: str = NUMERICAL_FRAME,
    images_in_eclipse: bool = False,
    satellite_in_umbra: bool | None = None,
) -> TargetAccess:
    """Доступность наземной цели для надирного прибора.

    Линия визирования — угол места цели не ниже горизонта эллипсоида. Оптика
    дополнительно требует освещённую цель и аппарат вне umbra. Радар с
    images_in_eclipse это правило освещения не наследует.
    """
    require_numerical_frame(frame)
    position = np.asarray(r_eci_m, dtype=float)
    angle = off_nadir_deg(position, lat_deg, lon_deg, time)
    elevation = elevation_deg(position, lat_deg, lon_deg, alt_m, time)
    sun = sun_position_m(time)
    lit = elevation_deg(sun, lat_deg, lon_deg, 0.0, time) >= 0.0
    umbra = in_eclipse(position, sun) if satellite_in_umbra is None else bool(satellite_in_umbra)
    return TargetAccess(
        off_nadir_deg=angle,
        elevation_deg=elevation,
        line_of_sight=elevation >= -1.0e-8,
        in_fov=angle <= float(half_fov_deg),
        target_lit=lit,
        satellite_in_umbra=umbra,
        illumination_required=not images_in_eclipse,
    )


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
