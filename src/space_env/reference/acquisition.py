"""Point-target acquisition opportunities; ideal steering geometry, not imaging."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import R_EARTH_EQUATOR
from space_env.core.frames import geodetic_to_ecef
from space_env.reference.frames import _vector
from space_env.reference.geometry_events import (
    SearchSettings,
    coverage,
    positive_windows,
    provider_boundaries,
)
from space_env.reference.orbit import _ephemerides
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class AcquisitionTarget(StrictModel):
    id: str = Field(min_length=1)
    lat_deg: float = Field(ge=-90, le=90, strict=True)
    lon_deg: float = Field(ge=-180, le=180, strict=True)
    alt_m: float = Field(default=0.0, ge=0, le=20000, strict=True)
    available_from_sec: float = Field(default=0.0, ge=0, strict=True)
    available_until_sec: float | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def availability(self):
        if not self.id.strip() or (
            self.available_until_sec is not None
            and self.available_until_sec <= self.available_from_sec
        ):
            raise ValueError("target requires nonblank ID and ordered availability")
        return self


class AcquisitionSensor(StrictModel):
    max_off_nadir_deg: float = Field(default=45.0, gt=0, le=85, strict=True)
    min_elevation_deg: float = Field(default=5.0, ge=0, lt=90, strict=True)
    min_sun_elevation_deg: float = Field(default=-90.0, ge=-90, le=90, strict=True)
    max_range_m: float | None = Field(default=None, gt=0, strict=True)
    minimum_dwell_sec: float = Field(default=1.0, gt=0, strict=True)


def acquisition_geometry(provider, target, sensor, clock):
    target, sensor = (
        AcquisitionTarget.model_validate(target),
        AcquisitionSensor.model_validate(sensor),
    )
    r = _vector(provider.state(clock), 6)[:3]
    rotation = provider.context.matrix(provider.epoch.shifted(clock))
    point = geodetic_to_ecef(target.lat_deg, target.lon_deg, target.alt_m)
    ground_gcrs = rotation.T @ point
    line = r - ground_gcrs
    distance, radius = float(np.linalg.norm(line)), float(np.linalg.norm(r))
    if distance == 0 or radius <= R_EARTH_EQUATOR:
        raise ValueError("degenerate acquisition geometry")
    lat, lon = np.deg2rad([target.lat_deg, target.lon_deg])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    elevation = math.degrees(
        math.asin(float(np.clip(np.dot(up, rotation @ line) / distance, -1, 1)))
    )
    off_nadir = math.degrees(
        math.acos(float(np.clip(np.dot(line, r) / (distance * radius), -1, 1)))
    )
    sun, _ = _ephemerides(provider.epoch.shifted(clock))
    to_sun = rotation @ sun - point
    sun_elevation = math.degrees(
        math.asin(float(np.clip(np.dot(up, to_sun) / np.linalg.norm(to_sun), -1, 1)))
    )
    margins = [
        (elevation - sensor.min_elevation_deg) / 90,
        (sensor.max_off_nadir_deg - off_nadir) / 90,
    ]
    if sensor.min_sun_elevation_deg > -90:
        margins.append((sun_elevation - sensor.min_sun_elevation_deg) / 180)
    if sensor.max_range_m is not None:
        margins.append((sensor.max_range_m - distance) / sensor.max_range_m)
    available = clock >= target.available_from_sec and (
        target.available_until_sec is None or clock < target.available_until_sec
    )
    return {
        "time_sec": float(clock),
        "target_id": target.id,
        "range_m": distance,
        "off_nadir_deg": off_nadir,
        "elevation_deg": elevation,
        "sun_elevation_deg": sun_elevation,
        "margin": min(margins),
        "available": available,
        "feasible_geometry": available and min(margins) > 0,
    }


def acquisition_windows(
    provider, target, sensor, *, start_sec=None, end_sec=None, settings=None
):
    target, sensor = (
        AcquisitionTarget.model_validate(target),
        AcquisitionSensor.model_validate(sensor),
    )
    settings = SearchSettings.model_validate({} if settings is None else settings)
    begin, end = coverage(provider, start_sec, end_sec)
    left, right = (
        max(begin, target.available_from_sec),
        min(end, target.available_until_sec or end),
    )
    result = {
        "schema": "space-env.acquisition-forecast/1",
        "trajectory_sha256": provider.sha256,
        "target": target.model_dump(mode="json"),
        "sensor": sensor.model_dump(mode="json"),
        "coverage_sec": [begin, end],
        "search_coverage_sec": None,
        "search_details": None,
        "windows": [],
        "opportunities": [],
        "model": "WGS84-point/ideal-geocentric-nadir-cone/1",
        "limits": [
            "geometry only; no acquisition/product, slew/ADCS, power/memory or camera integration",
            "point target, ideal instantaneous steering cone; no swath/terrain/cloud/weather/refraction",
            "sampled event brackets; minimum dwell does not prove continuous sensor tracking",
        ],
    }
    if right <= left:
        return result
    search = positive_windows(
        lambda t: acquisition_geometry(provider, target, sensor, t)["margin"],
        left,
        right,
        settings=settings,
        boundaries=provider_boundaries(provider),
    )
    result.update(
        {
            "search_coverage_sec": [left, right],
            "windows": search["windows"],
            "search_details": search,
        }
    )
    for row in search["windows"]:
        usable_start = row["start_sec"] + (
            0.0 if row["clipped_start"] else settings.root_tolerance_sec
        )
        usable_end = row["end_sec"] - (
            0.0 if row["clipped_end"] else settings.root_tolerance_sec
        )
        usable = {
            "usable_start_sec": usable_start,
            "usable_end_sec": usable_end,
            "usable_duration_sec": max(0.0, usable_end - usable_start),
            "root_time_guard_sec": settings.root_tolerance_sec,
        }
        if usable["usable_duration_sec"] >= sensor.minimum_dwell_sec:
            identity = {
                "trajectory": provider.sha256,
                "target": target.model_dump(mode="json"),
                "sensor": sensor.model_dump(mode="json"),
                "window": row,
                "usable_window": usable,
            }
            result["opportunities"].append(
                {**row, **usable, "id": digest(identity), "target_id": target.id}
            )
    return result
