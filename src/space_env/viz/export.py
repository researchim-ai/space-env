"""
Запись эпизода в JSON, который читает визуализация viz/.
Тот же формат можно снимать из цикла обучения: положение, скорость, Солнце, тень, запас линии.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from space_env.core.constants import R_EARTH_EQUATOR
from space_env.core.frames import ecef_to_eci, gast_rad, geodetic_to_ecef
from space_env.core.sun import sun_position_m
from space_env.simulation.fleet import Fleet


def _vec(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(3)]


def _capture(fleet: Fleet) -> dict:
    engines = fleet.vehicles
    moment = next(iter(engines.values())).orbit.epoch
    stations = []
    seen: set[str] = set()
    for engine in engines.values():
        for station in engine.stations:
            if station.name in seen:
                continue
            seen.add(station.name)
            eci = ecef_to_eci(geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m), moment)
            stations.append({"name": station.name, "r": _vec(eci)})
    vehicles = {}
    for name, engine in engines.items():
        snap = engine.get_state_snapshot()
        vehicles[name] = {
            "r": _vec(engine.orbit.r),
            "v": _vec(engine.orbit.v),
            "soc": snap.battery_soc,
            "eclipse": snap.in_eclipse,
            "altitude_km": snap.altitude_km,
            "contact": snap.in_contact,
            "margin_db": snap.link_margin_db,
            "station": snap.contact_station,
        }
    return {
        "t": fleet.time_sec,
        "sun": _vec(sun_position_m(moment)),
        "gast": gast_rad(moment),
        "stations": stations,
        "vehicles": vehicles,
    }


def export_trajectory(
    fleet: Fleet,
    steps: int,
    dt_sec: float,
    path: str | Path | None = None,
) -> dict:
    """Пишет кадры t=0…steps включительно. Группировка продвигается на dt_sec."""
    if steps < 0 or dt_sec <= 0.0:
        raise ValueError("steps >= 0 и dt_sec > 0")
    start = next(iter(fleet.vehicles.values())).orbit.epoch
    frames = []
    for index in range(steps + 1):
        frames.append(_capture(fleet))
        if index < steps:
            fleet.step(dt_sec)
    document = {
        "version": 1,
        "epoch": start.datetime.isoformat(),
        "earth_radius_m": R_EARTH_EQUATOR,
        "dt_sec": dt_sec,
        "frames": frames,
    }
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document), encoding="utf-8")
    return document
