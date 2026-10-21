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
from space_env.core.time import Time
from space_env.simulation.fleet import Fleet


def _vec(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(3)]


def _capture(
    engines: dict,
    *,
    pairs: list | None = None,
    buffer_mb: dict[str, float] | None = None,
    delivered_mb: float = 0.0,
    route: list[dict] | None = None,
    memory_bits: dict[str, float] | None = None,
) -> dict:
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
            "buffer_mb": 0.0 if buffer_mb is None else buffer_mb.get(name, 0.0),
            "memory_bits": None if memory_bits is None else memory_bits.get(name),
        }
    pairs = [
        {
            "leader": pair.leader,
            "follower": pair.follower,
            "range_m": pair.range_m,
            "isl_margin_db": pair.isl_margin_db,
            "earth_blocked": pair.earth_blocked,
        }
        for pair in (pairs or [])
    ]
    return {
        "t": next(iter(engines.values())).time_sec,
        "sun": _vec(sun_position_m(moment)),
        "gast": gast_rad(moment),
        "stations": stations,
        "vehicles": vehicles,
        "pairs": pairs,
        "route": list(route or []),
        "delivered_mb": delivered_mb,
    }


def _document(frames: list[dict], epoch: Time, dt_sec: float, memory_capacity_bits: float | None) -> dict:
    document = {
        "version": 1,
        "epoch": epoch.datetime.isoformat(),
        "earth_radius_m": R_EARTH_EQUATOR,
        "dt_sec": dt_sec,
        "frames": frames,
    }
    if memory_capacity_bits is not None:
        document["memory_capacity_bits"] = memory_capacity_bits
    return document


def _write(document: dict, path: str | Path | None) -> dict:
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document), encoding="utf-8")
    return document


def export_trajectory(
    fleet: Fleet,
    steps: int,
    dt_sec: float,
    path: str | Path | None = None,
    *,
    route: list[dict] | None = None,
    memory_bits: dict[str, float] | None = None,
    memory_capacity_bits: float | None = None,
) -> dict:
    """Пишет кадры t=0…steps включительно. Группировка продвигается на dt_sec."""
    if steps < 0 or dt_sec <= 0.0:
        raise ValueError("steps >= 0 и dt_sec > 0")
    start = next(iter(fleet.vehicles.values())).orbit.epoch
    frames = []
    for index in range(steps + 1):
        frames.append(
            _capture(
                fleet.vehicles,
                pairs=fleet.pairs(),
                buffer_mb=fleet.buffer_mb,
                delivered_mb=fleet.delivered_mb,
                route=route,
                memory_bits=memory_bits,
            )
        )
        if index < steps:
            fleet.step(dt_sec)
    return _write(_document(frames, start, dt_sec, memory_capacity_bits), path)


def export_constellation_ops(path: str | Path | None = None, max_steps: int = 8) -> dict:
    """Эпизод цепочки: эвристика несёт пакет к аппарату над Фэрбанксом, кадры те же, что у просмотрщика."""
    import gymnasium as gym

    import space_env.envs  # регистрация среды
    from space_env.rl.expert import expert_action

    env = gym.make("space_env/ConstellationOps-v0", max_episode_steps=max_steps)
    observation, info = env.reset(seed=0)
    fleet = env.unwrapped._fleet
    start = next(iter(fleet.vehicles.values())).orbit.epoch
    frames = []
    route: list[dict] = []
    for _ in range(max_steps + 1):
        memory = {name: float(fleet.buffer_mb[name]) * 1.0e6 for name in fleet.vehicles}
        frames.append(
            _capture(
                fleet.vehicles,
                pairs=fleet.pairs(),
                buffer_mb=dict(fleet.buffer_mb),
                delivered_mb=fleet.delivered_mb,
                route=route,
                memory_bits=memory,
            )
        )
        if info.get("success"):
            break
        action = expert_action("space_env/ConstellationOps-v0", observation)
        observation, _reward, terminated, truncated, info = env.step(action)
        route = list(info.get("route") or [])
        if terminated or truncated:
            memory = {name: float(fleet.buffer_mb[name]) * 1.0e6 for name in fleet.vehicles}
            frames.append(
                _capture(
                    fleet.vehicles,
                    pairs=fleet.pairs(),
                    buffer_mb=dict(fleet.buffer_mb),
                    delivered_mb=fleet.delivered_mb,
                    route=route,
                    memory_bits=memory,
                )
            )
            break
    env.close()
    return _write(_document(frames, start, 60.0, 8.0e6), path)


def export_engine(
    engine,
    steps: int,
    dt_sec: float,
    path: str | Path | None = None,
    *,
    memory_bits: float | None = None,
    memory_capacity_bits: float | None = None,
) -> dict:
    """Один аппарат. Память в битах копируется в каждый кадр: буфер среды живёт снаружи движка."""
    if steps < 0 or dt_sec <= 0.0:
        raise ValueError("steps >= 0 и dt_sec > 0")
    name = engine.spacecraft.config.name
    start = engine.orbit.epoch
    stored = {name: memory_bits} if memory_bits is not None else None
    frames = []
    for index in range(steps + 1):
        frames.append(_capture({name: engine}, memory_bits=stored))
        if index < steps:
            engine.step(dt_sec)
    return _write(_document(frames, start, dt_sec, memory_capacity_bits), path)
