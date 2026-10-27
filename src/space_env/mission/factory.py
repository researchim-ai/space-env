"""Короткая и суточная миссии. Заявка ставится на след аппарата во время открытой станции."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from space_env.catalog.build import make_engine
from space_env.core.access import subsatellite_latlon, target_access
from space_env.core.ground import station_by_name

SHORT_VEHICLES = 2
SHORT_REQUESTS = 12
DAY_REQUESTS = 32
STATIONS = ("Kiruna", "Svalbard", "Matera")
_EPOCH = "2026-10-20T10:00:00Z"


def _fleet(instances: list[dict]):
    stations = [station_by_name(name) for name in STATIONS]
    engines = []
    for item in instances:
        engine = make_engine(item["model"], epoch=_EPOCH, nu_deg=float(item["nu_deg"]), with_stations=False)
        engine.spacecraft.config.name = item["id"]
        engine.stations = list(stations)
        engine.scenario.ground_stations = list(stations)
        engine._refresh_environment()
        engines.append(engine)
    return engines


def _opportunities(instances: list[dict], horizon: float, step: float) -> list[dict]:
    engines = _fleet(instances)
    found: list[dict] = []
    elapsed = 0.0
    while elapsed < horizon - 1.0:
        for engine in engines:
            snap = engine.get_state_snapshot()
            radar = bool(engine.spacecraft.config.payload.images_in_eclipse)
            if not snap.in_contact or not snap.contact_station:
                continue
            lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
            access = target_access(
                engine.orbit.r,
                lat,
                lon,
                engine.now(),
                half_fov_deg=engine.spacecraft.config.payload.half_fov_deg,
                frame=engine.orbit.frame,
                images_in_eclipse=radar,
                satellite_in_umbra=snap.in_eclipse,
            )
            if not access.visible:
                continue
            found.append(
                {
                    "vehicle": engine.spacecraft.config.name,
                    "lat_deg": round(lat, 4),
                    "lon_deg": round(lon, 4),
                    "release_sec": float(elapsed),
                    "payload": "sar" if radar else "optical",
                    "destination": snap.contact_station,
                }
            )
        elapsed += step
        if elapsed >= horizon:
            break
        for engine in engines:
            engine.step(step, "wait")
    return found


def fleet_strip(instances: list[dict], *, horizon: float = 12.0 * 3600.0, step: float = 90.0) -> list[dict]:
    """По одной заявке на аппарат, кроме носителей учебной Кируны и радара Шпицбергена."""
    spare = [item for item in instances if item["id"] not in {"sentinel-a", "radar"}]
    found = _opportunities(spare, horizon, step)
    chosen: list[dict] = []
    seen: set[str] = set()
    for item in found:
        vehicle = str(item["vehicle"])
        if vehicle in seen:
            continue
        seen.add(vehicle)
        access = float(item["release_sec"])
        release = max(0.0, access - 300.0)
        deadline = access + 1800.0
        if deadline <= release:
            continue
        chosen.append(
            {
                "id": f"fleet-{len(chosen):02d}",
                "lat_deg": item["lat_deg"],
                "lon_deg": item["lon_deg"],
                "release_sec": release,
                "deadline_sec": deadline,
                "size_bits": 8.0e6,
                "priority": 2,
                "payload": item["payload"],
                "destination": item["destination"],
                "quality": "full",
            }
        )
    if len(chosen) < 8:
        raise RuntimeError(f"для шестнадцати аппаратов найдено только {len(chosen)} отдельных проходов")
    return chosen


def build_mission(kind: str, seed: int) -> dict:
    if kind not in {"short", "day"}:
        raise ValueError("миссия бывает short или day")
    if kind == "short":
        horizon = 3600.0
        count = SHORT_REQUESTS
        step = 20.0
        instances = [
            {"id": "optic", "model": "sentinel-2a", "nu_deg": 80.0},
            {"id": "radar", "model": "sar-small", "nu_deg": 45.0},
        ]
    else:
        horizon = 86400.0
        count = DAY_REQUESTS
        step = 40.0
        instances = [
            {"id": "sentinel-a", "model": "sentinel-2a", "nu_deg": 80.0},
            {"id": "sentinel-b", "model": "sentinel-2a", "nu_deg": 180.0},
            {"id": "landsat", "model": "landsat-9", "nu_deg": 90.0},
            {"id": "radar", "model": "sar-small", "nu_deg": 45.0},
        ]
    found = _opportunities(instances, horizon, step)
    if len(found) < count:
        raise RuntimeError(f"для {kind} найдено только {len(found)} проходов, нужно {count}")
    rng = np.random.default_rng(seed)
    picked = sorted(int(index) for index in rng.choice(len(found), size=count, replace=False))
    requests = []
    for number, index in enumerate(picked):
        item = found[index]
        access = float(item["release_sec"])
        release = max(0.0, access - 300.0)
        deadline = min(horizon, access + 1800.0)
        if deadline <= release:
            deadline = min(horizon, release + 60.0)
        requests.append(
            {
                "id": f"req-{number:02d}",
                "lat_deg": item["lat_deg"],
                "lon_deg": item["lon_deg"],
                "release_sec": release,
                "deadline_sec": deadline,
                "size_bits": 8.0e6,
                "priority": int(rng.integers(1, 5)),
                "payload": item["payload"],
                "destination": item["destination"],
                "quality": "full",
            }
        )
    return {
        "version": 2,
        "epoch": _EPOCH,
        "horizon_sec": horizon,
        "instances": instances,
        "stations": list(STATIONS),
        "requests": requests,
    }


def write_mission(path: str | Path, kind: str, seed: int) -> Path:
    target = Path(path)
    target.write_text(yaml.safe_dump(build_mission(kind, seed), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target
