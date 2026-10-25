"""Сценарий второй версии: несколько аппаратов одной модели, без нового класса среды."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from space_env.catalog.build import make_engine
from space_env.core.ground import station_by_name
from space_env.simulation.world import MissionWorld

_FIELDS = {"version", "epoch", "horizon_sec", "instances", "stations", "requests"}
_INSTANCE_FIELDS = {"id", "model", "nu_deg"}
_REQUEST_FIELDS = {"id", "lat_deg", "lon_deg", "release_sec", "deadline_sec", "size_bits", "priority", "payload"}


def load_scenario_v2(path: str | Path) -> dict[str, Any]:
    """Читает YAML. Повтор идентификатора запрещён, одна модель может стоять на нескольких аппаратах."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("сценарий v2 должен быть словарём")
    unknown = set(loaded) - _FIELDS
    if unknown:
        raise ValueError(f"в сценарии неизвестные поля: {', '.join(sorted(unknown))}")
    if int(loaded.get("version", 0)) != 2:
        raise ValueError("сценарий v2 имеет version: 2")
    instances = loaded.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("в сценарии нужен список аппаратов")
    seen: set[str] = set()
    parsed = []
    for item in instances:
        if not isinstance(item, dict):
            raise ValueError("аппарат сценария должен быть словарём")
        extra = set(item) - _INSTANCE_FIELDS
        if extra:
            raise ValueError(f"у аппарата неизвестные поля: {', '.join(sorted(extra))}")
        ident = str(item.get("id", "")).strip()
        model = str(item.get("model", "")).strip()
        if not ident or not model:
            raise ValueError("у аппарата нужны id и model")
        if ident in seen:
            raise ValueError(f"идентификатор аппарата повторяется: {ident}")
        seen.add(ident)
        parsed.append({"id": ident, "model": model, "nu_deg": float(item.get("nu_deg", 0.0))})
    stations = loaded.get("stations", [])
    if not isinstance(stations, list) or any(not str(item).strip() for item in stations):
        raise ValueError("станции сценария должны быть списком имён")
    requests = []
    seen_requests: set[str] = set()
    for item in loaded.get("requests", []):
        if not isinstance(item, dict):
            raise ValueError("заявка сценария должна быть словарём")
        extra = set(item) - _REQUEST_FIELDS
        if extra:
            raise ValueError(f"у заявки неизвестные поля: {', '.join(sorted(extra))}")
        ident = str(item.get("id", "")).strip()
        payload = str(item.get("payload", "optical"))
        if payload not in {"optical", "sar"}:
            raise ValueError("payload заявки должен быть optical или sar")
        if not ident:
            raise ValueError("у заявки нужен id")
        if ident in seen_requests:
            raise ValueError(f"идентификатор заявки повторяется: {ident}")
        seen_requests.add(ident)
        requests.append(
            {
                "id": ident,
                "lat_deg": float(item["lat_deg"]),
                "lon_deg": float(item["lon_deg"]),
                "release_sec": float(item.get("release_sec", 0.0)),
                "deadline_sec": float(item["deadline_sec"]),
                "size_bits": float(item["size_bits"]),
                "priority": int(item.get("priority", 0)),
                "payload": payload,
            }
        )
    return {
        "version": 2,
        "epoch": str(loaded.get("epoch", "2026-10-20T00:00:00Z")),
        "horizon_sec": float(loaded.get("horizon_sec", 86400.0)),
        "instances": parsed,
        "stations": [str(item) for item in stations],
        "requests": requests,
    }


def scenario_manifest(path: str | Path) -> dict[str, Any]:
    """Отпечаток файла: хеш байтов и состав, без прогона."""
    raw = Path(path).read_bytes()
    spec = load_scenario_v2(path)
    return {
        "version": spec["version"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "epoch": spec["epoch"],
        "horizon_sec": spec["horizon_sec"],
        "instances": [item["id"] for item in spec["instances"]],
        "models": [item["model"] for item in spec["instances"]],
        "stations": list(spec["stations"]),
        "requests": [item["id"] for item in spec["requests"]],
    }


def world_from_scenario(path: str | Path, *, attach_stations: bool = False) -> MissionWorld:
    """Собирает мир. Имя аппарата — id экземпляра, модель можно повторять.

    attach_stations вешает станции файла и считает их окна на 6 ч.
    """
    spec = load_scenario_v2(path)
    try:
        stations = [station_by_name(name) for name in spec["stations"]] if attach_stations else []
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    engines = {}
    for item in spec["instances"]:
        engine = make_engine(
            item["model"],
            epoch=spec["epoch"],
            nu_deg=item["nu_deg"],
            with_stations=False,
        )
        engine.spacecraft.config.name = item["id"]
        if stations:
            engine.stations = list(stations)
            engine.scenario.ground_stations = list(stations)
            engine._refresh_passes()
            engine._refresh_environment()
        engines[item["id"]] = engine
    return MissionWorld(engines, epoch=spec["epoch"], horizon_sec=spec["horizon_sec"])
