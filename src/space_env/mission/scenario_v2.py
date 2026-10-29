"""Сценарий второй версии: несколько аппаратов одной модели, без нового класса среды."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from space_env.catalog.build import make_engine
from space_env.catalog.references import reference
from space_env.core.ground import station_by_name
from space_env.core.time import Time
from space_env.simulation.world import MissionWorld

_FIELDS = {"version", "epoch", "horizon_sec", "instances", "stations", "requests", "outages"}
_INSTANCE_FIELDS = {"id", "model", "nu_deg"}
_REQUEST_FIELDS = {
    "id",
    "lat_deg",
    "lon_deg",
    "release_sec",
    "deadline_sec",
    "size_bits",
    "priority",
    "payload",
    "destination",
    "quality",
    "geometry",
    "length_km",
    "vertices",
    "looks",
    "gap_sec",
    "min_stereo_deg",
    "max_stereo_deg",
    "depends_on",
}


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
        try:
            reference(model)
        except KeyError as exc:
            raise ValueError(f"неизвестная модель: {model}") from exc
        seen.add(ident)
        anomaly = float(item.get("nu_deg", 0.0))
        if anomaly != anomaly or anomaly in {float("inf"), float("-inf")}:
            raise ValueError("аномалия аппарата должна быть конечной")
        parsed.append({"id": ident, "model": model, "nu_deg": anomaly})
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
        quality = str(item.get("quality", "full"))
        if quality not in {"full", "partial"}:
            raise ValueError("качество заявки должно быть full или partial")
        geometry = str(item.get("geometry", "point"))
        if geometry not in {"point", "strip", "polygon", "revisit", "stereo"}:
            raise ValueError("геометрия заявки должна быть point, strip, polygon, revisit или stereo")
        length_km = float(item.get("length_km", 0.0))
        if length_km != length_km or length_km < 0.0:
            raise ValueError("длина полосы должна быть конечной и неотрицательной")
        if geometry == "strip" and length_km <= 0.0:
            raise ValueError("у полосы нужна положительная длина")
        vertices = item.get("vertices", [])
        if geometry == "polygon" and (not isinstance(vertices, list) or len(vertices) < 3):
            raise ValueError("у полигона нужны хотя бы три вершины")
        looks = int(item.get("looks", 1))
        gap_sec = float(item.get("gap_sec", 0.0))
        if geometry == "revisit" and (looks < 2 or gap_sec <= 0.0):
            raise ValueError("повтор требует не меньше двух взглядов и положительный промежуток")
        min_stereo = float(item.get("min_stereo_deg", 0.0))
        max_stereo = float(item.get("max_stereo_deg", 0.0))
        if geometry == "stereo" and not 0.0 <= min_stereo < max_stereo:
            raise ValueError("стереопара требует min_stereo_deg < max_stereo_deg")
        depends_on = str(item.get("depends_on", "")).strip()
        destination = str(item.get("destination", "")).strip()
        if not ident:
            raise ValueError("у заявки нужен id")
        if ident in seen_requests:
            raise ValueError(f"идентификатор заявки повторяется: {ident}")
        seen_requests.add(ident)
        lat = float(item["lat_deg"])
        lon = float(item["lon_deg"])
        release = float(item.get("release_sec", 0.0))
        deadline = float(item["deadline_sec"])
        size = float(item["size_bits"])
        numbers = (lat, lon, release, deadline, size)
        if any(number != number or number in {float("inf"), float("-inf")} for number in numbers):
            raise ValueError("числа заявки должны быть конечными")
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ValueError("координаты заявки вне диапазона")
        if release < 0.0 or size <= 0.0 or deadline <= release:
            raise ValueError("у заявки срок должен быть позже выпуска, а объём положительным")
        if destination and destination not in {str(name) for name in stations}:
            raise ValueError(f"станция заявки не из списка сценария: {destination}")
        requests.append(
            {
                "id": ident,
                "lat_deg": lat,
                "lon_deg": lon,
                "release_sec": release,
                "deadline_sec": deadline,
                "size_bits": size,
                "priority": int(item.get("priority", 0)),
                "payload": payload,
                "destination": destination,
                "quality": quality,
                "geometry": geometry,
                "length_km": length_km,
                "vertices": vertices,
                "looks": looks,
                "gap_sec": gap_sec,
                "min_stereo_deg": min_stereo,
                "max_stereo_deg": max_stereo,
                "depends_on": depends_on,
            }
        )
    outages = []
    for item in loaded.get("outages", []):
        if not isinstance(item, dict):
            raise ValueError("отказ станции должен быть словарём")
        station = str(item.get("station", "")).strip()
        start = float(item.get("start_sec", 0.0))
        end = float(item.get("end_sec", 0.0))
        if not station or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("у отказа станции нужны имя и конечный положительный интервал")
        if station not in {str(name) for name in stations}:
            raise ValueError(f"станция отказа не из списка сценария: {station}")
        outages.append({"station": station, "start_sec": start, "end_sec": end})
    horizon = float(loaded.get("horizon_sec", 86400.0))
    if horizon != horizon or horizon in {float("inf"), float("-inf")} or horizon <= 0.0:
        raise ValueError("горизонт сценария должен быть положительным")
    epoch = str(loaded.get("epoch", "2026-10-20T00:00:00Z"))
    try:
        Time(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError("эпоха сценария не разбирается") from exc
    return {
        "version": 2,
        "epoch": epoch,
        "horizon_sec": horizon,
        "instances": parsed,
        "stations": [str(item) for item in stations],
        "requests": requests,
        "outages": outages,
    }


def semantic_hash(spec: dict[str, Any]) -> str:
    """Отпечаток разобранного сценария. Комментарии и порядок ключей в файле его не меняют."""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scenario_manifest(path: str | Path) -> dict[str, Any]:
    """Отпечаток файла: хеш байтов и состав, без прогона."""
    raw = Path(path).read_bytes()
    spec = load_scenario_v2(path)
    return {
        "version": spec["version"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": semantic_hash(spec),
        "epoch": spec["epoch"],
        "horizon_sec": spec["horizon_sec"],
        "instances": [item["id"] for item in spec["instances"]],
        "models": [item["model"] for item in spec["instances"]],
        "stations": list(spec["stations"]),
        "requests": [item["id"] for item in spec["requests"]],
    }


def world_from_scenario(path: str | Path, *, attach_stations: bool = False, refresh_passes: bool = True) -> MissionWorld:
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
            if refresh_passes:
                engine._refresh_passes()
            engine._refresh_environment()
        engines[item["id"]] = engine
    return MissionWorld(engines, epoch=spec["epoch"], horizon_sec=spec["horizon_sec"])
