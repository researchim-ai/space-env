"""Сценарий смены оператора. Поля суток, не паспорт аппарата."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from space_env.mission.campaign import DAY_SEC, MISSION_KEYS

_FIELDS = {"version", "keys", "horizon_sec", "epoch", "nu_deg", "dv_budget_m_s", "until", "demand"}
_DEMAND_FIELDS = {"name", "place", "lat_deg", "lon_deg", "deadline_sec", "deliver_to", "size_bits"}


def _default_path() -> Path:
    cwd = Path.cwd() / "configs" / "scenarios" / "operator_shift.yaml"
    if cwd.is_file():
        return cwd
    return Path(__file__).resolve().parents[1] / "configs" / "operator_shift.yaml"


def _demand(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("demand в сценарии должен быть словарём")
    unknown = set(raw) - _DEMAND_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"в заявке неизвестные поля: {names}")
    place = str(raw.get("place", "subsatellite"))
    if place not in {"subsatellite", "coordinates"}:
        raise ValueError("place заявки должен быть subsatellite или coordinates")
    try:
        deadline = float(raw["deadline_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("у заявки должен быть положительный deadline_sec") from exc
    if not math.isfinite(deadline) or deadline <= 0.0:
        raise ValueError("срок заявки должен быть положительным")
    lat = lon = None
    if place == "coordinates":
        try:
            lat = float(raw["lat_deg"])
            lon = float(raw["lon_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("координаты заявки должны быть числами") from exc
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError("координаты заявки должны быть конечными")
    deliver = str(raw.get("deliver_to", "overhead"))
    if not deliver:
        raise ValueError("у заявки должна быть станция сдачи")
    size_bits = None
    if "size_bits" in raw:
        try:
            size_bits = float(raw["size_bits"])
        except (TypeError, ValueError) as exc:
            raise ValueError("размер заявки должен быть числом бит") from exc
        if not math.isfinite(size_bits) or size_bits <= 0.0:
            raise ValueError("размер заявки должен быть положительным")
    return {
        "name": str(raw.get("name", "request")),
        "place": place,
        "lat_deg": lat,
        "lon_deg": lon,
        "deadline_sec": deadline,
        "deliver_to": deliver,
        "size_bits": size_bits,
    }


def _checked(loaded: dict[str, Any], source: Path | None) -> dict[str, Any]:
    unknown = set(loaded) - _FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"в сценарии неизвестные поля: {names}")
    raw_keys = loaded.get("keys", MISSION_KEYS)
    if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
        raise ValueError("в сценарии смены нужен хотя бы один аппарат")
    keys = tuple(str(item) for item in raw_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("аппарат в сценарии повторяется")
    until = str(loaded.get("until", "full"))
    if until not in {"award", "full"}:
        raise ValueError("until в сценарии должен быть award или full")
    try:
        horizon = float(loaded.get("horizon_sec", DAY_SEC))
        nu_deg = float(loaded.get("nu_deg", 180.0))
        budget = float(loaded.get("dv_budget_m_s", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("горизонт, аномалия и бюджет должны быть числами") from exc
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("горизонт сценария должен быть положительным")
    if not math.isfinite(nu_deg) or not math.isfinite(budget) or budget < 0.0:
        raise ValueError("аномалия и бюджет должны быть конечными, бюджет — неотрицательным")
    return {
        "keys": keys,
        "horizon_sec": horizon,
        "epoch": str(loaded.get("epoch", "2026-10-20T00:00:00Z")),
        "nu_deg": nu_deg,
        "dv_budget_m_s": budget,
        "until": until,
        "demand": _demand(loaded.get("demand")),
        "source": "" if source is None else str(source),
    }


def load_mission_scenario(path: str | Path) -> dict[str, Any]:
    """Читает файл миссии. Неизвестное поле, повтор аппарата и плохой бюджет отвергаются."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"сценарий не найден: {source}")
    with source.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"сценарий {source} должен быть словарём")
    return _checked(loaded, source)


def load_shift_scenario(path: str | Path | None = None) -> dict[str, Any]:
    """Читает сутки смены. Нет файла — возвращает встроенные значения по умолчанию."""
    source = Path(path) if path is not None else _default_path()
    if not source.is_file():
        return _checked({}, None)
    return load_mission_scenario(source)
