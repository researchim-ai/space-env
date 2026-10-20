"""Стадии куррикулума: какие среды и какой горизонт эпизода крутить по очереди."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_STAGES: list[dict[str, Any]] = [
    {"name": "resources", "env": "space_env/ResourceManager-v0", "episodes": 2, "max_episode_steps": 20},
    {"name": "anomaly", "env": "space_env/AnomalyResponse-v0", "episodes": 2, "max_episode_steps": 12},
    {"name": "conjunction", "env": "space_env/ConjunctionAssessment-v0", "episodes": 2, "max_episode_steps": 12},
    {"name": "planner", "env": "space_env/MissionPlanner-v0", "episodes": 2, "max_episode_steps": 20},
]


def _config_path() -> Path:
    name = Path("configs") / "curriculum.yaml"
    candidates = [Path.cwd() / name, Path(__file__).resolve().parents[3] / name]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_curriculum(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Список стадий. Неизвестные ключи сохраняются, обязательны name и env."""
    source = Path(path) if path is not None else _config_path()
    if not source.exists():
        return [dict(stage) for stage in DEFAULT_STAGES]
    with source.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    stages = loaded.get("stages", [])
    if not stages:
        return [dict(stage) for stage in DEFAULT_STAGES]
    normalized: list[dict[str, Any]] = []
    for stage in stages:
        item = dict(stage)
        if "name" not in item or "env" not in item:
            raise ValueError("у стадии куррикулума должны быть поля name и env")
        item.setdefault("episodes", 1)
        item.setdefault("max_episode_steps", 20)
        normalized.append(item)
    return normalized
