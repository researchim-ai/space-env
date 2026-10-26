"""Снимок мира в JSON и сборка того же мира в другом процессе."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from space_env.mission.scenario_v2 import world_from_scenario
from space_env.simulation.world import MissionWorld


def encode(value):
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, set):
        return {"__set__": [encode(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "__dtype__": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def decode(value):
    if isinstance(value, dict) and "__ndarray__" in value and "__dtype__" in value:
        return np.array(value["__ndarray__"], dtype=value["__dtype__"])
    if isinstance(value, dict) and set(value) == {"__set__"}:
        return {decode(item) for item in value["__set__"]}
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def save_world(path: str | Path, world: MissionWorld, scenario: str | Path) -> None:
    """Пишет сценарий и снимок. Pickle не используется."""
    payload = {
        "schema": 1,
        "scenario": str(scenario),
        "snapshot": encode(world.snapshot()),
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_world(path: str | Path) -> MissionWorld:
    """Собирает мир из файла сценария и возвращает снимок."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema", 0)) != 1:
        raise ValueError("снимок мира имеет schema: 1")
    world = world_from_scenario(payload["scenario"], attach_stations=False)
    world.restore(decode(payload["snapshot"]))
    return world
