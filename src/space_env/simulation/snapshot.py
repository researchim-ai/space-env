"""Portable JSON world checkpoints; schema 1 remains a legacy path-based reader."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from space_env.simulation.world import MissionWorld


def encode(value):
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, set):
        return {"__set__": sorted((encode(item) for item in value), key=canonical_json)}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "__dtype__": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
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


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write(path: str | Path, content: str) -> None:
    """Replace only after serialization/write/fsync succeeds, on the same filesystem."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def journal(engine) -> list[dict]:
    return [
        {"time_sec": clock, "state": state.model_dump(mode="json"), "events": [
            {"kind": event.kind.value, "time_jd": None if event.time is None else event.time.jd,
             "time_sec": event.time_sec, "payload": encode(event.payload)} for event in events
        ]} for clock, state, events in engine.log
    ]


def world_payload(world: MissionWorld) -> dict:
    """Capture effective configuration, state and journal without catalog references."""
    definitions = {}
    for name, engine in world.engines.items():
        definitions[name] = {
            "physics_profile": engine.physics_profile,
            "epoch_jd": engine._epoch.jd,
            "frame": engine.orbit.frame,
            "spacecraft": engine.spacecraft.config.model_dump(mode="json"),
            "scenario": {**engine.scenario.model_dump(mode="json"),
                         "ground_stations": [station.model_dump(mode="json") for station in engine.stations]},
            "constraints": engine.constraints.model_dump(mode="json"),
            "mission_goal": engine.mission_goal.model_dump(mode="json"),
            "seed": engine.seed,
        }
    body = {
        "schema": 2, "epoch": world.epoch, "horizon_sec": world.horizon_sec,
        "definitions": definitions, "engine_order": list(world.engines), "snapshot": encode(world.snapshot()),
        "journals": {name: journal(engine) for name, engine in world.engines.items()},
    }
    return {**body, "sha256": digest(body)}


def world_from_payload(payload: dict) -> MissionWorld:
    """Reconstruct without YAML, catalog lookups or pass prediction."""
    from space_env.core.orbit import Orbit
    from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
    from space_env.core.time import Time
    from space_env.mission.constraints import Constraints
    from space_env.mission.events import MissionEvent
    from space_env.mission.goals import MissionGoal
    from space_env.mission.scenarios import ScenarioConfig
    from space_env.simulation.engine import SimulationEngine
    from space_env.simulation.state import SimulationState

    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("schema") != 2 or payload.get("sha256") != digest(body):
        raise ValueError("unsupported or corrupt portable world snapshot")
    state = decode(payload["snapshot"])
    if set(payload["definitions"]) != set(state["engines"]) or not payload["definitions"]:
        raise ValueError("snapshot engine definitions do not match state")
    order = payload.get("engine_order", list(payload["definitions"]))
    if len(order) != len(set(order)) or set(order) != set(payload["definitions"]):
        raise ValueError("snapshot actor order does not match engine definitions")
    engines = {}
    for name in order:
        definition = payload["definitions"][name]
        from space_env.core.frames import LEGACY_PHYSICS_PROFILE

        if definition.get("physics_profile", LEGACY_PHYSICS_PROFILE) != LEGACY_PHYSICS_PROFILE:
            raise ValueError("portable mission snapshot requires legacy-training/1; reference orbits are separate")
        craft = Spacecraft(config=SpacecraftConfig.model_validate(definition["spacecraft"]))
        craft.reset_state()
        scenario = ScenarioConfig.model_validate(definition["scenario"])
        # Empty stations avoid recomputing catalog passes during construction. All
        # effective stations/pass caches and initial inject side effects are restored.
        construction = scenario.model_copy(update={"ground_stations": []})
        engine = SimulationEngine(
            Orbit.from_vectors(state["engines"][name]["r"], state["engines"][name]["v"],
                               Time(definition["epoch_jd"]), frame=definition["frame"]),
            craft, mission_goal=MissionGoal.model_validate(definition["mission_goal"]),
            constraints=Constraints.model_validate(definition["constraints"]),
            scenario=construction, seed=definition["seed"],
        )
        engine.scenario = scenario
        engines[name] = engine
    world = MissionWorld(engines, epoch=payload["epoch"], horizon_sec=payload["horizon_sec"])
    world.restore(state)
    for name, rows in payload["journals"].items():
        engines[name].log[:] = [
            (row["time_sec"], SimulationState.model_validate(row["state"]), [
                MissionEvent(kind=event["kind"], time=None if event["time_jd"] is None else Time(event["time_jd"]),
                             time_sec=event["time_sec"], payload=decode(event["payload"]))
                for event in row["events"]
            ]) for row in rows
        ]
    return world


def save_world(path: str | Path, world: MissionWorld, scenario: str | Path | None = None) -> None:
    """Write schema 2. The former scenario argument is accepted but never read."""
    del scenario
    atomic_write(path, canonical_json(world_payload(world)))


def load_world(path: str | Path) -> MissionWorld:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") == 1:
        from space_env.mission.scenario_v2 import world_from_scenario

        world = world_from_scenario(payload["scenario"], attach_stations=False)
        world.restore(decode(payload["snapshot"]))
        return world
    return world_from_payload(payload)
