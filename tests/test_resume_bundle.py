"""Возобновление снимка, минутная лестница и строка обучения только из сходящегося повтора."""
import json
from pathlib import Path

import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.data_generators.mission_dataset import seal_run
from space_env.data_generators.verified_dataset import sft_example
from space_env.eval.load_report import simulate_load
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

_FAMILY = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "family_nominal.yaml"


def _overhead(name: str) -> SimulationEngine:
    epoch = Time("2024-03-01T00:00:00Z")
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0, terminals=1)
    position = ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), epoch)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    direction = np.cross(position, np.array([0.0, 0.0, 1.0]))
    direction = direction / np.linalg.norm(direction)
    craft = Spacecraft(config=SpacecraftConfig(name=name))
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(position, direction * speed, epoch),
        craft,
        scenario=ScenarioConfig(ground_stations=[station]),
    )
    engine.accept_bits(2.0e6, origin="imaging", request_id=name)
    return engine


def _world() -> MissionWorld:
    return MissionWorld({"a": _overhead("a")}, epoch="2024-03-01T00:00:00Z", horizon_sec=30.0)


def test_restore_then_continue_matches_the_uninterrupted_delivery():
    left = _world()
    right = _world()
    left.step(5.0, {"a": "comms"})
    right.step(5.0, {"a": "comms"})
    saved = right.snapshot()
    right.step(5.0, {"a": "wait"})
    right.restore(saved)
    left.step(5.0, {"a": "comms"})
    right.step(5.0, {"a": "comms"})
    assert right.now_sec() == left.now_sec()
    assert right.engines["a"].delivered_bits == left.engines["a"].delivered_bits
    assert right.engines["a"].delivered_bits > 0.0


def test_sixteen_vehicles_propagate_two_minutes():
    report = simulate_load(16, 100, duration_sec=120.0, step_sec=30.0)
    assert report["vehicles"] == 16
    assert report["requests"] == 100
    assert report["served_requests"] == 0
    assert report["outcome"] == "propagation_only"
    assert report["steps"] == 4
    assert report["simulated_sec"] == 120.0
    assert 0.0 < report["wall_sec"] < 60.0
    assert report["ladder"] == "multi_step_propagation"


def test_a_failed_replay_does_not_become_a_training_row(tmp_path: Path):
    record = tmp_path / "broken.json"
    record.write_text(
        json.dumps({"schema": 2, "scenario": str(_FAMILY), "sha256": "not-the-file", "schedule": [], "on_time": []}),
        encoding="utf-8",
    )
    assert sft_example(record) is None


def test_a_matching_kiruna_seal_becomes_a_training_row(tmp_path: Path):
    path = tmp_path / "nominal.json"
    sealed = seal_run(path, _FAMILY, step_sec=300.0)
    row = sft_example(path)
    assert row is not None
    assert row["on_time"] == sealed["on_time"]
    assert row["on_time"] == ["kiruna-pass"]
    assert any(item["kind"] == "imaging" or item["kind"] == "comms" for item in row["commands"])
