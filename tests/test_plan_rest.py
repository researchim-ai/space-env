"""Узкое окно, полный дуплекс и отчёт по семенам со сбоями в знаменателе."""
import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.eval.statistics import assign_split, run_seeded
from space_env.mission.tasking import TaskRequest, _better, _relay
from space_env.mission.tools import choose_request
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

EPOCH = Time("2024-01-01T00:00:00Z")


def test_rolling_prefers_the_tighter_window_and_not_the_higher_priority():
    wide = {"id": "wide", "released": True, "release_sec": 0.0, "deadline_sec": 100.0, "priority": 1}
    tight = {"id": "tight", "released": True, "release_sec": 80.0, "deadline_sec": 110.0, "priority": 1}
    assert choose_request([wide, tight], "rolling")["id"] == "tight"
    assert choose_request([wide, tight], "deadline")["id"] == "wide"
    assert choose_request([wide, tight], "fifo")["id"] == "wide"
    soon = {"id": "soon", "released": True, "release_sec": 0.0, "deadline_sec": 20.0, "priority": 1}
    hot = {"id": "hot", "released": True, "release_sec": 0.0, "deadline_sec": 50.0, "priority": 9}
    assert choose_request([soon, hot], "rolling", now_sec=0.0, horizon_sec=60.0)["id"] == "soon"
    assert choose_request([soon, hot], "priority")["id"] == "hot"
    left = TaskRequest("wide", 0.0, 0.0, 0.0, 100.0, 1.0, 1, "optical")
    right = TaskRequest("tight", 0.0, 0.0, 80.0, 110.0, 1.0, 1, "optical")
    assert _better(right, left, "rolling") is True
    assert _better(left, right, "deadline") is True


def _contact(name: str, lon_deg: float, *, duplex: bool) -> SimulationEngine:
    position = ecef_to_eci(geodetic_to_ecef(0.0, lon_deg, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    direction = np.cross(position, np.array([0.0, 0.0, 1.0]))
    direction = direction / np.linalg.norm(direction)
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0, terminals=2)
    craft = Spacecraft(config=SpacecraftConfig(name=name))
    craft.reset_state()
    craft.config.comms.full_duplex = duplex
    engine = SimulationEngine(
        Orbit.from_vectors(position, direction * speed, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[station]),
    )
    engine.spacecraft.config.name = name
    return engine


def test_full_duplex_relays_while_the_radio_can_still_reach_the_station():
    request = TaskRequest("job", 0.0, 0.0, 0.0, 100.0, 1.0e6, 1, "optical")

    def world(duplex: bool) -> MissionWorld:
        source = _contact("SC-1", 0.0, duplex=duplex)
        neighbor = _contact("SC-2", 0.4, duplex=duplex)
        source.accept_bits(1.0e6, origin="collect", request_id="job")
        return MissionWorld(
            {"SC-1": source, "SC-2": neighbor},
            epoch="2024-01-01T00:00:00Z",
            horizon_sec=30.0,
        )

    half = world(False)
    assert _relay(half, [request], 0.0, 1.0, []) == []
    assert half.engines["SC-2"].stored_bits == pytest.approx(0.0)
    both = world(True)
    moved = _relay(both, [request], 0.0, 1.0, [])
    assert moved
    assert moved[0]["source"] == "SC-1"
    assert moved[0]["destination"] == "SC-2"
    assert both.engines["SC-2"].stored_bits > 0.0


def test_seed_report_keeps_the_failed_seed_in_the_denominator():
    assert assign_split("task-a", "v1") == assign_split("task-a", "v2")
    splits = {assign_split(f"task-{index}", "v") for index in range(30)}
    assert splits == {"train", "holdout"}

    def trial(seed: int) -> bool:
        if seed == 1:
            raise RuntimeError("сбой")
        return seed != 3

    report = run_seeded(trial, [0, 1, 3])
    assert report["seeds"] == [0, 1, 3]
    assert report["successes"] == 1
    assert report["failures"] == 2
    assert report["success_rate"] == pytest.approx(1.0 / 3.0)
    assert report["rows"][1]["error"] == "RuntimeError"
    assert report["rows"][1]["success"] is False
    assert report["wall_sec"] >= 0.0
