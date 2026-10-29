"""Параллельный флот, бюджет вызовов, реестр и первая ступень нагрузки."""
from pathlib import Path

import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.eval.ci_profiles import PROFILES
from space_env.eval.load_report import ladder_report
from space_env.mission.registry import scenario_registry
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.tools import MissionTools
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.parallel import ParallelFleet
from space_env.simulation.world import MissionWorld

_NOMINAL = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "family_nominal.yaml"
_ROOT = Path(__file__).resolve().parents[1]


def _overhead(name: str) -> SimulationEngine:
    epoch = Time("2024-03-01T00:00:00Z")
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0, terminals=2)
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
    return MissionWorld(
        {"a": _overhead("a"), "b": _overhead("b")},
        epoch="2024-03-01T00:00:00Z",
        horizon_sec=30.0,
    )


def test_each_agent_sees_only_itself_and_the_joint_step_matches_the_world():
    fleet = ParallelFleet(_world(), dt_sec=5.0)
    observations, _infos = fleet.reset()
    assert set(observations["a"]) == {"soc", "contact"}
    assert "b" not in observations["a"]
    other = _world()
    _obs, rewards, _term, _trunc, infos = fleet.step({"a": 2, "b": 2})
    other.step(5.0, {"a": "comms", "b": "comms"})
    assert rewards["a"] == pytest.approx(other.engines["a"].delivered_bits)
    assert rewards["b"] == pytest.approx(other.engines["b"].delivered_bits)
    assert rewards["a"] > 0.0
    assert rewards["b"] > 0.0
    assert infos["a"]["delivered_bits"] == pytest.approx(rewards["a"])
    assert "delivered_bits" not in _obs["a"]
    with pytest.raises(KeyError):
        fleet.step({"a": 0})


def test_call_budget_stops_the_third_tool_call():
    tools = MissionTools(_NOMINAL, call_budget=2)
    tools.observe()
    tools.forecast("sentinel-a", 30.0)
    with pytest.raises(RuntimeError, match="бюджет вызовов исчерпан"):
        tools.observe()
    assert tools.session.world.now_sec() == pytest.approx(0.0)


def test_registry_marks_the_holdout_file_and_keeps_kiruna_in_train():
    rows = {item["id"]: item for item in scenario_registry()}
    assert rows["mission_holdout"]["holdout"] is True
    assert rows["mission_holdout"]["present"] is True
    assert rows["family_nominal"]["holdout"] is False
    assert rows["family_nominal"]["present"] is True


def test_ci_profiles_point_at_files_that_exist():
    assert set(PROFILES) == {"fast", "reference", "full"}
    for files in PROFILES.values():
        assert files
        for name in files:
            assert (_ROOT / name).is_file()


def test_sixteen_vehicles_record_a_one_second_step():
    report = ladder_report(((16, 100, 1.0),))[0]
    assert report["vehicles"] == 16
    assert report["requests"] == 100
    assert len(report["request_ids"]) == 100
    assert report["served_requests"] == 0
    assert report["outcome"] == "propagation_only"
    assert report["time_sec"] == 1.0
    assert 0.0 < report["wall_sec"] < 120.0
    assert report["rss_kb"] >= 0


def test_sixty_four_and_two_hundred_fifty_six_propagate_one_second():
    rows = ladder_report(((64, 1000, 1.0), (256, 10000, 1.0)))
    assert [(row["vehicles"], row["requests"]) for row in rows] == [(64, 1000), (256, 10000)]
    for row in rows:
        assert row["served_requests"] == 0
        assert row["outcome"] == "propagation_only"
        assert row["time_sec"] == 1.0
        assert row["wall_sec"] < 60.0
