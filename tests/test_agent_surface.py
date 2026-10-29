"""Словарная среда, совместный шаг и короткий замер нагрузки."""
import numpy as np
import gymnasium as gym

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.envs.dict_env import DICT_KEYS
from space_env.eval.load_report import load_report
from space_env.mission.scenarios import ScenarioConfig
from space_env.rl.expert import expert_action
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.joint import JointExecutor
from space_env.simulation.world import MissionWorld

import space_env.envs  # noqa: F401


def test_mission_dict_observation_is_a_dictionary_and_a_step_moves_time():
    env = gym.make("space_env/MissionDict-v0", max_episode_steps=2)
    try:
        obs, info = env.reset(seed=0)
        assert tuple(obs) == DICT_KEYS
        assert "success" in info
        _obs, _reward, _terminated, truncated, _info = env.step(0)
        assert truncated is False
        assert float(obs["clock"][0]) == 0.0
        assert float(_obs["clock"][0]) > 0.0
    finally:
        env.close()
    assert expert_action("space_env/MissionDict-v0", {}) == 0


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


def test_joint_step_matches_the_world_step():
    left = MissionWorld(
        {"a": _overhead("a"), "b": _overhead("b")},
        epoch="2024-03-01T00:00:00Z",
        horizon_sec=30.0,
    )
    right = MissionWorld(
        {"a": _overhead("a"), "b": _overhead("b")},
        epoch="2024-03-01T00:00:00Z",
        horizon_sec=30.0,
    )
    orders = {"a": "comms", "b": "comms"}
    joint = JointExecutor(left, dt_sec=5.0).step(orders)
    right.step(5.0, orders)
    direct = {name: float(engine.delivered_bits) for name, engine in right.engines.items()}
    assert joint == direct
    assert joint["a"] > 0.0
    assert joint["b"] > 0.0
    assert left.now_sec() == right.now_sec()


def test_short_load_records_the_fleet_it_actually_stepped():
    report = load_report(4, 8, step_sec=1.0)
    assert report["vehicles"] == 4
    assert report["requests"] == 8
    assert report["time_sec"] == 1.0
    assert report["wall_sec"] > 0.0
    assert report["ladder"] == "recorded_short_step"
