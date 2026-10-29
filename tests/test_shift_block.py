"""Книга станции, общий модем и фазирование вдоль трека."""
import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.relative import cw_propagate, eci_to_rtn, mean_motion_rad_s
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.ground_schedule import book_passes
from space_env.mission.phasing import phase_plan
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from space_env.simulation.world import MissionWorld

EPOCH = Time("2024-01-01T00:00:00Z")


def _overhead(name: str, *, terminals: int = 1, share: bool = False) -> SimulationEngine:
    station = GroundStation(
        name="Equator",
        lat_deg=0.0,
        lon_deg=0.0,
        min_elevation_deg=5.0,
        terminals=terminals,
        share_bandwidth=share,
    )
    position = ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    axis = np.array([0.0, 0.0, 1.0])
    direction = np.cross(position, axis)
    direction = direction / np.linalg.norm(direction)
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(position, direction * speed, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[station]),
    )
    engine.spacecraft.config.name = name
    engine.accept_bits(20.0e6, origin="imaging", request_id=name)
    return engine


def test_one_terminal_keeps_the_higher_priority_and_the_book_blocks_the_step():
    first = {"vehicle": "a", "station": "Equator", "start_sec": 0.0, "end_sec": 10.0, "priority": 1}
    second = {"vehicle": "b", "station": "Equator", "start_sec": 0.0, "end_sec": 10.0, "priority": 5}
    single = book_passes([first, second], {"Equator": 1})
    assert [item["vehicle"] for item in single["accepted"]] == ["b"]
    assert single["rejected"][0]["reason"] == "терминал занят"
    both = book_passes([first, second], {"Equator": 2})
    assert {item["vehicle"] for item in both["accepted"]} == {"a", "b"}
    apart = book_passes(
        [first, {**second, "start_sec": 10.0, "end_sec": 20.0, "priority": 1}],
        {"Equator": 1},
    )
    assert len(apart["accepted"]) == 2
    empty = book_passes(
        [{"vehicle": "a", "station": "Equator", "start_sec": 5.0, "end_sec": 5.0, "priority": 1}],
        {"Equator": 1},
    )
    assert empty["rejected"][0]["reason"] == "пустое окно"

    world = MissionWorld(
        {"a": _overhead("a", terminals=2), "b": _overhead("b", terminals=2)},
        epoch="2024-01-01T00:00:00Z",
        horizon_sec=30.0,
    )
    world.use_book(book_passes([first], {"Equator": 2}))
    world.step(5.0, {"a": "comms", "b": "comms"})
    assert world.engines["a"].delivered_bits > 0.0
    assert world.engines["b"].delivered_bits == pytest.approx(0.0)
    assert world.last_rejected == [{"actor_id": "b", "kind": "comms", "reason": "вне окна"}]


def test_shared_modem_gives_each_vehicle_half_the_solo_rate():
    solo = _overhead("solo", terminals=2, share=True)
    solo_world = MissionWorld({"solo": solo}, epoch="2024-01-01T00:00:00Z", horizon_sec=30.0)
    solo_world.step(5.0, {"solo": "comms"})
    pair = MissionWorld(
        {"a": _overhead("a", terminals=2, share=True), "b": _overhead("b", terminals=2, share=True)},
        epoch="2024-01-01T00:00:00Z",
        horizon_sec=30.0,
    )
    pair.step(5.0, {"a": "comms", "b": "comms"})
    assert pair.engines["a"].delivered_bits == pytest.approx(solo.delivered_bits * 0.5, rel=1e-6)
    assert pair.engines["b"].delivered_bits == pytest.approx(solo.delivered_bits * 0.5, rel=1e-6)
    assert pair.last_rejected == []


def test_phase_burn_follows_the_secular_rate_and_stops_outside_the_linear_model():
    radius = 7.0e6
    horizon = 600.0
    error = 3_000.0
    motion = mean_motion_rad_s(radius)
    plan = phase_plan(
        error,
        radius_m=radius,
        horizon_sec=horizon,
        dv_budget_m_s=80.0,
        mission_reserve_m_s=0.0,
    )
    assert plan["accepted"] is True
    assert plan["dv_m_s"] == pytest.approx(error / (3.0 * horizon))
    end = cw_propagate(np.array([0.0, error, 0.0, 0.0, plan["dv_m_s"], 0.0]), motion, horizon)
    oscillation = 4.0 * np.sin(motion * horizon) / motion * plan["dv_m_s"]
    assert end[1] - oscillation == pytest.approx(0.0, abs=1e-4)
    far = phase_plan(100_000.0, radius_m=radius, horizon_sec=horizon, dv_budget_m_s=80.0, mission_reserve_m_s=0.0)
    assert far["reason"] == "слишком далеко для линейной модели"
    assert far["dv_m_s"] == pytest.approx(0.0)
    long = phase_plan(error, radius_m=radius, horizon_sec=5_000.0, dv_budget_m_s=80.0, mission_reserve_m_s=0.0)
    assert long["reason"] == "горизонт длиннее линейной модели"

    fleet = Fleet.string_of_pearls(2, separation_deg=0.05, scenario=ScenarioConfig(ground_stations=[]))
    twin = Fleet.string_of_pearls(2, separation_deg=0.05, scenario=ScenarioConfig(ground_stations=[]))
    along = next(item.along_m for item in fleet.pairs())
    payload = {"along_error_m": along, "horizon_sec": 600.0, "mission_reserve_m_s": 0.0}
    fleet.step(1.0, {"SC-2": "phase"}, {"SC-2": payload})
    assert fleet.vehicles["SC-2"]._phase_plan["accepted"] is True
    spent = abs(fleet.vehicles["SC-2"]._phase_plan["dv_m_s"])
    assert fleet.vehicles["SC-2"].spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(80.0 - spent)
    twin.step(1.0, {"SC-2": "wait"})
    burned = _along_rate(fleet)
    waited = _along_rate(twin)
    assert abs(burned - waited) > 0.5

    poor = Fleet.string_of_pearls(2, separation_deg=0.05, scenario=ScenarioConfig(ground_stations=[]))
    before = poor.vehicles["SC-2"].spacecraft.state.propulsion.delta_v_remaining_m_s
    poor.step(1.0, {"SC-2": "phase"}, {"SC-2": {**payload, "mission_reserve_m_s": 79.0}})
    assert poor.vehicles["SC-2"]._phase_reason == "бюджет миссии"
    assert poor.vehicles["SC-2"].spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(before)


def _along_rate(fleet: Fleet) -> float:
    leader = fleet.vehicles["SC-1"]
    follower = fleet.vehicles["SC-2"]
    relative = eci_to_rtn(follower.orbit.v - leader.orbit.v, leader.orbit.r, leader.orbit.v)
    return float(relative[1])
