"""Заявки на общих часах: цепочка сдачи и одно зеркало."""
from pathlib import Path

import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.tasking import (
    SlotOffer,
    StationOutage,
    TaskRequest,
    exact_slot_plan,
    requests_from_scenario,
    run_policy,
    run_scenario,
)
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

_EO = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "eo_72h.yaml"
_EO16 = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "eo_72h_16.yaml"


def _overhead(
    name: str,
    request_id: str,
    bits: float,
    *,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    station: str | None = "Equator",
    radar: bool = False,
) -> SimulationEngine:
    epoch = Time("2026-10-20T12:00:00Z")
    position = ecef_to_eci(geodetic_to_ecef(lat_deg, lon_deg, 400_000.0), epoch)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    radial = position / np.linalg.norm(position)
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(radial, axis))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(radial, axis)
    tangent = tangent / np.linalg.norm(tangent)
    craft = Spacecraft(config=SpacecraftConfig(name=name))
    craft.reset_state()
    craft.config.payload.half_fov_deg = 70.0
    craft.config.payload.store_rate_bps = 1000.0
    craft.config.payload.memory_bits = 0.0
    craft.config.payload.images_in_eclipse = radar
    stations = [] if station is None else [GroundStation(name=station, lat_deg=lat_deg, lon_deg=lon_deg, min_elevation_deg=5.0)]
    engine = SimulationEngine(
        Orbit.from_vectors(position, tangent * speed, epoch),
        craft,
        scenario=ScenarioConfig(ground_stations=stations),
    )
    if bits > 0.0:
        engine.accept_bits(bits, origin="imaging", request_id=request_id)
    return engine


def test_one_request_is_imaged_and_receipted():
    engine = _overhead("solo", "scene", 0.0)
    world = MissionWorld({"solo": engine}, epoch="2026-10-20T12:00:00Z", horizon_sec=20.0)
    from space_env.core.access import subsatellite_latlon

    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    request = TaskRequest("scene", lat, lon, 0.0, 20.0, 1000.0, 1, "optical")
    report = run_policy(world, [request], step_sec=10.0, policy="fifo")
    assert any(item["request_id"] == "scene" and item["destination"] == "Equator" for item in report["receipts"])
    assert report["unmet"] == []
    assert report["score"]["on_time"] == ["scene"]
    assert report["score"]["weighted_on_time"] == 1.0


def test_one_station_serves_the_higher_priority_only():
    world = MissionWorld(
        {
            "alpha": _overhead("alpha", "high", 5.0e6),
            "beta": _overhead("beta", "low", 5.0e6),
        },
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )
    requests = [
        TaskRequest("high", 0.0, 0.0, 0.0, 100.0, 1.0, 5, "optical"),
        TaskRequest("low", 0.0, 0.0, 0.0, 100.0, 1.0, 1, "optical"),
    ]
    run_policy(world, requests, step_sec=10.0, policy="priority")
    assert world.engines["alpha"].delivered_bits > 0.0
    assert world.engines["beta"].delivered_bits == 0.0


def test_a_request_released_later_is_not_imaged_before_its_time():
    engine = _overhead("solo", "later", 0.0)
    world = MissionWorld({"solo": engine}, epoch="2026-10-20T12:00:00Z", horizon_sec=30.0)
    from space_env.core.access import subsatellite_latlon

    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    request = TaskRequest("later", lat, lon, 10.0, 30.0, 1000.0, 1, "optical")
    report = run_policy(world, [request], step_sec=10.0, policy="fifo")
    times = [float(item["time_sec"]) for item in report["receipts"] if item["request_id"] == "later"]
    assert times
    assert min(times) >= 10.0


def test_station_outage_blocks_delivery_and_names_the_reason():
    engine = _overhead("solo", "held", 5.0e6)
    world = MissionWorld({"solo": engine}, epoch="2026-10-20T12:00:00Z", horizon_sec=10.0)
    request = TaskRequest("held", 0.0, 0.0, 0.0, 10.0, 1.0, 1, "optical")
    report = run_policy(
        world,
        [request],
        step_sec=10.0,
        policy="fifo",
        outages=[StationOutage("Equator", 0.0, 10.0)],
    )
    assert engine.delivered_bits == 0.0
    assert report["unmet"] == [{"request_id": "held", "reason": "станция недоступна", "delivered_bits": 0.0}]
    assert report["score"]["weighted_on_time"] == 0.0
    assert report["score"]["unmet_by_reason"] == {"станция недоступна": 1}


def test_two_stations_deliver_in_the_same_step():
    world = MissionWorld(
        {
            "west": _overhead("west", "west-scene", 5.0e6, lon_deg=0.0, station="West"),
            "east": _overhead("east", "east-scene", 5.0e6, lon_deg=90.0, station="East"),
        },
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )
    requests = [
        TaskRequest("west-scene", 0.0, 0.0, 0.0, 100.0, 1.0, 2, "optical"),
        TaskRequest("east-scene", 0.0, 90.0, 0.0, 100.0, 1.0, 2, "optical"),
    ]
    report = run_policy(world, requests, step_sec=10.0, policy="fifo")
    destinations = {item["request_id"]: item["destination"] for item in report["receipts"]}
    assert destinations["west-scene"] == "West"
    assert destinations["east-scene"] == "East"
    assert report["unmet"] == []


def test_optical_and_radar_image_their_own_requests():
    engine = _overhead("optic", "unused", 0.0, station=None)
    from space_env.core.access import subsatellite_latlon

    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    world = MissionWorld(
        {
            "optic": engine,
            "radar": _overhead("radar", "unused", 0.0, station=None, radar=True),
        },
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )
    requests = [
        TaskRequest("day", lat, lon, 0.0, 10.0, 1000.0, 1, "optical"),
        TaskRequest("night", lat, lon, 0.0, 10.0, 1000.0, 1, "sar"),
    ]
    run_policy(world, requests, step_sec=10.0, policy="fifo")
    assert {item["request_id"] for item in world.engines["optic"]._products} == {"day"}
    assert {item["request_id"] for item in world.engines["radar"]._products} == {"night"}


def test_a_later_higher_priority_request_takes_the_next_image():
    engine = _overhead("solo", "early", 0.0, station=None)
    from space_env.core.access import subsatellite_latlon

    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    world = MissionWorld({"solo": engine}, epoch="2026-10-20T12:00:00Z", horizon_sec=20.0)
    requests = [
        TaskRequest("early", lat, lon, 0.0, 40.0, 100_000.0, 1, "optical"),
        TaskRequest("urgent", lat, lon, 10.0, 40.0, 1000.0, 9, "optical"),
    ]
    run_policy(world, requests, step_sec=10.0, policy="priority")
    early = next(item for item in engine._products if item["request_id"] == "early")
    urgent = next(item for item in engine._products if item["request_id"] == "urgent")
    assert early["generated_time_sec"] < 10.0
    assert urgent["generated_time_sec"] >= 10.0
    assert early["size_bits"] < 15_000.0
    assert urgent["size_bits"] > 5_000.0


def test_score_counts_only_the_station_that_stayed_open():
    world = MissionWorld(
        {
            "west": _overhead("west", "open", 5.0e6, lon_deg=0.0, station="West"),
            "east": _overhead("east", "closed", 5.0e6, lon_deg=90.0, station="East"),
        },
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )
    requests = [
        TaskRequest("open", 0.0, 0.0, 0.0, 100.0, 1.0, 5, "optical"),
        TaskRequest("closed", 0.0, 90.0, 0.0, 10.0, 1.0, 1, "optical"),
    ]
    report = run_policy(
        world,
        requests,
        step_sec=10.0,
        policy="priority",
        outages=[StationOutage("East", 0.0, 10.0)],
    )
    assert report["score"]["on_time"] == ["open"]
    assert report["score"]["weighted_on_time"] == 5.0 / 6.0
    assert report["score"]["unmet_by_reason"] == {"станция недоступна": 1}


def test_a_pass_that_opens_inside_the_step_is_still_imaged():
    spec = load_scenario_v2(_EO)
    requests = requests_from_scenario(spec)

    def flown(step_sec: float) -> dict:
        world = world_from_scenario(_EO, attach_stations=True)
        world.horizon_sec = 1200.0
        report = run_policy(world, requests, step_sec=step_sec, policy="priority")
        assert world.engines["sentinel-a"].time_sec == pytest.approx(1200.0)
        return report

    coarse = flown(60.0)
    fine = flown(30.0)

    def station(report: dict, request_id: str) -> str:
        mine = [item for item in report["receipts"] if item["request_id"] == request_id]
        mine.sort(key=lambda item: float(item["time_sec"]))
        return str(mine[0]["destination"])

    assert station(coarse, "kiruna-day") == station(fine, "kiruna-day") == "Kiruna"
    assert station(coarse, "svalbard-radar") == station(fine, "svalbard-radar") == "Svalbard"
    assert coarse["kpis"]["payload_energy_j"] > 0.0


def test_scenario_file_requests_load_without_a_seventy_two_hour_run():
    loaded = requests_from_scenario(load_scenario_v2(_EO))
    assert [item.request_id for item in loaded] == ["kiruna-day", "svalbard-radar"]
    assert loaded[0].payload == "optical"
    assert loaded[1].payload == "sar"
    assert loaded[0].priority == 2
    assert loaded[0].deadline_sec == 86400.0


def test_exact_assignment_prefers_weight_and_refuses_a_large_instance():
    single = exact_slot_plan([SlotOffer("alpha", 0, 5.0), SlotOffer("beta", 0, 1.0)])
    assert [item.vehicle for item in single] == ["alpha"]
    both = exact_slot_plan([SlotOffer("alpha", 0, 5.0), SlotOffer("beta", 1, 4.0)])
    assert {(item.vehicle, item.slot) for item in both} == {("alpha", 0), ("beta", 1)}
    with pytest.raises(ValueError, match="проверяемого"):
        exact_slot_plan([SlotOffer(f"v{index}", slot, 1.0) for index in range(4) for slot in range(4)])


def _pair_world() -> MissionWorld:
    return MissionWorld(
        {
            "west": _overhead("west", "west-scene", 5.0e6, lon_deg=0.0, station="West"),
            "east": _overhead("east", "east-scene", 5.0e6, lon_deg=90.0, station="East"),
        },
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )


def test_fifo_and_priority_both_deliver_on_two_stations():
    requests = [
        TaskRequest("west-scene", 0.0, 0.0, 0.0, 100.0, 1.0, 2, "optical"),
        TaskRequest("east-scene", 0.0, 90.0, 0.0, 100.0, 1.0, 2, "optical"),
    ]
    fifo = run_policy(_pair_world(), requests, step_sec=10.0, policy="fifo")
    priority = run_policy(_pair_world(), requests, step_sec=10.0, policy="priority")
    assert fifo["score"]["weighted_on_time"] == 1.0
    assert priority["score"]["weighted_on_time"] == 1.0
    assert fifo["kpis"]["safety_violations"] == 0
    assert priority["kpis"]["replans"] == 1


def test_a_failed_radio_relays_the_request_to_the_neighbor():
    dest = _overhead("down", "scene", 0.0)
    axis = np.cross(dest.orbit.r, dest.orbit.v)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(1.5)

    def turn(vec: np.ndarray) -> np.ndarray:
        return (
            vec * np.cos(theta)
            + np.cross(axis, vec) * np.sin(theta)
            + axis * float(np.dot(axis, vec)) * (1.0 - np.cos(theta))
        )

    craft = Spacecraft(config=SpacecraftConfig(name="up"))
    craft.reset_state()
    craft.config.payload.memory_bits = 0.0
    source = SimulationEngine(
        Orbit.from_vectors(turn(dest.orbit.r), turn(dest.orbit.v), dest.orbit.epoch),
        craft,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    source.accept_bits(5.0e6, origin="imaging", request_id="scene")
    source.arm_fault("radio_latch")
    world = MissionWorld(
        {"down": dest, "up": source},
        epoch="2026-10-20T12:00:00Z",
        horizon_sec=10.0,
    )
    report = run_policy(
        world,
        [TaskRequest("scene", 0.0, 0.0, 0.0, 100.0, 1.0, 1, "optical")],
        step_sec=10.0,
        policy="fifo",
    )
    assert report["relays"][0]["source"] == "up"
    assert report["relays"][0]["destination"] == "down"
    assert any(item["request_id"] == "scene" and item["destination"] == "Equator" for item in report["receipts"])
    assert source.delivered_bits == 0.0
    assert report["kpis"]["relay_bits"] > 0.0
    assert report["score"]["on_time"] == ["scene"]
    assert report["kpis"]["active_faults"] == 1


def test_seventy_two_hour_file_delivers_kiruna_and_names_every_request():
    report = run_scenario(_EO, step_sec=30.0)
    assert report["manifest"]["horizon_sec"] == 259200.0
    assert report["kpis"]["replans"] >= 8640
    assert "kiruna-day" in report["score"]["on_time"]
    assert "svalbard-radar" in report["score"]["on_time"]
    assert any(
        item["request_id"] == "kiruna-day" and item["destination"] == "Kiruna" for item in report["receipts"]
    )
    assert any(
        item["request_id"] == "svalbard-radar" and item["destination"] == "Svalbard" for item in report["receipts"]
    )
    assert report["unmet"] == []
    assert report["kpis"]["quality_full_fraction"] == 1.0
    assert report["kpis"]["payload_energy_j"] > 0.0
    assert report["kpis"]["mean_delay_sec"] is not None
    assert report["kpis"]["mean_delay_sec"] < 86400.0
    assert report["kpis"]["fuel_m_s"] == 0.0
    assert report["kpis"]["safety_violations"] > 0
    assert report["kpis"]["ground_bits"] >= 8.0e6
    assert report["kpis"]["runtime_sec"] > 0.0


def test_sixteen_vehicles_finish_seventy_two_hours_with_a_reason_for_every_request():
    report = run_scenario(_EO16, step_sec=60.0)
    names = report["manifest"]["instances"]
    requests = set(report["manifest"]["requests"])
    on_time = set(report["score"]["on_time"])
    unmet = {item["request_id"] for item in report["unmet"]}
    assert len(names) == 16
    assert len(set(names)) == 16
    assert "sentinel-2a" in report["manifest"]["models"]
    assert "landsat-9" in report["manifest"]["models"]
    assert "sar-small" in report["manifest"]["models"]
    assert report["manifest"]["horizon_sec"] == 259200.0
    assert on_time.isdisjoint(unmet)
    assert on_time | unmet == requests
    assert report["kpis"]["replans"] >= 4320
    assert report["kpis"]["runtime_sec"] > 0.0
    assert report["kpis"]["safety_violations"] > 0
    assert report["kpis"]["fuel_m_s"] == 0.0
