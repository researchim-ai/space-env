"""Деградация, длящаяся команда, случайная облачность и выдержка ретрансляции."""
import random

import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, SubsystemHealth
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.quality import cloud_cover, frame_grade
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.commands import MissionCommand
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from space_env.simulation.world import MissionWorld

EPOCH = Time("2024-01-01T00:00:00Z")


def _overhead() -> SimulationEngine:
    position = ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    direction = np.cross(position, np.array([0.0, 0.0, 1.0]))
    direction = direction / np.linalg.norm(direction)
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(position, direction * speed, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[station]),
    )
    engine.accept_bits(20.0e6, origin="imaging", request_id="frame")
    return engine


def _imager(seed: int, sigma: float) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    velocity = np.cross(radius, np.array([0.0, 0.0, 1.0]))
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[]),
        seed=seed,
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon, cloud_sigma=sigma))
    return engine


def test_degraded_radio_sends_half_and_degraded_camera_writes_a_partial_frame():
    clear = _overhead()
    weak = _overhead()
    weak.spacecraft.state.health["comms"] = SubsystemHealth.DEGRADED
    clear.step(1.0, "comms")
    weak.step(1.0, "comms")
    assert weak.delivered_bits == pytest.approx(clear.delivered_bits * 0.5, rel=1e-6)
    assert weak.delivered_bits > 0.0

    nominal = _imager(1, 0.0)
    degraded = _imager(1, 0.0)
    degraded.spacecraft.state.health["payload"] = SubsystemHealth.DEGRADED
    nominal.step(10.0, "imaging")
    degraded.step(10.0, "imaging")
    assert nominal._products[0]["quality"] == "full"
    assert degraded._products[0]["quality"] == "partial"
    assert degraded._imaging_count == 1


def test_cloud_draw_is_the_same_for_one_seed_and_can_block_the_frame():
    assert cloud_cover(0.0, 0.0, 1.0) == pytest.approx(0.0)
    assert cloud_cover(0.0, 1.0, 1.0) == pytest.approx(1.0)
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, cloud_fraction=cloud_cover(0.0, 1.0, 1.0)) == "none"
    blocked = next(seed for seed in range(40) if cloud_cover(0.0, 1.0, random.Random(seed).random()) >= 0.8)
    open_seed = next(seed for seed in range(40) if cloud_cover(0.0, 1.0, random.Random(seed).random()) <= 0.2)
    dark = _imager(blocked, 1.0)
    again = _imager(blocked, 1.0)
    bright = _imager(open_seed, 1.0)
    dark.step(10.0, "imaging")
    again.step(10.0, "imaging")
    bright.step(10.0, "imaging")
    assert dark._imaging_count == 0
    assert again._imaging_count == 0
    assert bright._imaging_count == 1
    assert bright._products[0]["quality"] == "full"


def test_a_timed_pass_keeps_running_until_its_end_and_then_stops():
    def armed() -> MissionWorld:
        return MissionWorld({"a": _overhead()}, epoch="2024-01-01T00:00:00Z", horizon_sec=40.0)

    world = armed()
    orders = world.accept([MissionCommand("pass", "a", "comms", duration_sec=10.0)])
    world.step(4.0, orders)
    world.settle()
    assert world.commands.get("pass")["status"] == "running"
    first = world.engines["a"].delivered_bits
    assert first > 0.0
    with pytest.raises(ValueError, match="операция ещё идёт"):
        world.accept([MissionCommand("other", "a", "imaging")])
    world.step(4.0)
    world.settle()
    assert world.engines["a"].delivered_bits > first
    assert world.commands.get("pass")["status"] == "running"
    world.step(4.0)
    world.settle()
    assert world.commands.get("pass")["status"] == "completed"
    stopped = world.engines["a"].delivered_bits
    world.step(4.0)
    world.settle()
    assert world.engines["a"].delivered_bits == pytest.approx(stopped)

    cancelled = armed()
    orders = cancelled.accept([MissionCommand("pass", "a", "comms", duration_sec=10.0)])
    cancelled.step(4.0, orders)
    cancelled.settle()
    held = cancelled.engines["a"].delivered_bits
    cancelled.cancel_operation("pass")
    cancelled.step(4.0)
    cancelled.settle()
    assert cancelled.commands.get("pass")["status"] == "cancelled"
    assert cancelled.engines["a"].delivered_bits == pytest.approx(held)


def test_forward_hold_keeps_the_bundle_until_the_timer():
    fleet = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.vehicles["SC-2"].spacecraft.config.comms.forward_hold_sec = 5.0
    fleet.collect("SC-1", 1.0)
    assert fleet.relay("SC-1", "SC-2", 0.01) > 0.0
    assert fleet.relay("SC-2", "SC-3", 0.01) == pytest.approx(0.0)
    assert fleet.vehicles["SC-3"].stored_bits == pytest.approx(0.0)
    fleet.vehicles["SC-2"]._time_sec = 5.0
    assert fleet.relay("SC-2", "SC-3", 0.01) > 0.0
    assert fleet.vehicles["SC-3"].stored_bits > 0.0
