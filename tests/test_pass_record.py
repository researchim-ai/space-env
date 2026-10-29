"""Ёмкость прохода, смаз кадра и одинаковый исход двух прогонов."""
import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, elevation_deg, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.link import pass_capacity_bits
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.eval.run_record import freeze_run, same_outcome
from space_env.mission.quality import frame_grade
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld
from space_env.core.access import subsatellite_latlon

EPOCH = Time("2024-01-01T00:00:00Z")


def _place(lon_deg: float) -> SimulationEngine:
    position = ecef_to_eci(geodetic_to_ecef(0.0, lon_deg, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, position / np.linalg.norm(position)))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    direction = np.cross(position, axis)
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


def _lon_at_elevation(target_deg: float) -> float:
    lo, hi = 0.0, 25.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        engine = _place(mid)
        elevation = elevation_deg(engine.orbit.r, 0.0, 0.0, 0.0, EPOCH)
        if elevation > target_deg:
            lo = mid
        else:
            hi = mid
    return hi


def test_low_elevation_carries_fewer_bits_than_the_whole_window_at_full_rate():
    assert pass_capacity_bits([(5.0, 10.0)], 1.0e6) == pytest.approx(5.0e6)
    assert pass_capacity_bits([(5.0, 1.0)], 1.0e6) == pytest.approx(1.25e6)
    assert pass_capacity_bits([(5.0, 1.0)], 1.0e6) < 5.0 * 1.0e6
    zenith = _place(0.0)
    low = _place(_lon_at_elevation(10.0))
    assert elevation_deg(low.orbit.r, 0.0, 0.0, 0.0, EPOCH) == pytest.approx(10.0, abs=0.2)
    zenith.step(1.0, "comms")
    low.step(1.0, "comms")
    assert zenith.get_state_snapshot().in_contact is True
    assert low.get_state_snapshot().in_contact is True
    assert low.delivered_bits > 0.0
    assert low.delivered_bits < zenith.delivered_bits * 0.5
    assert zenith.delivered_bits <= zenith.pass_capacity_bits + 1.0
    assert low.delivered_bits <= low.pass_capacity_bits + 1.0


def _over_target() -> SimulationEngine:
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
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    engine.spacecraft.config.slew_accel_deg_s2 = 2.0
    engine.spacecraft.config.slew_rate_deg_s = 10.0
    engine.spacecraft.config.slew_settle_deg = 180.0
    return engine


def test_smear_marks_a_moving_optical_frame_and_leaves_the_default_alone():
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, slew_rate_deg_s=2.0, smear_rate_deg_s=0.0) == "full"
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, slew_rate_deg_s=2.0, smear_rate_deg_s=1.0) == "partial"
    assert frame_grade(
        off_nadir_deg=0.0,
        half_fov_deg=30.0,
        slew_rate_deg_s=2.0,
        smear_rate_deg_s=1.0,
        images_in_eclipse=True,
    ) == "full"
    assert frame_grade(
        off_nadir_deg=0.0,
        half_fov_deg=30.0,
        cloud_fraction=0.8,
        slew_rate_deg_s=2.0,
        smear_rate_deg_s=1.0,
    ) == "none"

    smeared = _over_target()
    plain = _over_target()
    smeared.spacecraft.config.payload.smear_rate_deg_s = 0.5
    for engine in (smeared, plain):
        engine.step(2.0, "point_sun")
        engine.spacecraft.state.pointing = "nadir"
        engine._slew_remaining_deg = engine._boresight_separation_deg()
        engine._slew_span_deg = max(engine._slew_remaining_deg, 1.0)
    smeared.step(1.0, "imaging")
    plain.step(1.0, "imaging")
    assert smeared._slew_rate_deg_s > 0.5
    assert smeared._products[0]["quality"] == "partial"
    assert plain._products[0]["quality"] == "full"


def test_two_equal_runs_freeze_to_the_same_outcome():
    def world() -> MissionWorld:
        return MissionWorld({"a": _place(0.0)}, epoch="2024-01-01T00:00:00Z", horizon_sec=30.0)

    left = world()
    right = world()
    other = world()
    commands = [{"actor_id": "a", "kind": "comms", "dt_sec": 1.0}]
    left.step(1.0, {"a": "comms"})
    right.step(1.0, {"a": "comms"})
    other.step(1.0, {"a": "wait"})
    first = freeze_run(left, seed=7, commands=commands)
    second = freeze_run(right, seed=7, commands=commands)
    skipped = freeze_run(other, seed=7, commands=commands)
    assert same_outcome(first, second) is True
    assert same_outcome(first, skipped) is False
    assert first["seed"] == 7
    assert "health" not in first["vehicles"]["a"]
    renamed = dict(first)
    renamed["commands"] = []
    assert same_outcome(first, renamed) is True
