"""Разворот с ускорением и порог, после которого съёмка снова разрешена."""
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.attitude import align_z, slew_limited
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")


def test_limited_slew_starts_from_rest_and_stops_on_the_target():
    start = align_z(np.array([0.0, 0.0, 1.0]))
    goal = align_z(np.array([0.0, 0.0, -1.0]))
    turned, remaining, rate = slew_limited(start, goal, 0.0, 10.0, 2.0, 1.0)
    assert rate == pytest.approx(2.0, abs=1e-6)
    assert remaining == pytest.approx(179.0, abs=1e-3)
    assert float(np.dot(turned, start)) > 0.99
    finished, left, stopped = slew_limited(start, goal, 0.0, 10.0, 2.0, 40.0)
    assert left == pytest.approx(0.0, abs=1e-4)
    assert stopped == pytest.approx(0.0, abs=1e-4)
    assert finished == pytest.approx(goal, abs=1e-5)


def test_a_new_command_drops_the_slew_rate():
    engine = _engine(slew_rate_deg_s=10.0, slew_accel_deg_s2=2.0)
    engine.step(1.0, "point_sun")
    assert engine._slew_rate_deg_s == pytest.approx(2.0, abs=1e-3)
    held = engine.capture()
    engine.step(1.0, "wait")
    engine.restore(held)
    assert engine._slew_rate_deg_s == pytest.approx(2.0, abs=1e-3)
    engine.step(0.05, "point_nadir")
    assert engine._slew_rate_deg_s < 0.2


def test_imaging_waits_until_the_slew_is_inside_the_settle_angle():
    tight = _ready(slew_settle_deg=1.0)
    tight.step(20.0, "point_sun")
    tight.step(1.0, "point_nadir")
    snap, _events = tight.step(10.0, "imaging")
    assert tight._slew_remaining_deg > 1.0
    assert tight.last_command_ok is False
    assert snap.imaging_count == 0
    wide = _ready(slew_settle_deg=10.0)
    wide.step(20.0, "point_sun")
    wide.step(1.0, "point_nadir")
    snap, _events = wide.step(10.0, "imaging")
    assert wide._slew_remaining_deg < 10.0
    assert snap.imaging_count == 1


def _engine(**slew) -> SimulationEngine:
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, EPOCH)
    craft = Spacecraft(config=SpacecraftConfig(**slew))
    craft.reset_state()
    return SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))


def _ready(**slew) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    axis = np.array([0.0, 0.0, 1.0])
    velocity = np.cross(radius, axis)
    velocity = velocity / np.linalg.norm(velocity) * speed
    fields = {"slew_rate_deg_s": 0.2, "slew_accel_deg_s2": 0.2}
    fields.update(slew)
    craft = Spacecraft(config=SpacecraftConfig(**fields))
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine._now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    return engine
