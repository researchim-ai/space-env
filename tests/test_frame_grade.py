"""Качество кадра отдельно от экспозиции и от сдачи."""
import numpy as np

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.quality import frame_grade
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")


def test_cloud_and_off_nadir_grade_are_declared_before_the_run():
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, cloud_fraction=0.0) == "full"
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, cloud_fraction=0.8) == "none"
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, cloud_fraction=0.5) == "partial"
    assert frame_grade(off_nadir_deg=0.0, half_fov_deg=30.0, cloud_fraction=1.0, images_in_eclipse=True) == "full"
    assert frame_grade(off_nadir_deg=20.0, half_fov_deg=30.0, quality_off_nadir_deg=10.0) == "partial"
    assert frame_grade(off_nadir_deg=40.0, half_fov_deg=30.0) == "none"


def _engine(cloud: float) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    axis = np.array([0.0, 0.0, 1.0])
    velocity = np.cross(radius, axis)
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(
        GroundTarget(name="here", lat_deg=lat, lon_deg=lon, cloud_fraction=cloud)
    )
    return engine


def test_a_cloudy_optical_target_does_not_become_a_product():
    clear = _engine(0.0)
    cloudy = _engine(0.9)
    clear_snap, _events = clear.step(10.0, "imaging")
    cloudy_snap, _events = cloudy.step(10.0, "imaging")
    assert clear_snap.imaging_count == 1
    assert clear._products[0]["quality"] == "full"
    assert cloudy_snap.imaging_count == 0
    assert cloudy.last_command_ok is False
