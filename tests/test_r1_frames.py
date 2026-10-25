"""Регрессии R1.1: полночь UTC, цель за лимбом, подпись системы координат."""
import math

import numpy as np
import pytest
from astropy.time import Time as AstropyTime

from space_env.catalog.build import make_engine
from space_env.core.access import off_nadir_deg, subsatellite_latlon, target_access
from space_env.core.constants import R_EARTH_EQUATOR
from space_env.core.environment import in_eclipse
from space_env.core.frames import (
    FrameError,
    copy_state,
    ecef_to_eci,
    ecef_to_geodetic,
    geocentric_latitude_deg,
    geodetic_to_ecef,
    require_same_frame,
)
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PayloadConfig, Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.viz.export import _document


_UTC_SAMPLES = (
    "1970-01-01T00:00:00Z",
    "2024-02-29T06:00:00Z",
    "2024-03-01T00:00:00Z",
    "2025-12-31T23:30:00Z",
    "2026-01-01T00:00:00Z",
    "2026-01-31T23:00:00Z",
    "2026-02-01T00:00:00Z",
    "2026-10-20T00:00:00Z",
    "2026-10-20T00:00:00.400000Z",
    "2026-10-20T12:00:00Z",
    "2099-12-31T23:59:59Z",
)


def test_utc_roundtrip_stays_within_a_millisecond_of_astropy():
    epoch = Time("2026-10-20T00:00:00Z")
    assert epoch.datetime.isoformat() == "2026-10-20T00:00:00+00:00"
    assert Time(epoch.datetime).seconds_since(epoch) == pytest.approx(0.0, abs=1e-3)
    assert Time("2026-10-20T05:00:00+05:00").seconds_since(epoch) == pytest.approx(0.0, abs=1e-3)
    for text in _UTC_SAMPLES:
        moment = Time(text)
        reference = AstropyTime(text, scale="utc")
        assert moment.jd == pytest.approx(float(reference.jd), abs=2e-8)
        assert abs(Time(moment.datetime).seconds_since(moment)) < 1e-3
        back = AstropyTime(moment.datetime, scale="utc")
        assert abs(float((back - reference).to_value("sec"))) < 1e-3
    document = _document([], epoch, 60.0, None)
    assert document["epoch"] == "2026-10-20T00:00:00+00:00"


def _craft(images_in_eclipse: bool = False) -> Spacecraft:
    craft = Spacecraft(config=SpacecraftConfig(payload=PayloadConfig(images_in_eclipse=images_in_eclipse, half_fov_deg=30.0)))
    craft.reset_state()
    return craft


def _sunlit_state():
    epoch = Time("2026-10-20T00:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7_000_000.0
    tangent = np.cross(np.array([0.0, 0.0, 1.0]), position)
    velocity = tangent / np.linalg.norm(tangent) * 7500.0
    return epoch, position, velocity


def test_far_side_target_never_becomes_a_product_and_near_side_does():
    epoch, position, velocity = _sunlit_state()
    lat, lon = subsatellite_latlon(position, epoch)
    far_lon = (lon + 360.0) % 360.0 - 180.0
    assert off_nadir_deg(position, -lat, far_lon, epoch) == pytest.approx(0.0, abs=1.0)
    hidden = target_access(position, -lat, far_lon, epoch, half_fov_deg=30.0)
    assert hidden.line_of_sight is False
    assert hidden.visible is False
    near = target_access(position, lat, lon, epoch, half_fov_deg=30.0)
    assert near.line_of_sight is True
    assert near.visible is True

    def shoot(name: str, point_lat: float, point_lon: float) -> SimulationEngine:
        engine = SimulationEngine(
            Orbit.from_vectors(position, velocity, epoch),
            _craft(),
            scenario=ScenarioConfig(
                ground_stations=[],
                ground_targets=[GroundTarget(name=name, lat_deg=point_lat, lon_deg=point_lon)],
            ),
        )
        engine.step(1.0, "imaging")
        return engine

    far = shoot("far-side", -lat, far_lon)
    assert far.get_state_snapshot().imaging_count == 0
    assert far.last_command_ok is False
    close = shoot("near-side", lat, lon)
    assert close.get_state_snapshot().imaging_count == 1
    assert close.last_command_ok is True


def test_limb_stays_visible_and_the_point_beyond_it_does_not():
    epoch = Time("2026-06-21T12:00:00Z")
    height = R_EARTH_EQUATOR + 400_000.0
    theta = math.acos(R_EARTH_EQUATOR / height)
    limb = np.array([R_EARTH_EQUATOR * math.cos(theta), R_EARTH_EQUATOR * math.sin(theta), 0.0])
    beyond = np.array([R_EARTH_EQUATOR * math.cos(theta * 1.4), R_EARTH_EQUATOR * math.sin(theta * 1.4), 0.0])
    satellite = ecef_to_eci(np.array([height, 0.0, 0.0]), epoch)
    limb_lat, limb_lon, _alt = ecef_to_geodetic(limb)
    far_lat, far_lon, _alt = ecef_to_geodetic(beyond)
    limb_access = target_access(satellite, limb_lat, limb_lon, epoch, half_fov_deg=75.0, images_in_eclipse=True)
    narrow = target_access(satellite, limb_lat, limb_lon, epoch, half_fov_deg=30.0, images_in_eclipse=True)
    far_access = target_access(satellite, far_lat, far_lon, epoch, half_fov_deg=75.0, images_in_eclipse=True)
    assert limb_access.elevation_deg == pytest.approx(0.0, abs=0.5)
    assert limb_access.line_of_sight is True
    assert limb_access.in_fov is True
    assert limb_access.visible is True
    assert narrow.line_of_sight is True
    assert narrow.visible is False
    assert far_access.line_of_sight is False


def test_radar_keeps_a_night_target_and_optical_does_not():
    epoch = Time("2026-10-20T00:00:00Z")
    sun = sun_position_m(epoch)
    position = -sun / np.linalg.norm(sun) * 7_000_000.0
    lat, lon = subsatellite_latlon(position, epoch)
    assert in_eclipse(position, sun) is True
    optical = target_access(position, lat, lon, epoch, half_fov_deg=30.0)
    radar = target_access(position, lat, lon, epoch, half_fov_deg=30.0, images_in_eclipse=True)
    assert optical.line_of_sight is True
    assert optical.visible is False
    assert radar.visible is True


def test_zero_burn_and_two_body_step_keep_the_teme_label():
    epoch = Time("2026-10-20T00:00:00Z")
    orbit = Orbit.from_vectors(np.array([6_800_000.0, 0.0, 0.0]), np.array([0.0, 7600.0, 0.0]), epoch, frame="TEME")
    burned = orbit.with_delta_v(np.zeros(3))
    stepped = orbit.propagate(1.0)
    assert burned.frame == "TEME"
    assert stepped.frame == "TEME"
    assert np.allclose(burned.r, orbit.r)
    assert np.allclose(burned.v, orbit.v)
    with pytest.raises(FrameError):
        Orbit.from_vectors(orbit.r, orbit.v, epoch, frame="EME2000")
    with pytest.raises(FrameError):
        Orbit.from_vectors(orbit.r, orbit.v, epoch, frame="ITRF")
    with pytest.raises(FrameError):
        require_same_frame("GCRF", "TEME")
    with pytest.raises(FrameError):
        SimulationEngine(orbit, _craft(), scenario=ScenarioConfig(ground_stations=[]))
    with pytest.raises(FrameError, match="TEME"):
        make_engine("sentinel-2a", use_tle=True)
    position, velocity, covariance = copy_state(orbit.r, orbit.v, "TEME", np.eye(6))
    assert np.allclose(position, orbit.r)
    assert np.allclose(velocity, orbit.v)
    assert covariance.shape == (6, 6)
    with pytest.raises(FrameError):
        copy_state(orbit.r, orbit.v, "TEME", np.eye(3))


def test_geodetic_latitude_is_not_replaced_by_geocentric():
    ecef = geodetic_to_ecef(45.0, 10.0, 0.0)
    lat, lon, alt = ecef_to_geodetic(ecef)
    assert lat == pytest.approx(45.0, abs=1e-6)
    assert lon == pytest.approx(10.0, abs=1e-6)
    assert alt == pytest.approx(0.0, abs=1e-2)
    assert abs(geocentric_latitude_deg(ecef) - lat) > 0.1
