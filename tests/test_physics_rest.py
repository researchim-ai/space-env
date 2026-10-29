"""Переход TEME, схождение с переменной плотностью, углы и сверка Pc."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, TEME
from astropy.time import Time as AstropyTime

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.frames import _teme_to_itrs, teme_to_numerical
from space_env.core.lifetime import circular_lifetime_years, drag_ode_lifetime_years
from space_env.core.orbit import Orbit
from space_env.core.probability import collision_probability, collision_probability_integral
from space_env.core.propagate import propagate_state
from space_env.core.time import Time
from space_env.estimation.knowledge import knowledge_claim, look_shift_rad, range_shift_m


def test_teme_itrs_matches_astropy_and_the_engine_frame_is_not_the_raw_vector():
    from astropy.utils import iers
    import astropy.coordinates.builtin_frames.intermediate_rotation_transforms as transforms

    from space_env.core.dut1 import dut1_seconds, pole_arcsec

    iers.conf.auto_download = False
    epoch = Time("2026-10-20T00:00:00Z")
    position = np.array([7_000_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7_500.0, 0.0])
    ours = _teme_to_itrs(position, epoch)
    ut1 = float(epoch.jd) + dut1_seconds(epoch) / 86400.0
    xp, yp = pole_arcsec(epoch)
    arcsec = np.pi / (180.0 * 3600.0)
    transforms.get_polar_motion = lambda time: (xp * arcsec, yp * arcsec)
    astropy_time = AstropyTime(ut1, format="jd", scale="ut1")
    teme = TEME(CartesianRepresentation(*(position * u.m)), obstime=astropy_time)
    itrs = teme.transform_to(ITRS(obstime=astropy_time))
    reference = np.array(
        [itrs.cartesian.x.to_value(u.m), itrs.cartesian.y.to_value(u.m), itrs.cartesian.z.to_value(u.m)]
    )
    assert np.linalg.norm(ours - reference) < 1.0e-3
    numerical, speed = teme_to_numerical(position, velocity, epoch)
    assert np.linalg.norm(numerical - position) > 1.0
    assert np.linalg.norm(speed) > 1_000.0


def test_teme_gcrs_matches_astropy_and_is_not_the_engine_vector():
    from astropy.utils import iers
    import astropy.coordinates.builtin_frames.intermediate_rotation_transforms as transforms

    from space_env.core.dut1 import dut1_seconds, pole_arcsec
    from space_env.core.frames import teme_to_gcrs

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    epoch = Time("2026-10-20T00:00:00Z")
    position = np.array([7_000_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7_500.0, 0.0])
    ut1 = float(epoch.jd) + dut1_seconds(epoch) / 86400.0
    xp, yp = pole_arcsec(epoch)
    arcsec = np.pi / (180.0 * 3600.0)
    transforms.get_polar_motion = lambda time: (xp * arcsec, yp * arcsec)
    astropy_time = AstropyTime(ut1, format="jd", scale="ut1")
    teme = TEME(CartesianRepresentation(*(position * u.m)), obstime=astropy_time)
    gcrs = teme.transform_to(GCRS(obstime=astropy_time))
    reference = np.array(
        [gcrs.cartesian.x.to_value(u.m), gcrs.cartesian.y.to_value(u.m), gcrs.cartesian.z.to_value(u.m)]
    )
    ours, _speed = teme_to_gcrs(position, velocity, epoch)
    numerical, _numerical_speed = teme_to_numerical(position, velocity, epoch)
    assert np.linalg.norm(ours - reference) < 1.0e-3
    assert np.linalg.norm(ours - numerical) > 1.0


def test_iau_moon_stays_in_the_orbit_envelope_and_sun_series_stay_close():
    from space_env.core.bodies import moon_iau2000_m, moon_meeus_m, moon_position_m
    from space_env.core.sun import sun_iau2000_m, sun_meeus_m, sun_position_m

    epoch = Time("2026-10-20T12:00:00Z")
    iau = moon_iau2000_m(epoch)
    meeus = moon_meeus_m(epoch)
    assert moon_position_m(epoch).shape == (3,)
    assert np.allclose(moon_position_m(epoch), iau)
    assert 3.50e8 < float(np.linalg.norm(iau)) < 4.10e8
    assert 3.50e8 < float(np.linalg.norm(meeus)) < 4.10e8
    assert 1.0e5 < float(np.linalg.norm(iau - meeus)) < 20.0e6
    sun_iau = sun_iau2000_m(epoch)
    assert np.allclose(sun_position_m(epoch), sun_iau)
    sun_meeus = sun_meeus_m(epoch)
    cosine = float(np.dot(sun_iau, sun_meeus) / (np.linalg.norm(sun_iau) * np.linalg.norm(sun_meeus)))
    assert 0.0 < np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < 1.0


def test_drag_ode_reenters_a_low_light_craft_and_holds_a_high_dense_one():
    low = drag_ode_lifetime_years(200_000.0, mass_kg=10.0, area_m2=1.0, drag_coefficient=2.2)
    high = drag_ode_lifetime_years(800_000.0, mass_kg=1_000.0, area_m2=0.01, drag_coefficient=2.2)
    assert low["model"] == "drag_ode_table_density"
    assert low["reentered"] is True
    assert float(low["years"]) < 1.0
    assert float(low["final_altitude_m"]) <= 120_000.0 + 1.0
    assert high["reentered"] is False
    assert float(high["years"]) == pytest.approx(25.0)
    scale = circular_lifetime_years(200_000.0, mass_kg=10.0, area_m2=1.0, drag_coefficient=2.2)
    assert scale["model"] == "exponential_scale_height"
    assert float(low["years"]) > 0.0


def test_angles_see_cross_track_while_range_does_not_and_pc_matches_the_disk():
    assert knowledge_claim(1, 180.0)["cross_track"] is False
    angles = knowledge_claim(1, 180.0, measurement="angles")
    assert angles["cross_track"] is True
    assert angles["operational_od"] is False
    epoch = Time("2024-06-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    lat, lon = subsatellite_latlon(orbit.r, epoch)
    position, velocity, _samples = propagate_state(
        orbit.r,
        orbit.v,
        90.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_drag=False,
        include_j2=False,
        substep_sec=10.0,
    )
    moment = epoch + 90.0
    _along_range, cross_range = range_shift_m(position, velocity, moment, lat, lon)
    _along_angle, cross_angle = look_shift_rad(position, velocity, moment, lat, lon)
    from space_env.core.frames import eci_to_ecef, geodetic_to_ecef

    slant = float(np.linalg.norm(eci_to_ecef(position, moment) - geodetic_to_ecef(lat, lon, 0.0)))
    assert cross_range < 50.0
    assert cross_angle * slant > 200.0

    chen = collision_probability(0.0, 10.0, 100.0)
    integral = collision_probability_integral(np.zeros(2), np.eye(2) * 100.0**2, 10.0)
    assert chen == pytest.approx(0.005)
    assert abs(integral - chen) / chen < 0.05
    assert collision_probability(5_000.0, 10.0, 50.0) == pytest.approx(0.0)
