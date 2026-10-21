"""Линк-бюджет, разворот, сопротивление, NRLMSISE, интеграл Pc, рост ковариации."""
import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.encounter import PropagationContext, secondary_from_geometry
from space_env.core.frames import gast_rad, gmst_rad
from space_env.core.gravity import ambient_density_kg_m3, density_nrlmsise_kg_m3
from space_env.core.link import free_space_path_loss_db, link_margin_db
from space_env.core.orbit import Orbit
from space_env.core.probability import collision_probability, collision_probability_integral
from space_env.core.propagate import propagate_state
from space_env.core.time import Time
from space_env.eval.holdout import compare_policies
from space_env.rl.expert import expert_action


def test_gast_differs_from_gmst_by_nutation_only():
    moment = Time("2024-06-15T00:00:00Z")
    delta = abs(gast_rad(moment) - gmst_rad(moment))
    assert delta < np.deg2rad(20.0 / 3600.0)


def test_nutation_at_j2000_follows_the_18_year_term():
    from space_env.core.frames import nutation_rad

    moment = Time(2451545.0)
    longitude, _obliquity = nutation_rad(moment)
    arcsec = np.rad2deg(longitude) * 3600.0
    omega = np.deg2rad(450160.398036 / 3600.0)
    expected = -17.1996 * np.sin(omega)
    assert arcsec == pytest.approx(expected, abs=1.0)


def test_link_margin_is_higher_overhead_than_far_away():
    overhead = link_margin_db(
        400e3, frequency_hz=2.2e9, tx_power_w=2.0, tx_gain_dbi=6.0,
        rx_gain_dbi=32.0, system_loss_db=3.0, required_power_dbw=-125.0,
    )
    far = link_margin_db(
        2_500e3, frequency_hz=2.2e9, tx_power_w=2.0, tx_gain_dbi=6.0,
        rx_gain_dbi=32.0, system_loss_db=3.0, required_power_dbw=-125.0,
    )
    assert overhead > 0.0
    assert far < overhead
    assert free_space_path_loss_db(2_500e3, 2.2e9) > free_space_path_loss_db(400e3, 2.2e9)


def test_integral_matches_chan_at_the_origin_and_falls_with_miss():
    sigma = 200.0
    radius = 15.0
    cov = np.eye(2) * sigma**2
    at_origin = collision_probability_integral(np.zeros(2), cov, radius)
    chan = collision_probability(0.0, radius, sigma)
    assert at_origin == pytest.approx(chan, rel=0.25)
    far = collision_probability_integral(np.array([3000.0, 0.0]), cov, radius)
    assert far < at_origin * 0.01


def test_drag_lowers_sma_over_a_day():
    a = 6_671_000.0
    speed = np.sqrt(MU_EARTH / a)
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, speed, 0.0])
    r2, v2, _samples = propagate_state(
        r, v, 86_400.0,
        mass_kg=100.0, drag_area_m2=0.9, drag_coefficient=2.2,
        include_j2=False, substep_sec=30.0, density_scale=1.0, f107=150.0,
    )
    sma = 1.0 / (2.0 / np.linalg.norm(r2) - np.dot(v2, v2) / MU_EARTH)
    assert sma < a - 200.0


def test_covariance_grows_and_burn_moves_the_miss():
    a = 6_800_000.0
    speed = np.sqrt(MU_EARTH / a)
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, speed, 0.0])
    debris = PropagationContext(mass_kg=50.0, drag_area_m2=0.2, drag_coefficient=2.2, include_drag=False)
    primary = PropagationContext(mass_kg=100.0, drag_area_m2=0.0, drag_coefficient=0.0, include_drag=False)
    enc = secondary_from_geometry(r, v, 200.0, 40.0, 120.0, 600.0, 200.0, 600.0, 15.0, debris)
    before = float(np.trace(enc.covariance[:3, :3]))
    enc.advance(120.0, np.sqrt(MU_EARTH / a**3))
    after = float(np.trace(enc.covariance[:3, :3]))
    assert after > before
    miss_before, _pc, _rx, _rz = enc.predict(r, v, primary, 700.0)
    v_burned = v + np.array([0.0, 1.0, 0.0])
    miss_after, _pc2, _rx2, _rz2 = enc.predict(r, v_burned, primary, 700.0)
    assert abs(miss_after - miss_before) > 100.0


def test_slew_is_not_instant():
    from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
    from space_env.mission import ScenarioConfig
    from space_env.simulation import SimulationEngine

    epoch = Time("2024-06-21T12:00:00Z")
    orbit = Orbit.from_keplerian(6_800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    craft = Spacecraft(config=SpacecraftConfig(slew_rate_deg_s=1.0))
    craft.reset_state()
    engine = SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))
    engine.step(0.2, "point_sun")
    assert engine.get_state_snapshot().slew_remaining_deg > 0.0
    engine.step(400.0, "wait")
    assert engine.get_state_snapshot().slew_remaining_deg == 0.0
    from space_env.core.sun import sun_position_m

    sun = sun_position_m(engine._now())
    sun = sun / np.linalg.norm(sun)
    assert float(np.dot(engine.boresight_eci(), sun)) > 0.999


def test_polar_motion_shifts_ecef_by_metres_and_roundtrips():
    from space_env.core.dut1 import pole_arcsec
    from space_env.core.frames import _rotate_z, ecef_to_eci, eci_to_ecef

    moment = Time("2026-07-16T00:00:00Z")
    xp, yp = pole_arcsec(moment)
    assert xp == pytest.approx(0.212967, abs=1e-4)
    assert yp == pytest.approx(0.376427, abs=1e-4)
    vector = np.array([6_500_000.0, 0.0, 0.0])
    full = eci_to_ecef(vector, moment)
    spin_only = _rotate_z(vector, gast_rad(moment))
    shift = float(np.linalg.norm(full - spin_only))
    assert 1.0 < shift < 40.0
    assert np.allclose(ecef_to_eci(full, moment), vector, atol=1e-4)
    quiet = Time("2024-01-01T00:00:00Z")
    assert float(np.linalg.norm(eci_to_ecef(vector, quiet) - _rotate_z(vector, gast_rad(quiet)))) < 1e-6


def test_swpc_snapshot_supplies_flux_on_the_tle_day():
    from space_env.catalog.build import make_engine
    from space_env.core.space_weather import indices_for

    observed = indices_for(Time("2026-10-20T12:00:00Z"))
    assert observed is not None
    assert observed.f107 == pytest.approx(121.0)
    assert observed.ap == pytest.approx(8.0)
    assert observed.f107a == pytest.approx(113.5669, abs=0.01)
    assert indices_for(Time("2024-01-01T00:00:00Z")) is None
    engine = make_engine("sentinel-2a", epoch="2026-10-20T00:00:00Z", with_stations=False)
    assert engine.scenario.f107 == pytest.approx(121.0)
    assert engine.scenario.ap == pytest.approx(8.0)


def test_dut1_from_the_iers_snapshot_shifts_gmst():
    from space_env.core.constants import OMEGA_EARTH
    from space_env.core.dut1 import dut1_seconds

    inside = Time("2026-07-16T00:00:00Z")
    assert dut1_seconds(inside) == pytest.approx(0.0115424, abs=1e-6)
    assert dut1_seconds(Time("2024-01-01T00:00:00Z")) == 0.0
    day = inside.jd - 2451545.0
    manual = ((18.697374558 + 24.06570982441908 * day) % 24.0) / 24.0 * 2.0 * np.pi
    delta = (gmst_rad(inside) - manual + np.pi) % (2.0 * np.pi) - np.pi
    assert delta == pytest.approx(OMEGA_EARTH * dut1_seconds(inside), rel=0.01)


def test_moon_stays_near_its_orbit_and_the_leo_tide_is_small():
    from datetime import datetime

    from space_env.core.bodies import moon_position_m
    from space_env.core.gravity import acceleration_m_s2

    moon = moon_position_m(Time("2024-01-01T00:00:00Z"))
    assert 340.0e6 < float(np.linalg.norm(moon)) < 430.0e6
    later = moon_position_m(Time("2024-01-08T00:00:00Z"))
    cosine = float(np.dot(moon, later) / (np.linalg.norm(moon) * np.linalg.norm(later)))
    assert cosine < 0.95
    position = np.array([7.0e6, 0.0, 0.0])
    velocity = np.array([0.0, 7.5e3, 0.0])
    when = datetime(2024, 1, 1)
    common = dict(
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=False,
        include_drag=False,
        when=when,
    )
    bare = acceleration_m_s2(position, velocity, **common)
    tide = acceleration_m_s2(position, velocity, include_third_body=True, **common)
    extra = float(np.linalg.norm(tide - bare))
    assert 1.0e-7 < extra < 5.0e-6


def test_nrlmsise_density_at_400_km():
    pytest.importorskip("nrlmsise00")
    from datetime import datetime

    rho = density_nrlmsise_kg_m3(400_000.0, datetime(2024, 1, 1), 0.0, 0.0, f107=150.0, f107a=150.0, ap=4.0)
    assert 1e-13 < rho < 1e-10


def test_expert_beats_waiting_on_radio_fault():
    action = expert_action("space_env/AnomalyResponse-v0", np.array([0.6, 0.4, 0.5, 0.0, 0.5, 1.0]))
    assert action == 3
    summary = compare_policies("space_env/ResourceManager-v0", episodes=1, max_episode_steps=4, seed=0)
    assert "expert_success_rate" in summary
