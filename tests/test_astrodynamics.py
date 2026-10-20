"""J2, Клохесси–Уилтшир, вероятность Чена, один сеанс на зону, съёмка по цели."""
import numpy as np
import pytest

from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR
from space_env.core.frames import ecef_to_geodetic, geodetic_to_ecef
from space_env.core.orbit import Orbit
from space_env.core.probability import collision_probability
from space_env.core.propagate import propagate_state
from space_env.core.relative import closest_approach, cw_propagate, mean_motion_rad_s
from space_env.core.time import Time


def _raan_deg(r: np.ndarray, v: np.ndarray) -> float:
    h = np.cross(r, v)
    node = np.cross(np.array([0.0, 0.0, 1.0]), h)
    return float(np.rad2deg(np.arctan2(node[1], node[0])))


def test_geodetic_roundtrip():
    original = geodetic_to_ecef(48.0, 11.0, 500.0)
    lat, lon, alt = ecef_to_geodetic(original)
    assert lat == pytest.approx(48.0, abs=1e-4)
    assert lon == pytest.approx(11.0, abs=1e-4)
    assert alt == pytest.approx(500.0, abs=1.0)


def test_j2_off_matches_kepler_short_arc():
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, 0.0, 0.0, 0.0, 0.0, epoch)
    kepler = orbit.propagate(600.0)
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        600.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=False,
        include_drag=False,
        substep_sec=5.0,
    )
    assert np.linalg.norm(r - kepler.r) < 50.0


def test_j2_regresses_node_westward():
    """Средняя скорость узла для круговой орбиты: −(3/2) n J2 (Re/a)² cos i."""
    a = 6800e3
    inc = np.deg2rad(51.6)
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(a, 0.0, inc, np.deg2rad(40.0), 0.0, 0.0, epoch)
    hours = 6.0
    r, v, _samples = propagate_state(
        orbit.r,
        orbit.v,
        hours * 3600.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_j2=True,
        include_drag=False,
        substep_sec=20.0,
    )
    delta = _raan_deg(r, v) - 40.0
    if delta > 180.0:
        delta -= 360.0
    if delta < -180.0:
        delta += 360.0
    n = np.sqrt(MU_EARTH / a**3)
    analytical_deg = np.rad2deg(-1.5 * n * J2_EARTH * (R_EARTH_EQUATOR / a) ** 2 * np.cos(inc) * hours * 3600.0)
    assert delta < 0.0
    assert delta == pytest.approx(analytical_deg, rel=0.25)


def test_cw_roundtrip_recovers_miss():
    radius = 6800e3
    n = mean_motion_rad_s(radius)
    miss = np.array([250.0, 0.0, -80.0, 0.0, -120.0, 0.0])
    back = cw_propagate(miss, n, -500.0)
    forward = cw_propagate(back, n, 500.0)
    assert forward == pytest.approx(miss, abs=1e-4)
    distance, _t, state = closest_approach(back, n, 700.0, dt_sec=0.25)
    assert distance == pytest.approx(np.linalg.norm(miss[:3]), rel=0.02)
    assert abs(state[0]) == pytest.approx(250.0, rel=0.05)


def test_along_track_burn_changes_miss_without_a_fixed_gain():
    radius = 6800e3
    n = mean_motion_rad_s(radius)
    at_tca = np.array([200.0, 0.0, 50.0, 0.0, -120.0, 0.0])
    rel = cw_propagate(at_tca, n, -600.0)
    plain, _t, _state = closest_approach(rel, n, 800.0, dt_sec=0.25)
    burned = rel.copy()
    burned[4] += 1.0  # +1 м/с вдоль трека у цели относительно преследователя
    shifted, _t2, _state2 = closest_approach(burned, n, 800.0, dt_sec=0.25)
    assert plain == pytest.approx(np.hypot(200.0, 50.0), rel=0.05)
    assert abs(shifted - plain) > 50.0
    assert abs(shifted - plain) != pytest.approx(2000.0, abs=100.0)


def test_chan_probability_falls_as_miss_grows():
    close = collision_probability(100.0, 15.0, 200.0)
    far = collision_probability(2000.0, 15.0, 200.0)
    assert close > 1.0e-4
    assert far < 1.0e-6
    assert close > far


def test_one_comms_session_per_contact():
    from space_env.core.spacecraft import PowerConfig, PropulsionConfig, Spacecraft, SpacecraftConfig
    from space_env.mission import ScenarioConfig
    from space_env.simulation import SimulationEngine

    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, 0.0, 0.0, 0.0, 0.0, epoch)
    craft = Spacecraft(
        config=SpacecraftConfig(
            power=PowerConfig(),
            propulsion=PropulsionConfig(),
        )
    )
    craft.reset_state()
    engine = SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))
    engine._in_contact = True
    engine._command_comms([], 30.0)
    engine._command_comms([], 30.0)
    assert engine._comms_sessions == 1
    assert engine.last_command_ok is True
    engine._downlink_armed = False
    engine._command_comms([], 30.0)
    assert engine._comms_sessions == 2
