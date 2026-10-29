"""Прилив Солнца по ERFA и ретрансляция, которая двигает орбиту между хопами."""
import numpy as np
import pytest

from space_env.core.bodies import (
    MU_MOON_M3_S2,
    MU_SUN_M3_S2,
    moon_position_m,
    third_body_acceleration_m_s2,
    tidal_acceleration_m_s2,
)
from space_env.core.environment import shadow_state
from space_env.core.gravity import _srp_acceleration_m_s2
from space_env.core.sun import sun_iau2000_m, sun_meeus_m, sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet

EPOCH = Time("2026-10-20T12:00:00Z")


def test_shadow_pressure_and_tide_share_one_sun():
    position = np.array([7.0e6, 0.0, 0.0])
    operational = sun_position_m(EPOCH)
    iau = sun_iau2000_m(EPOCH)
    meeus = sun_meeus_m(EPOCH)
    assert np.allclose(operational, iau)
    cosine = float(np.dot(iau, meeus) / (np.linalg.norm(iau) * np.linalg.norm(meeus)))
    assert 0.0 < np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < 1.0
    got = third_body_acceleration_m_s2(position, EPOCH)
    expect = tidal_acceleration_m_s2(position, moon_position_m(EPOCH), MU_MOON_M3_S2)
    expect = expect + tidal_acceleration_m_s2(position, operational, MU_SUN_M3_S2)
    assert np.allclose(got, expect, rtol=0.0, atol=1e-16)
    other = tidal_acceleration_m_s2(position, moon_position_m(EPOCH), MU_MOON_M3_S2)
    other = other + tidal_acceleration_m_s2(position, meeus, MU_SUN_M3_S2)
    assert float(np.linalg.norm(got - other)) > 0.0
    hat = operational / np.linalg.norm(operational)
    moment = EPOCH.datetime.replace(tzinfo=None)
    dark = -hat * 7.0e6
    lit = hat * 7.0e6
    assert shadow_state(dark, operational).kind == "umbra"
    assert shadow_state(lit, operational).flux_fraction == pytest.approx(1.0)
    assert float(np.linalg.norm(_srp_acceleration_m_s2(dark, moment, 100.0, 1.0, 1.2))) == pytest.approx(0.0)
    pressure = _srp_acceleration_m_s2(lit, moment, 100.0, 1.0, 1.2)
    assert float(np.dot(pressure, hat)) < 0.0
    assert float(np.linalg.norm(pressure)) > 0.0


def test_live_forward_moves_the_orbit_between_hops():
    fleet = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.collect("SC-1", 1.0)
    start = fleet.vehicles["SC-1"].orbit.r.copy()
    result = fleet.forward_live("SC-1", ["SC-2", "SC-3"], 30.0)
    assert result["arrived"] is True
    assert result["holder"] == "SC-3"
    assert result["time_sec"] == pytest.approx(60.0)
    assert float(np.linalg.norm(fleet.vehicles["SC-1"].orbit.r - start)) > 1_000.0
    assert fleet.vehicles["SC-3"].stored_bits > 0.0

    frozen = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    frozen.collect("SC-1", 1.0)
    parked = frozen.forward("SC-1", ["SC-2", "SC-3"], 0.01)
    assert parked["arrived"] is True
    assert frozen.time_sec == pytest.approx(0.0)
