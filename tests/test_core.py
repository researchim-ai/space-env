"""Тесты ядра: время, орбита, КА, среда."""
from datetime import datetime, timezone

import numpy as np
import pytest

from space_env.core import (
    Time,
    Orbit,
    KeplerianElements,
    in_eclipse,
    R_EARTH_EQUATOR,
)
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, PowerConfig


def test_time_jd_roundtrip():
    dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    t = Time(dt)
    assert t.datetime.year == 2024
    t2 = Time(t.jd)
    assert abs(t2.jd - t.jd) < 1e-9


def test_time_add_seconds():
    t = Time("2024-01-01T00:00:00Z")
    t2 = t + 3600.0
    assert t2.seconds_since(t) == pytest.approx(3600.0)


def test_orbit_from_keplerian():
    # LEO: a ≈ 6800 km, e=0, i=51.6°
    a_m = 6800e3
    epoch = Time("2024-01-01T00:00:00Z")
    orb = Orbit.from_keplerian(
        a_m, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch
    )
    assert orb.epoch == epoch
    assert orb.altitude_km() == pytest.approx(422.0, abs=5.0)  # a - R_earth ≈ 422 km
    ke = orb.keplerian()
    assert ke.a == pytest.approx(a_m, rel=1e-6)
    assert ke.ecc == pytest.approx(0.0, abs=1e-10)


def test_orbit_propagate():
    a_m = 6800e3
    epoch = Time("2024-01-01T00:00:00Z")
    orb0 = Orbit.from_keplerian(a_m, 0.0, 0.0, 0.0, 0.0, 0.0, epoch)
    orb1 = orb0.propagate(600.0)  # 10 min
    assert orb1.epoch.seconds_since(epoch) == pytest.approx(600.0)
    # Высота должна остаться примерно той же (круговая)
    assert orb1.altitude_km() == pytest.approx(orb0.altitude_km(), abs=1.0)


def test_orbit_from_vectors():
    # Простая круговая орбита: r = (R+h, 0, 0), v = (0, v_circ, 0) в плоскости xy
    h_m = 400e3
    R = R_EARTH_EQUATOR
    r_m = np.array([R + h_m, 0.0, 0.0])
    v_circ = np.sqrt(398600.4418e9 / (R + h_m))  # м/с, упрощённо μ
    v_m_s = np.array([0.0, v_circ, 0.0])
    epoch = Time("2024-01-01T00:00:00Z")
    orb = Orbit.from_vectors(r_m, v_m_s, epoch)
    assert orb.altitude_km() == pytest.approx(400.0, abs=2.0)


def test_eclipse_behind_earth():
    # КА на оси Земля–Солнце за Землёй (в цилиндрической тени)
    r_sat = np.array([7e6, 0.0, 0.0])   # LEO по x; Солнце по +x
    r_sun = np.array([1.5e11, 0.0, 0.0])
    assert in_eclipse(r_sat, r_sun) is True  # в цилиндре тени


def test_eclipse_in_sun():
    # КА в стороне от оси Земля–Солнце — не в цилиндрической тени
    r_sun = np.array([1.5e11, 0.0, 0.0])
    r_sat_off_axis = np.array([7e6, 8e6, 0.0])  # перпендикулярно оси > R_EARTH
    assert in_eclipse(r_sat_off_axis, r_sun) is False


def test_spacecraft_config_and_reset():
    from space_env.core.spacecraft import PropulsionConfig

    cfg = SpacecraftConfig(
        name="TestSat",
        power=PowerConfig(solar_panel_area_m2=1.0, battery_capacity_Wh=100.0, consumption_nominal_W=5.0, consumption_peak_W=20.0),
        propulsion=PropulsionConfig(delta_v_total_m_s=100.0, mass_kg=10.0),
    )
    sc = Spacecraft(config=cfg)
    sc.reset_state()
    assert sc.state.power.battery_soc == 1.0
    assert sc.state.propulsion.delta_v_remaining_m_s == 100.0
