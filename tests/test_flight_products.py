"""Солнечное давление, окна связи и разбор CDM."""
import numpy as np
import pytest

from space_env.core.cdm import parse_cdm_kvn
from space_env.core.constants import SOLAR_PRESSURE_1AU_N_M2
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.gravity import acceleration_m_s2
from space_env.core.ground import GroundStation
from space_env.core.passes import find_passes
from space_env.core.sun import sun_position_m
from space_env.core.time import Time


_CDM = """
CCSDS_CDM_VERS = 1.0
TCA = 2024-01-01T00:10:00.000
MISS_DISTANCE = 0.250 [km]
RELATIVE_SPEED = 0.120 [km/s]
COLLISION_PROBABILITY = 1.2e-4
AREA_PC = 706.86 [m**2]
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 43123
CR_R = 0.04 [km**2]
CT_T = 2.0 [km**2]
CN_N = 0.04 [km**2]
CT_R = 0.0 [km**2]
CN_R = 0.0 [km**2]
CN_T = 0.0 [km**2]
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 25544
CR_R = 0.01 [km**2]
CT_T = 1.0 [km**2]
CN_N = 0.01 [km**2]
"""


def test_cdm_combines_covariances_and_hard_body():
    message = parse_cdm_kvn(_CDM)
    assert message.miss_distance_m == pytest.approx(250.0)
    assert message.relative_speed_m_s == pytest.approx(120.0)
    assert message.hard_body_radius_m == pytest.approx(15.0, rel=0.01)
    assert message.reported_probability == pytest.approx(1.2e-4)
    cov = message.combined_covariance_m2()
    assert cov[0, 0] == pytest.approx(0.05 * 1.0e6)
    assert message.combined_sigma_m() == pytest.approx(np.sqrt(0.05e6), rel=0.01)


def test_srp_pushes_away_from_the_sun_and_vanishes_in_umbra():
    epoch = Time("2024-06-15T00:00:00Z")
    sun = sun_position_m(epoch)
    s_hat = sun / np.linalg.norm(sun)
    r_sunlit = s_hat * 7.0e6
    r_dark = -s_hat * 7.0e6
    common = dict(mass_kg=100.0, drag_area_m2=0.0, drag_coefficient=0.0, include_j2=False, include_drag=False, srp_area_m2=1.2, srp_coefficient=1.2, when=epoch.datetime.replace(tzinfo=None))
    a_sun = acceleration_m_s2(r_sunlit, np.array([0.0, 7500.0, 0.0]), **common)
    a_dark = acceleration_m_s2(r_dark, np.array([0.0, 7500.0, 0.0]), **common)
    # Гравитация вдоль радиуса; давление — против направления на Солнце.
    srp = a_sun - (-3.986004418e14 * r_sunlit / np.linalg.norm(r_sunlit) ** 3)
    expected = 1.2 * (1.2 / 100.0) * SOLAR_PRESSURE_1AU_N_M2
    assert float(np.dot(srp, -s_hat)) == pytest.approx(expected, rel=0.05)
    srp_dark = a_dark - (-3.986004418e14 * r_dark / np.linalg.norm(r_dark) ** 3)
    assert float(np.linalg.norm(srp_dark)) < expected * 0.05


def test_pass_is_open_at_zenith():
    epoch = Time("2024-03-01T00:00:00Z")
    r_ecef = geodetic_to_ecef(0.0, 0.0, 500_000.0)
    r = ecef_to_eci(r_ecef, epoch)
    speed = 7610.0
    v = np.cross(r, np.array([0.0, 0.0, 1.0]))
    v = v / np.linalg.norm(v) * speed
    windows = find_passes(
        r, v, epoch,
        [GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)],
        horizon_sec=600.0,
        mass_kg=100.0,
        drag_area_m2=0.9,
        drag_coefficient=2.2,
        frequency_hz=2.2e9,
        tx_power_w=2.0,
        tx_gain_dbi=6.0,
        rx_gain_dbi=32.0,
        system_loss_db=3.0,
        required_power_dbw=-125.0,
        sample_sec=30.0,
    )
    assert windows
    assert windows[0].aos_sec == pytest.approx(0.0)
    assert windows[0].station == "Equator"
