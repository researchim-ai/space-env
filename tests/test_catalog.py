"""Опорные борта: опубликованные числа, снимок TLE, геометрия полосы и звезды Иридиума."""
import numpy as np
import pytest

from space_env.catalog.build import mean_orbit, read_tle, sentinel2_design_fleet, tle_orbit
from space_env.catalog.geometry import (
    ASSUMED_HYDRAZINE_ISP_S,
    half_swath_deg,
    rocket_delta_v_m_s,
    slant_range_m,
)
from space_env.catalog.references import reference
from space_env.core.constants import R_EARTH_MEAN
from space_env.core.ground import KIRUNA
from space_env.core.link import earth_occults, link_margin_db
from space_env.core.sun import sun_position_m
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet


def test_landsat_and_iridium_keep_their_published_mass_and_rate():
    landsat = reference("landsat-9").spacecraft_config()
    iridium = reference("iridium-next").spacecraft_config()
    assert landsat.propulsion.mass_kg == 2623.0
    assert landsat.comms.data_rate_max_bps == pytest.approx(384e6)
    assert iridium.propulsion.mass_kg == 860.0
    assert 40.0 < iridium.payload.half_fov_deg < 70.0


def test_catalog_node_follows_the_meeus_sun_and_not_the_shadow_sun():
    import math

    from space_env.catalog.geometry import raan_rad_for_ltdn
    from space_env.core.sun import sun_iau2000_m, sun_meeus_m
    from space_env.core.time import Time

    moment = Time("2026-10-20T10:00:00Z")

    def node(vector: np.ndarray) -> float:
        sun_ra = math.atan2(float(vector[1]), float(vector[0]))
        return sun_ra + (10.5 - 12.0) * math.pi / 12.0 - math.pi

    def wrap(delta: float) -> float:
        return (delta + math.pi) % (2.0 * math.pi) - math.pi

    got = raan_rad_for_ltdn(moment, 10.5)
    assert abs(wrap(got - node(sun_meeus_m(moment)))) < 1e-12
    assert abs(wrap(got - node(sun_iau2000_m(moment)))) > 1e-3


def test_mean_sentinel_steps_near_786_km():
    from space_env.catalog.build import make_engine

    engine = make_engine("sentinel-2a", with_stations=False)
    state, _events = engine.step(30.0)
    assert state.altitude_km == pytest.approx(786.0, abs=40.0)
    assert {station.name for station in engine.stations} == set()


def test_sentinel_uses_published_mass_power_and_swath():
    spec = reference("sentinel-2a")
    craft = spec.spacecraft_config()
    assert craft.propulsion.mass_kg == 1140.0
    assert craft.power.consumption_nominal_W == 1400.0
    assert craft.power.battery_capacity_Wh == pytest.approx(102.0 * 28.0)
    assert craft.comms.data_rate_max_bps == pytest.approx(560e6)
    assert craft.payload.half_fov_deg == pytest.approx(half_swath_deg(786_000.0, 290_000.0))
    assert 10.0 < craft.payload.half_fov_deg < 12.0
    expected_dv = rocket_delta_v_m_s(1140.0, 123.0, ASSUMED_HYDRAZINE_ISP_S)
    assert craft.propulsion.delta_v_total_m_s == pytest.approx(expected_dv)
    assert 200.0 < expected_dv < 300.0
    # Генерация в надире на 1 а.е. совпадает с опубликованными 2300 Вт BOL.
    assert craft.power.solar_panel_area_m2 * craft.power.panel_efficiency * 1361.0 == pytest.approx(2300.0, rel=0.01)


def test_descending_node_local_time_is_the_published_ltdn():
    orbit = mean_orbit("sentinel-2a", epoch="2026-06-21T00:00:00Z", nu_deg=180.0)
    sun = sun_position_m(orbit.epoch)
    sun_ra = np.arctan2(sun[1], sun[0])
    node_ra = np.arctan2(orbit.r[1], orbit.r[0])
    delta = (node_ra - sun_ra + np.pi) % (2.0 * np.pi) - np.pi
    local_hours = (12.0 + delta * 12.0 / np.pi) % 24.0
    assert local_hours == pytest.approx(10.5, abs=0.05)
    assert abs(orbit.r[2]) < 1.0e3


def test_xband_margin_is_zero_at_the_mask_and_positive_at_zenith():
    spec = reference("sentinel-2a")
    edge = slant_range_m(spec.altitude_m, 5.0)
    zenith = spec.altitude_m
    common = dict(
        frequency_hz=spec.downlink_hz,
        tx_power_w=spec.tx_power_w,
        tx_gain_dbi=spec.tx_gain_dbi,
        rx_gain_dbi=spec.rx_gain_dbi,
        system_loss_db=3.0,
        required_power_dbw=spec.required_power_dbw,
    )
    assert link_margin_db(edge, **common) == pytest.approx(0.0, abs=0.05)
    assert link_margin_db(zenith, **common) > 5.0
    assert KIRUNA.alt_m == pytest.approx(402.2)
    assert KIRUNA.lat_deg == pytest.approx(67.8571, abs=1e-3)


def test_sentinel_pair_is_opposite_in_one_plane():
    fleet = sentinel2_design_fleet()
    first = fleet.vehicles["Sentinel-2A"].orbit.r
    second = fleet.vehicles["Sentinel-2B"].orbit.r
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    assert cosine == pytest.approx(-1.0, abs=1e-6)
    assert fleet.vehicles["Sentinel-2A"].spacecraft.config.propulsion.mass_kg == 1140.0


def test_tle_snapshot_sits_near_the_mission_altitude():
    pytest.importorskip("sgp4")
    for key, low, high in (("sentinel-2a", 760.0, 820.0), ("sentinel-2b", 760.0, 820.0), ("landsat-9", 680.0, 740.0)):
        line1, line2 = read_tle(key)
        assert line1.startswith("1 ")
        assert int(line2.split()[1]) == reference(key).norad_id
        orbit = tle_orbit(key)
        altitude_km = (float(np.linalg.norm(orbit.r)) - R_EARTH_MEAN) / 1000.0
        assert low < altitude_km < high


def test_iridium_star_spans_half_the_equator_and_the_neighbor_link_clears_earth():
    spec = reference("iridium-next")
    assert spec.crosslink_hz == pytest.approx(23.05e9)
    assert any("ЭИИМ кросс-линка не опубликован" in line for line in spec.assumptions)
    a_m = R_EARTH_MEAN + spec.altitude_m
    fleet = Fleet.walker_star(
        12,
        6,
        2,
        a_m=a_m,
        inc_deg=spec.inclination_deg,
        scenario=ScenarioConfig(ground_stations=[]),
        spacecraft=spec.spacecraft_config(),
    )
    raans = sorted(float(np.rad2deg(engine.orbit.keplerian().raan) % 360.0) for engine in fleet.vehicles.values())
    assert max(raans) < 180.0
    assert raans[-1] - raans[0] == pytest.approx(150.0, abs=1.0)
    plane = Fleet.walker_star(
        11,
        1,
        0,
        a_m=a_m,
        inc_deg=spec.inclination_deg,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    expected = 2.0 * a_m * np.sin(np.deg2rad(0.5 * 360.0 / 11.0))
    neighbor = min(plane.pairs(), key=lambda pair: abs(pair.range_m - expected))
    assert neighbor.range_m == pytest.approx(expected, rel=0.02)
    leader = plane.vehicles[neighbor.leader]
    follower = plane.vehicles[neighbor.follower]
    assert earth_occults(leader.orbit.r, follower.orbit.r) is False
    assert neighbor.range_m > 3.0e6
