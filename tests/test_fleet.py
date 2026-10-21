"""Группировка: общая шкала времени, дистанция в цепочке, скрининг, изоляция манёвра."""
import numpy as np
import pytest

import gymnasium as gym

from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet


def _quiet() -> ScenarioConfig:
    return ScenarioConfig(ground_stations=[])


def test_string_of_pearls_range_matches_chord():
    a_m = 6_800_000.0
    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, a_m=a_m, scenario=_quiet())
    pair = fleet.pairs()[0]
    expected = 2.0 * a_m * np.sin(np.deg2rad(0.5))
    assert pair.range_m == pytest.approx(expected, rel=0.02)
    assert abs(pair.along_m) > abs(pair.radial_m)


def test_fleet_step_keeps_one_clock_and_isolates_maneuver():
    fleet = Fleet.string_of_pearls(2, separation_deg=5.0, scenario=_quiet())
    before = fleet.vehicles["SC-2"].spacecraft.state.propulsion.delta_v_remaining_m_s
    fleet.step(30.0, {"SC-1": "maneuver"}, {"SC-1": {"dv_m_s": 1.0, "direction": "prograde"}})
    assert fleet.time_sec == pytest.approx(30.0)
    assert fleet.vehicles["SC-1"].time_sec == fleet.vehicles["SC-2"].time_sec
    assert fleet.vehicles["SC-1"].last_command_ok is True
    assert fleet.vehicles["SC-2"].spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(before)


def test_screen_only_reports_the_close_pair():
    close = Fleet.string_of_pearls(2, separation_deg=0.02, scenario=_quiet())
    far = Fleet.string_of_pearls(2, separation_deg=10.0, scenario=_quiet())
    close_hits = close.screen(range_limit_m=50_000.0)
    far_hits = far.screen(range_limit_m=50_000.0)
    assert len(close_hits) == 1
    assert close_hits[0].probability > 0.0
    assert far_hits == []


def test_walker_splits_nodes_between_two_planes():
    fleet = Fleet.walker(6, 2, 1, scenario=_quiet())
    assert set(fleet.vehicles) == {f"P{plane}-S{slot}" for plane in (1, 2) for slot in (1, 2, 3)}
    raans = [float(np.rad2deg(engine.orbit.keplerian().raan) % 360.0) for engine in fleet.vehicles.values()]
    near_zero = [angle for angle in raans if angle < 20.0 or angle > 340.0]
    near_half = [angle for angle in raans if abs(angle - 180.0) < 20.0]
    assert len(near_zero) == 3
    assert len(near_half) == 3


def test_in_plane_link_is_clear_and_opposite_side_is_blocked():
    from space_env.core.link import earth_occults

    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=_quiet())
    pair = fleet.pairs()[0]
    assert pair.earth_blocked is False
    assert pair.isl_margin_db > 0.0
    assert earth_occults(np.array([7.0e6, 0.0, 0.0]), np.array([-7.0e6, 0.0, 0.0])) is True


def test_cdm_covariance_replaces_the_default_screen():
    from space_env.core.cdm import CdmObject, ConjunctionMessage

    fleet = Fleet.string_of_pearls(2, separation_deg=0.02, scenario=_quiet())
    baseline = fleet.screen()[0].probability
    message = ConjunctionMessage(miss_distance_m=250.0, hard_body_radius_m=5_000.0)
    message.objects = [CdmObject(covariance_km2=np.diag([0.04, 2.0, 0.04]))]
    fleet.use_cdm("SC-1", "SC-2", message)
    from_cdm = fleet.screen()[0].probability
    assert from_cdm > baseline * 5.0


def test_constellation_env_steps_every_vehicle():
    env = gym.make("space_env/Constellation-v0", max_episode_steps=2)
    obs, info = env.reset(seed=0)
    assert obs.shape == (2 * 5 + 2,)
    assert info["vehicles"] == ["SC-1", "SC-2"]
    obs2, _reward, _term, _trunc, info2 = env.step(np.array([1, 0]))
    assert obs2.shape == obs.shape
    assert info2["commanded"] == "SC-2"
    env.close()


def test_relay_moves_buffer_and_stops_when_earth_blocks_the_path():
    near = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=_quiet())
    near.collect("SC-1", 5.0)
    moved = near.relay("SC-1", "SC-2", 60.0)
    assert moved > 0.0
    assert near.buffer_mb["SC-2"] == pytest.approx(moved)
    assert near.buffer_mb["SC-1"] == pytest.approx(5.0 - moved)

    far = Fleet.string_of_pearls(2, separation_deg=180.0, scenario=_quiet())
    far.collect("SC-1", 5.0)
    assert far.relay("SC-1", "SC-2", 60.0) == 0.0
    assert far.buffer_mb["SC-2"] == 0.0


def test_downlink_drains_the_buffer_only_while_the_station_link_is_up():
    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=_quiet())
    fleet.collect("SC-1", 100.0)
    assert fleet.downlink("SC-1", 10.0) == 0.0
    engine = fleet.vehicles["SC-1"]
    engine._in_contact = True
    engine._link_margin_db = 10.0
    moved = fleet.downlink("SC-1", 10.0)
    assert moved == pytest.approx(10.0)
    assert fleet.delivered_mb == pytest.approx(moved)
    assert fleet.buffer_mb["SC-1"] == pytest.approx(90.0)


def test_cdm_with_velocity_is_mapped_from_tca_and_position_only_stays_put():
    from space_env.core.cdm import CdmObject, ConjunctionMessage
    from space_env.core.relative import cw_stm, mean_motion_rad_s

    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=_quiet())
    state = np.zeros((6, 6))
    state[4, 4] = 1.0
    message = ConjunctionMessage(tca="2023-12-31T23:50:00Z", miss_distance_m=250.0)
    message.objects = [CdmObject(state_covariance_si=state)]
    fleet.use_cdm("SC-1", "SC-2", message)
    mapped = fleet.pair_covariance("SC-1", "SC-2")
    primary = fleet.vehicles["SC-1"]
    motion = mean_motion_rad_s(float(np.linalg.norm(primary.orbit.r)))
    phi = cw_stm(motion, 600.0)
    expected = phi @ state @ phi.T
    assert mapped[1, 1] == pytest.approx(expected[1, 1], rel=1e-6)
    assert mapped[1, 1] > 1.0e4

    frozen = ConjunctionMessage(tca="2023-12-31T23:50:00Z", miss_distance_m=250.0)
    frozen.objects = [CdmObject(covariance_km2=np.diag([0.04, 2.0, 0.04]))]
    fleet.use_cdm("SC-1", "SC-2", frozen)
    held = fleet.pair_covariance("SC-1", "SC-2")
    assert held[1, 1] == pytest.approx(2.0e6, rel=1e-9)


def test_constellation_stores_a_frame_outside_umbra():
    env = gym.make("space_env/Constellation-v0", max_episode_steps=3)
    env.reset(seed=0)
    _obs, reward, _term, _trunc, info = env.step(np.array([0, 1]))
    snap = env.unwrapped._fleet.vehicles["SC-1"].get_state_snapshot()
    if snap.in_eclipse:
        assert info["buffer_mb"]["SC-1"] == 0.0
    else:
        assert info["buffer_mb"]["SC-1"] == pytest.approx(8.0)
        assert reward >= 0.2
        _obs, reward_relay, _term, _trunc, relayed = env.step(np.array([0, 4]))
        assert relayed["buffer_mb"]["SC-2"] > 0.0
        assert reward_relay > 0.0
    env.close()
