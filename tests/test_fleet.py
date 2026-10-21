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


def test_constellation_env_steps_every_vehicle():
    env = gym.make("space_env/Constellation-v0", max_episode_steps=2)
    obs, info = env.reset(seed=0)
    assert obs.shape == (2 * 4 + 1,)
    assert info["vehicles"] == ["SC-1", "SC-2"]
    obs2, _reward, _term, _trunc, info2 = env.step(np.array([1, 0]))
    assert obs2.shape == obs.shape
    assert info2["commanded"] == "SC-2"
    env.close()
