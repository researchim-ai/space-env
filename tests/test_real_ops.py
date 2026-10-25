"""Солнечно-синхронный узел, память Sentinel-2 и путь по сетке Iridium."""
import math

import gymnasium as gym
import numpy as np
import pytest

import space_env.envs

from space_env.catalog.geometry import (
    MEAN_SOLAR_RATE_RAD_S,
    j2_nodal_rate_rad_s,
    mean_semimajor_axis_m,
    sun_sync_inclination_rad,
)
from space_env.catalog.mesh import (
    circular_position_m,
    designed_neighbors,
    iridium_shell,
    link_open,
    route,
    shell_slots,
)
from space_env.catalog.traffic import dump_bits, store_bits
from space_env.core.constants import R_EARTH_MEAN
from space_env.core.link import earth_occults


def test_published_inclinations_match_sun_synchronous_j2():
    sentinel = sun_sync_inclination_rad(mean_semimajor_axis_m(786_000.0))
    landsat = sun_sync_inclination_rad(mean_semimajor_axis_m(705_000.0))
    assert math.degrees(sentinel) == pytest.approx(98.62, abs=0.25)
    assert math.degrees(landsat) == pytest.approx(98.2, abs=0.4)
    iridium_a = mean_semimajor_axis_m(780_000.0)
    assert j2_nodal_rate_rad_s(iridium_a, math.radians(86.4)) < 0.0
    assert j2_nodal_rate_rad_s(mean_semimajor_axis_m(786_000.0), sentinel) == pytest.approx(
        MEAN_SOLAR_RATE_RAD_S, rel=1e-6
    )


def test_scene_store_respects_memory_and_the_forty_minute_cap():
    buffer, used, added = store_bits(0.0, 60.0, rate_bps=450e6, memory_bits=2.4e12, imaging_sec=0.0, imaging_cap_sec=2400.0)
    assert added == pytest.approx(450e6 * 60.0)
    assert buffer == pytest.approx(added)
    assert used == pytest.approx(60.0)
    full, _used, extra = store_bits(2.4e12, 60.0, rate_bps=450e6, memory_bits=2.4e12, imaging_sec=0.0, imaging_cap_sec=2400.0)
    assert extra == 0.0
    assert full == pytest.approx(2.4e12)
    _buffer, capped, added_late = store_bits(
        0.0, 120.0, rate_bps=450e6, memory_bits=2.4e12, imaging_sec=2390.0, imaging_cap_sec=2400.0
    )
    assert capped == pytest.approx(2400.0)
    assert added_late == pytest.approx(450e6 * 10.0)
    left, moved = dump_bits(added, 2.0, 560e6)
    assert moved == pytest.approx(560e6 * 2.0)
    assert left == pytest.approx(added - moved)


def test_iridium_in_plane_neighbor_is_open_and_five_hops_reach_the_far_slot():
    slots, a_m, inclination = iridium_shell()
    assert len(slots) == 66
    assert len(shell_slots(66, 6, 2)) == 66
    start = slots[0]
    fore = designed_neighbors(start, 66, 6, 2)["fore"]
    assert fore.plane == 0 and fore.slot == 1
    assert link_open(start, fore, a_m, inclination) is True
    separation = 2.0 * a_m * math.sin(math.radians(0.5 * 360.0 / 11.0))
    left = circular_position_m(a_m, inclination, start.raan_deg, start.nu_deg)
    right = circular_position_m(a_m, inclination, fore.raan_deg, fore.nu_deg)
    assert float(np.linalg.norm(right - left)) == pytest.approx(separation, rel=0.01)
    assert earth_occults(left, right) is False
    # Лимб близко, но хорда его не касается: высота пролёта луча больше радиуса плюс 80 км.
    assert float(np.linalg.norm(left)) > R_EARTH_MEAN
    goal = next(item for item in slots if item.plane == 0 and item.slot == 5)
    path = route(start, goal, a_m=a_m, inclination_deg=inclination, total=66, planes=6, phasing=2)
    assert path is not None
    assert len(path) <= 5
    assert "fore" in path


def test_sentinel_acquisition_stores_a_scene_on_the_descending_node():
    env = gym.make(
        "space_env/SentinelAcquisition-v0",
        max_episode_steps=2,
        with_stations=False,
        start_at_pass=False,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (7,)
    assert info["targets"][0] == "S2-1"
    assert env.unwrapped._engine.get_state_snapshot().in_eclipse is False
    _obs, reward, terminated, _trunc, step_info = env.step(1)
    assert step_info["imaging_count"] == 1
    assert 0.0 < step_info["stored_bits"] <= 450e6 * 60.0 + 1.0
    assert step_info["stored_bits"] == pytest.approx(env.unwrapped._engine.produced_bits)
    assert step_info["delivered_bits"] == 0.0
    assert reward == 0.0
    assert terminated is False
    env.close()


def test_sentinel_dump_is_rewarded_only_inside_the_pass():
    env = gym.make("space_env/SentinelAcquisition-v0", max_episode_steps=12, step_sec=60.0)
    env.reset(seed=0)
    env.unwrapped._engine.preload_bits(2.0e11)
    delivered = 0.0
    reward = 0.0
    obs, _reward, _term, _trunc, _info = env.step(0)
    for _ in range(8):
        if obs[2] > 0.5:
            _obs, reward, _term, _trunc, info = env.step(2)
            delivered = info["delivered_bits"]
            break
        obs, _reward, _term, _trunc, _info = env.step(0)
    assert delivered > 0.0
    assert reward > 0.0
    env.close()


def test_landsat_recorder_is_the_handbook_figure():
    env = gym.make(
        "space_env/LandsatAcquisition-v0",
        max_episode_steps=2,
        with_stations=False,
        start_at_pass=False,
    )
    env.reset(seed=0)
    _obs, reward, _term, _trunc, info = env.step(1)
    assert env.unwrapped._memory_bits == pytest.approx(3.14e12)
    assert 0.0 < info["stored_bits"] <= 384e6 * 60.0 + 1.0
    assert info["stored_bits"] == pytest.approx(env.unwrapped._engine.produced_bits)
    assert reward == 0.0
    env.close()


def test_retrograde_burns_stop_the_sentinel_pair_drift():
    env = gym.make("space_env/SentinelPhasing-v0", max_episode_steps=12, step_sec=300.0)
    env.reset(seed=0)
    env.step(0)
    assert abs(env.unwrapped._rate_m_s) > 0.3
    fresh = gym.make("space_env/SentinelPhasing-v0", max_episode_steps=12, step_sec=300.0)
    fresh.reset(seed=0)
    fresh.step(2)
    fresh.step(0)
    assert abs(fresh.unwrapped._rate_m_s) < 0.05
    assert abs(fresh.unwrapped._error_m) < 50.0
    fresh.close()
    env.close()


def test_tle_bstar_does_not_replace_the_sentinel_face():
    from space_env.catalog.drag import drag_comparison

    sentinel = drag_comparison("sentinel-2a")
    assert sentinel.bstar_per_er == pytest.approx(1.7214e-5, rel=1e-3)
    assert sentinel.geometric_over_tle > 10.0
    landsat = drag_comparison("landsat-9")
    assert landsat.tle_cd_area_per_kg > 0.0
    assert 0.3 < landsat.geometric_cd_area_per_kg / landsat.tle_cd_area_per_kg < 3.0


def test_iridium_routes_to_a_slot_above_fairbanks():
    from space_env.catalog.mesh import circular_position_m, iridium_shell, route_to_gateway, slots_above
    from space_env.core.frames import elevation_deg
    from space_env.core.ground import FAIRBANKS
    from space_env.core.time import Time

    slots, radius_m, inclination = iridium_shell()
    epoch = Time("2026-08-01T00:00:00Z")
    visible = slots_above(
        slots, radius_m, inclination, epoch, FAIRBANKS.lat_deg, FAIRBANKS.lon_deg, mask_deg=5.0
    )
    assert visible
    routed = route_to_gateway(
        slots[0],
        FAIRBANKS.lat_deg,
        FAIRBANKS.lon_deg,
        a_m=radius_m,
        inclination_deg=inclination,
        epoch=epoch,
        total=66,
        planes=6,
        phasing=2,
    )
    assert routed is not None
    goal, path = routed
    assert goal in visible or (goal.plane, goal.slot) in {(item.plane, item.slot) for item in visible}
    position = circular_position_m(radius_m, inclination, goal.raan_deg, goal.nu_deg)
    assert elevation_deg(position, FAIRBANKS.lat_deg, FAIRBANKS.lon_deg, 0.0, epoch) >= 5.0
    assert len(path) <= 5


def test_operations_sft_mentions_hops_and_bstar():
    from space_env.data_generators.sft import generate_sft_from_operations

    records = list(generate_sft_from_operations())
    text = " ".join(item["instruction"] + item["output"] for item in records)
    assert "скачков" in text
    assert "B*" in text
    assert "2.4" in text
