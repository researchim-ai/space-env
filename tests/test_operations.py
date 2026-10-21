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
