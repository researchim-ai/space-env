"""Scheduled idle attitude uses real slew time and battery dynamics."""
import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PowerConfig, Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.coast import coast_to_job
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine


def engine():
    epoch = Time("2024-01-01T00:00:00Z")
    sun = sun_position_m(epoch)
    sun /= np.linalg.norm(sun)
    direction = np.cross(sun, [0., 0., 1.])
    direction /= np.linalg.norm(direction)
    craft = Spacecraft(config=SpacecraftConfig(power=PowerConfig(
        solar_panel_area_m2=1, battery_capacity_Wh=100,
        consumption_nominal_W=250, consumption_peak_W=300,
    )))
    craft.reset_state()
    craft.state.power.battery_soc = .6
    return SimulationEngine(
        Orbit.from_vectors(direction * 7e6, sun * np.sqrt(MU_EARTH / 7e6), epoch),
        craft, scenario=ScenarioConfig(ground_stations=[]),
    )


def test_idle_charging_returns_to_nadir_without_moving_job_time():
    charged, idle = engine(), engine()
    coast_to_job(charged, 600)
    for _ in range(10):
        idle.step(60, "wait")
    assert charged.time_sec == idle.time_sec == 600
    assert charged.spacecraft.state.pointing == "nadir"
    assert charged.get_state_snapshot().slew_remaining_deg < 1e-6
    assert charged.spacecraft.state.power.battery_soc > idle.spacecraft.state.power.battery_soc + .1
    assert any(snap.pointing == "sun" for _, snap, _ in charged.log)
    assert not charged.products()


def test_short_idle_does_not_start_a_slew_that_would_miss_the_job():
    craft = engine()
    coast_to_job(craft, 30)
    assert craft.time_sec == 30
    assert all(snap.pointing == "nadir" for _, snap, _ in craft.log)
    with pytest.raises(ValueError):
        coast_to_job(craft, 10)
