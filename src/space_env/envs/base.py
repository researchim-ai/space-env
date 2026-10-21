"""
Базовая обёртка и сборка типового LEO для RL-сред.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.orbit import Orbit
from space_env.core.spacecraft import (
    PowerConfig,
    PropulsionConfig,
    Spacecraft,
    SpacecraftConfig,
    ThermalConfig,
)
from space_env.core.time import Time
from space_env.mission.constraints import Constraints
from space_env.mission.goals import MissionGoal
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine


def make_box_obs(shape: tuple[int, ...], low: float = 0.0, high: float = 1.0) -> gym.Space:
    return gym.spaces.Box(
        low=np.float32(low),
        high=np.float32(high),
        shape=shape,
        dtype=np.float32,
    )


def normalize_altitude_km(alt_km: float, alt_min: float = 200.0, alt_max: float = 1000.0) -> float:
    return float(np.clip((alt_km - alt_min) / (alt_max - alt_min), 0.0, 1.0))


def normalize_temp(temp_C: float, t_min: float = -20.0, t_max: float = 50.0) -> float:
    return float(np.clip((temp_C - t_min) / (t_max - t_min), 0.0, 1.0))


def make_leo_engine(
    *,
    epoch: str = "2024-01-01T00:00:00Z",
    a_m: float = 6800e3,
    inc_deg: float = 51.6,
    seed: int | None = None,
    power: PowerConfig | None = None,
    thermal: ThermalConfig | None = None,
    propulsion: PropulsionConfig | None = None,
    mission_goal: MissionGoal | None = None,
    constraints: Constraints | None = None,
    scenario: ScenarioConfig | None = None,
    initial_temp_c: float | None = None,
    nu_rad: float = 0.0,
    name: str = "SC",
) -> SimulationEngine:
    """Круговая LEO и КА с заданными подсистемами."""
    t0 = Time(epoch)
    orbit = Orbit.from_keplerian(a_m, 0.0, np.deg2rad(inc_deg), 0.0, 0.0, nu_rad, t0)
    spacecraft = Spacecraft(
        config=SpacecraftConfig(
            name=name,
            power=power or PowerConfig(),
            thermal=thermal or ThermalConfig(),
            propulsion=propulsion or PropulsionConfig(),
        )
    )
    spacecraft.reset_state()
    if initial_temp_c is not None:
        spacecraft.state.thermal.temp_C = initial_temp_c
    return SimulationEngine(
        orbit,
        spacecraft,
        mission_goal=mission_goal or MissionGoal(),
        constraints=constraints or Constraints(),
        scenario=scenario or ScenarioConfig(),
        seed=seed,
    )


def format_ansi(title: str, rows: list[tuple[str, str]]) -> str:
    """Текстовый кадр render_mode='ansi'."""
    lines = [title, "-" * len(title)]
    for key, value in rows:
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
