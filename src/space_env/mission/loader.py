"""Сборка движка из YAML-сценария в configs/scenarios."""
from __future__ import annotations

from pathlib import Path

import yaml

from space_env.core.ground import GroundStation
from space_env.core.spacecraft import PowerConfig, PropulsionConfig, ThermalConfig
from space_env.envs.base import make_leo_engine
from space_env.mission.constraints import Constraints
from space_env.mission.goals import MissionGoal
from space_env.mission.scenarios import ScenarioConfig, ScenarioType
from space_env.simulation.engine import SimulationEngine


def load_engine(path: str | Path, *, seed: int | None = None) -> SimulationEngine:
    """Читает сценарий и возвращает готовый SimulationEngine на круговой LEO."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    orbit = data.get("orbit", {})
    craft = data.get("spacecraft", {})
    goals = data.get("goals", {})
    limits = data.get("constraints", {})
    stations_raw = data.get("ground_stations")
    stations = None if stations_raw is None else [GroundStation(**item) for item in stations_raw]
    scenario = ScenarioConfig(
        scenario_type=ScenarioType(data.get("scenario_type", "nominal")),
        name=str(data.get("name", source.stem)),
        seed=seed,
        ground_stations=stations,
    )
    return make_leo_engine(
        epoch=str(data.get("epoch", "2024-01-01T00:00:00Z")),
        a_m=float(orbit.get("a_m", 6800e3)),
        inc_deg=float(orbit.get("inc_deg", 51.6)),
        seed=seed,
        power=PowerConfig(
            solar_panel_area_m2=float(craft.get("solar_panel_area_m2", 2.0)),
            battery_capacity_Wh=float(craft.get("battery_capacity_Wh", 200.0)),
            consumption_nominal_W=float(craft.get("consumption_nominal_W", 15.0)),
            consumption_peak_W=float(craft.get("consumption_peak_W", 40.0)),
        ),
        thermal=ThermalConfig(),
        propulsion=PropulsionConfig(
            delta_v_total_m_s=float(craft.get("delta_v_total_m_s", 50.0)),
            mass_kg=float(craft.get("mass_kg", 50.0)),
        ),
        mission_goal=MissionGoal(**goals) if goals else MissionGoal(),
        constraints=Constraints(**limits) if limits else Constraints(),
        scenario=scenario,
    )
