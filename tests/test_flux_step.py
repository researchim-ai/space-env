"""Поток батареи усредняется по отсчётам шага, а не берётся в его конце."""
import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.core.environment import shadow_state
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PowerConfig, PropulsionConfig, Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-06-15T00:00:00Z")
RADIUS = 6_800_000.0


def _craft(position: np.ndarray, velocity: np.ndarray, *, pointing: str = "sun") -> SimulationEngine:
    orbit = Orbit.from_vectors(position, velocity, EPOCH)
    craft = Spacecraft(
        config=SpacecraftConfig(
            power=PowerConfig(
                solar_panel_area_m2=2.0,
                battery_capacity_Wh=100.0,
                consumption_nominal_W=50.0,
            ),
            propulsion=PropulsionConfig(delta_v_total_m_s=100.0, mass_kg=50.0),
        )
    )
    craft.reset_state()
    engine = SimulationEngine(
        orbit,
        craft,
        scenario=ScenarioConfig(ground_stations=[], initial_battery_soc=0.5),
    )
    engine.spacecraft.state.pointing = pointing
    return engine


def _sun_basis() -> tuple[np.ndarray, np.ndarray]:
    sun = sun_position_m(EPOCH)
    forward = sun / np.linalg.norm(sun)
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, forward))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    limb = np.cross(forward, axis)
    limb = limb / np.linalg.norm(limb)
    return forward, limb


def _place(alpha: float) -> np.ndarray:
    forward, limb = _sun_basis()
    return -forward * RADIUS * np.cos(alpha) + limb * RADIUS * np.sin(alpha)


def test_a_shadow_entry_keeps_the_lit_part_of_the_step():
    forward, limb = _sun_basis()
    lit_alpha = None
    for alpha in np.linspace(0.0, np.pi / 2.0, 181):
        if shadow_state(_place(float(alpha)), sun_position_m(EPOCH)).kind != "umbra":
            lit_alpha = float(alpha)
            break
    assert lit_alpha is not None and lit_alpha > 0.0
    start = _place(lit_alpha)
    assert shadow_state(start, sun_position_m(EPOCH)).kind != "umbra"
    speed = np.sqrt(MU_EARTH / RADIUS)
    # Уменьшение угла ведёт к антисолнечной оси и внутрь конуса тени.
    into_shadow = -(forward * np.sin(lit_alpha) + limb * np.cos(lit_alpha))
    into_shadow = into_shadow / np.linalg.norm(into_shadow)
    crossing = _craft(start, into_shadow * speed)
    dark = _craft(-forward * RADIUS, limb * speed)
    lit = _craft(forward * RADIUS, limb * speed)
    dark.spacecraft.state.pointing = "sun"
    lit.spacecraft.state.pointing = "sun"
    crossing.step(20.0)
    dark.step(20.0)
    lit.step(20.0)
    assert crossing.get_state_snapshot().in_eclipse is True
    assert dark.get_state_snapshot().battery_soc < crossing.get_state_snapshot().battery_soc
    assert crossing.get_state_snapshot().battery_soc < lit.get_state_snapshot().battery_soc


def test_nadir_panels_use_the_cosine_along_the_step():
    forward, limb = _sun_basis()
    start_angle = np.deg2rad(25.0)
    start = forward * RADIUS * np.cos(start_angle) + limb * RADIUS * np.sin(start_angle)
    speed = np.sqrt(MU_EARTH / RADIUS) * 40.0
    moving = _craft(start, limb * speed, pointing="nadir")
    parked = _craft(start, limb * 1.0, pointing="nadir")
    edge = _craft(limb * RADIUS, forward * 1.0, pointing="nadir")
    moving.step(20.0)
    parked.step(20.0)
    edge.step(20.0)
    assert parked.get_state_snapshot().in_eclipse is False
    assert edge.get_state_snapshot().battery_soc < moving.get_state_snapshot().battery_soc
    assert moving.get_state_snapshot().battery_soc < parked.get_state_snapshot().battery_soc
