"""Короткий замер сборки и одного шага. Часы кампании сюда не подставляются."""
from __future__ import annotations

import resource
import time

import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld


def load_report(vehicles: int, requests: int, *, step_sec: float = 1.0) -> dict:
    """Собирает N аппаратов без станций, делает один шаг и пишет время стены."""
    if vehicles < 1 or requests < 0:
        raise ValueError("нужен хотя бы один аппарат")
    started = time.perf_counter()
    epoch = Time("2024-01-01T00:00:00Z")
    engines = {}
    for index in range(int(vehicles)):
        radius = np.array([7.0e6, 0.0, 0.0])
        speed = np.sqrt(MU_EARTH / 7.0e6)
        phase = np.deg2rad(index * 0.2)
        position = np.array([radius[0] * np.cos(phase), radius[0] * np.sin(phase), 0.0])
        velocity = np.array([-speed * np.sin(phase), speed * np.cos(phase), 0.0])
        craft = Spacecraft(config=SpacecraftConfig(name=f"V{index}"))
        craft.reset_state()
        engines[f"V{index}"] = SimulationEngine(
            Orbit.from_vectors(position, velocity, epoch),
            craft,
            scenario=ScenarioConfig(ground_stations=[]),
        )
    world = MissionWorld(engines, epoch="2024-01-01T00:00:00Z", horizon_sec=max(step_sec, 1.0) + 1.0)
    world.step(step_sec)
    return {
        "vehicles": int(vehicles),
        "requests": int(requests),
        "request_ids": [f"load-{index}" for index in range(int(requests))],
        "served_requests": 0,
        "outcome": "propagation_only",
        "wall_sec": time.perf_counter() - started,
        "time_sec": float(world.now_sec()),
        "rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ladder": "recorded_short_step",
    }


def simulate_load(
    vehicles: int,
    requests: int,
    *,
    duration_sec: float,
    step_sec: float = 30.0,
) -> dict:
    """Крутит N аппаратов кусками step_sec, пока не наберётся duration_sec. Сутки сюда не подставляются."""
    if duration_sec <= 0.0 or step_sec <= 0.0:
        raise ValueError("длительность и шаг должны быть положительными")
    started = time.perf_counter()
    epoch = Time("2024-01-01T00:00:00Z")
    engines = {}
    for index in range(int(vehicles)):
        radius = 7.0e6
        speed = np.sqrt(MU_EARTH / radius)
        phase = np.deg2rad(index * 0.2)
        position = np.array([radius * np.cos(phase), radius * np.sin(phase), 0.0])
        velocity = np.array([-speed * np.sin(phase), speed * np.cos(phase), 0.0])
        craft = Spacecraft(config=SpacecraftConfig(name=f"V{index}"))
        craft.reset_state()
        engines[f"V{index}"] = SimulationEngine(
            Orbit.from_vectors(position, velocity, epoch),
            craft,
            scenario=ScenarioConfig(ground_stations=[]),
        )
    world = MissionWorld(engines, epoch="2024-01-01T00:00:00Z", horizon_sec=float(duration_sec) + 1.0)
    steps = 0
    while world.now_sec() + 1e-9 < duration_sec:
        dt = min(float(step_sec), float(duration_sec) - world.now_sec())
        world.step(dt)
        steps += 1
    return {
        "vehicles": int(vehicles),
        "requests": int(requests),
        "served_requests": 0,
        "outcome": "propagation_only",
        "steps": steps,
        "simulated_sec": float(world.now_sec()),
        "wall_sec": time.perf_counter() - started,
        "rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ladder": "multi_step_propagation",
    }


def ladder_report(rungs: tuple[tuple[int, int, float], ...] = ((16, 100, 1.0),)) -> list[dict]:
    """Каждая ступень — отдельный короткий шаг. 24 и 72 часа этим вызовом не исполняются."""
    rows = []
    for index, (vehicles, requests, step_sec) in enumerate(rungs):
        row = load_report(vehicles, requests, step_sec=step_sec)
        row["rung"] = index
        rows.append(row)
    return rows
