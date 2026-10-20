"""Тесты движка симуляции."""
import numpy as np
import pytest

from space_env.core import Orbit, Time
from space_env.core.spacecraft import (
    PowerConfig,
    PropulsionConfig,
    Spacecraft,
    SpacecraftConfig,
    SubsystemHealth,
)
from space_env.mission import EventKind, MissionGoal, ScenarioConfig, ScenarioType, AnomalyInject
from space_env.simulation import SimulationEngine, SimulationState


def test_engine_nominal_run():
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    sc = Spacecraft(config=SpacecraftConfig(
        power=PowerConfig(solar_panel_area_m2=2.0, battery_capacity_Wh=200.0, consumption_nominal_W=10.0, consumption_peak_W=30.0),
        propulsion=PropulsionConfig(delta_v_total_m_s=100.0, mass_kg=50.0),
    ))
    sc.reset_state()
    engine = SimulationEngine(orbit, sc, seed=42)
    assert engine.time_sec == 0.0
    assert len(engine.log) == 1
    snap0 = engine.log[0][1]
    assert snap0.battery_soc == 1.0
    snap1, events1 = engine.step(60.0)
    assert engine.time_sec == 60.0
    assert len(engine.log) == 2
    assert isinstance(snap1, SimulationState)
    assert snap1.time_sec == 60.0
    kinds = {event.kind for event in events1}
    assert EventKind.ANOMALY not in kinds
    assert EventKind.CONJUNCTION_WARNING not in kinds


def test_engine_anomaly_inject():
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, 0.0, 0.0, 0.0, 0.0, epoch)
    sc = Spacecraft(config=SpacecraftConfig(
        power=PowerConfig(solar_panel_area_m2=1.0, battery_capacity_Wh=100.0, consumption_nominal_W=5.0, consumption_peak_W=20.0),
        propulsion=PropulsionConfig(delta_v_total_m_s=50.0, mass_kg=10.0),
    ))
    sc.reset_state()
    scenario = ScenarioConfig(
        scenario_type=ScenarioType.ANOMALY,
        anomaly_injects=[AnomalyInject(subsystem="power", at_sec=30.0, severity="degraded")],
        seed=1,
    )
    engine = SimulationEngine(orbit, sc, scenario=scenario, seed=1)
    events_all = []
    for _ in range(40):
        snap, events = engine.step(1.0)
        events_all.extend(events)
    assert any(e.kind.value == "anomaly" for e in events_all)
    assert sc.state.health["power"] == SubsystemHealth.DEGRADED
