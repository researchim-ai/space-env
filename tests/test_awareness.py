"""Истина, измерение и безопасность не смешиваются."""
import numpy as np

from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PropulsionConfig, Spacecraft, SpacecraftConfig, SubsystemHealth
from space_env.core.time import Time
from space_env.mission.awareness import SensorModel, SensorStream, separate_scores
from space_env.mission.scenarios import AnomalyInject, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

_EPOCH = "2026-10-20T10:00:00Z"


def _world() -> MissionWorld:
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.9, 0.0, 0.0, 0.0, Time(_EPOCH))
    craft = Spacecraft(config=SpacecraftConfig(name="a", propulsion=PropulsionConfig(delta_v_total_m_s=20.0)))
    craft.reset_state()
    engine = SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[], anomaly_injects=[
        AnomalyInject(subsystem="payload", at_sec=0.0, severity="failed"),
    ]))
    return MissionWorld({"a": engine}, epoch=_EPOCH, horizon_sec=100.0, seed=3)


def test_a_hidden_failure_stays_out_of_the_measurement():
    world = _world()
    world.step(5.0, {"a": "wait"})
    health = world.engines["a"].spacecraft.state.health["payload"]
    assert health == SubsystemHealth.FAILED
    packet = world.measure("a", SensorStream(SensorModel(), np.random.default_rng(1)))
    assert packet["missing"] is False
    assert "health" not in packet
    assert "subsystem" not in packet
    assert "severity" not in packet
    assert set(packet) == {"kind", "missing", "stale", "time_sec", "soc", "stored_bits", "in_contact", "station"}


def test_delay_period_and_dropout_follow_the_sensor_and_the_seed():
    tape = [
        {"time_sec": 0.0, "soc": 0.9, "stored_bits": 10.0, "in_contact": False, "station": ""},
        {"time_sec": 10.0, "soc": 0.2, "stored_bits": 10.0, "in_contact": True, "station": "Kiruna"},
    ]
    delayed = SensorStream(SensorModel(delay_sec=10.0), np.random.default_rng(1))
    assert delayed.read(10.0, tape)["soc"] == 0.9
    held = SensorStream(SensorModel(period_sec=10.0, bias_soc=0.05), np.random.default_rng(1))
    first = held.read(0.0, tape)
    second = held.read(4.0, tape)
    assert second["stale"] is True
    assert second["soc"] == first["soc"]
    quiet = SensorStream(SensorModel(dropout=1.0), np.random.default_rng(1))
    assert quiet.read(10.0, tape)["reason"] == "пропуск"
    left = SensorStream(SensorModel(sigma_soc=0.02), np.random.default_rng(7))
    right = SensorStream(SensorModel(sigma_soc=0.02), np.random.default_rng(7))
    assert left.read(10.0, tape)["soc"] == right.read(10.0, tape)["soc"]


def test_safety_is_not_added_to_delivery():
    world = _world()
    world.engines["a"].spacecraft.state.power.battery_soc = 0.05
    scores = separate_scores(on_time_fraction=1.0, safety_events=world.safety_events())
    assert scores == {"delivery": 1.0, "safety": 1}
    assert "combined" not in scores
