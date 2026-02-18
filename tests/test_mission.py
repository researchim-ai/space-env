"""Тесты миссий: цели, ограничения, события, сценарии."""
import pytest

from space_env.mission import (
    Constraints,
    EventKind,
    MissionGoal,
    MissionEvent,
    MissionProfile,
    MissionType,
    ScenarioConfig,
    ScenarioType,
    AnomalyInject,
    ConjunctionInject,
)


def test_mission_goal_check():
    g = MissionGoal(imaging_count=10, comms_sessions_min=2, delta_v_budget_m_s=50.0)
    r = g.check(imaging_count=12, comms_sessions=3, delta_v_used_m_s=40.0)
    assert r["imaging"] is True
    assert r["comms_sessions"] is True
    assert r["delta_v_within_budget"] is True


def test_mission_event():
    e = MissionEvent(kind=EventKind.ANOMALY, time_sec=100.0, payload={"subsystem": "power"})
    assert e.kind == EventKind.ANOMALY
    assert e.time_sec == 100.0
    assert e.payload["subsystem"] == "power"


def test_scenario_config_nominal():
    s = ScenarioConfig(scenario_type=ScenarioType.NOMINAL)
    assert s.anomaly_injects == []
    assert s.conjunction_injects == []


def test_scenario_config_with_injects():
    s = ScenarioConfig(
        scenario_type=ScenarioType.ANOMALY,
        anomaly_injects=[AnomalyInject(subsystem="power", at_sec=60.0, severity="degraded")],
        conjunction_injects=[ConjunctionInject(at_sec=120.0, miss_distance_km=5.0)],
        seed=42,
    )
    assert len(s.anomaly_injects) == 1
    assert s.anomaly_injects[0].subsystem == "power"
    assert len(s.conjunction_injects) == 1
    assert s.conjunction_injects[0].miss_distance_km == 5.0
    assert s.seed == 42
