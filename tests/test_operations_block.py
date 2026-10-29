"""Скрытый отказ, уклонение в бюджете миссии и расписание компенсации торможения."""
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, SubsystemHealth
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.awareness import SensorModel, SensorStream
from space_env.mission.scenarios import AnomalyInject, ConjunctionInject, GroundTarget, ScenarioConfig
from space_env.mission.stationkeep import drag_makeup_schedule
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

EPOCH = Time("2024-01-01T00:00:00Z")


def _sun_engine(scenario: ScenarioConfig | None = None) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    axis = np.array([0.0, 0.0, 1.0])
    velocity = np.cross(radius, axis)
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=scenario or ScenarioConfig(ground_stations=[]),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    return engine


def _overhead(loss_inject: AnomalyInject | None) -> SimulationEngine:
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)
    position = ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    axis = np.array([0.0, 0.0, 1.0])
    direction = np.cross(position, axis)
    direction = direction / np.linalg.norm(direction)
    injects = [] if loss_inject is None else [loss_inject]
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(position, direction * speed, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[station], anomaly_injects=injects),
    )
    engine.accept_bits(20.0e6, origin="imaging", request_id="frame")
    return engine


def test_hidden_cause_cuts_the_rate_or_latches_safe_without_rewriting_the_past():
    cause = "tx_chain"
    fault = AnomalyInject(
        subsystem="comms",
        at_sec=6.0,
        severity="failed",
        symptom="rate",
        cause=cause,
        hidden=True,
    )
    late = _overhead(fault)
    clear = _overhead(None)
    late.step(5.0, "comms")
    clear.step(5.0, "comms")
    assert late.delivered_bits == pytest.approx(clear.delivered_bits)
    assert late.delivered_bits > 0.0
    late.step(5.0, "comms")
    clear.step(5.0, "comms")
    late_second = late.delivered_bits - clear.delivered_bits
    assert late.symptom_report()["rate_factor"] == pytest.approx(0.25)
    assert late_second < 0.0
    assert late.spacecraft.state.health["comms"] == SubsystemHealth.NOMINAL
    report = late.symptom_report()
    assert cause not in report
    assert "cause" not in report
    world = MissionWorld({"a": late}, epoch="2024-01-01T00:00:00Z", horizon_sec=30.0)
    world.note_truth()
    reading = world.measure("a", SensorStream(SensorModel(), np.random.default_rng(1)))
    assert cause not in reading
    assert "health" not in reading

    safe = _sun_engine(
        ScenarioConfig(
            ground_stations=[],
            anomaly_injects=[
                AnomalyInject(
                    subsystem="payload",
                    at_sec=0.0,
                    symptom="false_safe",
                    cause="heater_sense",
                    hidden=True,
                    recovery_sec=15.0,
                )
            ],
        )
    )
    safe.step(10.0, "imaging")
    assert safe._imaging_count == 0
    assert safe.symptom_report()["false_safe"] is True
    assert "heater_sense" not in safe.symptom_report()
    assert safe.spacecraft.state.power.battery_soc > 0.5
    assert safe.spacecraft.state.health["payload"] == SubsystemHealth.NOMINAL
    safe.step(10.0, "recover")
    assert safe.symptom_report()["false_safe"] is True
    safe.step(10.0, "wait")
    assert safe.symptom_report()["symptom"] == ""
    assert safe._safe_latched is False
    safe.step(10.0, "imaging")
    assert safe._imaging_count == 1


def test_avoidance_spends_mission_dv_or_leaves_the_encounter_budget_intact():
    def engine() -> SimulationEngine:
        return _sun_engine(
            ScenarioConfig(
                ground_stations=[],
                conjunction_injects=[
                    ConjunctionInject(
                        at_sec=0.0,
                        miss_distance_km=0.2,
                        miss_radial_m=200.0,
                        miss_cross_m=0.0,
                        tca_in_sec=120.0,
                        combined_sigma_m=50.0,
                        hard_body_radius_m=10.0,
                    )
                ],
            )
        )

    accepted = engine()
    refused = engine()
    waited = engine()
    for item in (accepted, refused, waited):
        item.step(1.0, "wait")
    assert accepted._message_miss_m == pytest.approx(200.0)
    payload = {
        "hard_body_radius_m": 10.0,
        "sigma_m": 50.0,
        "target_miss_m": 1_000.0,
        "dv_per_m": 0.001,
        "mission_reserve_m_s": 0.0,
    }
    accepted.step(1.0, "avoid", payload)
    assert accepted.last_command_ok is True
    assert accepted._avoid_plan["accepted"] is True
    assert accepted._avoid_plan["dv_m_s"] == pytest.approx(0.8)
    assert accepted._avoid_plan["pc_after"] < accepted._avoid_plan["pc_before"]
    assert accepted.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(79.2)
    refused.step(1.0, "avoid", {**payload, "mission_reserve_m_s": 79.5})
    assert refused.last_command_ok is False
    assert refused._avoid_reason == "бюджет миссии"
    assert refused.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(80.0)
    waited.step(1.0, "wait")
    accepted_miss = accepted._conjunction_view()[0]
    waited_miss = waited._conjunction_view()[0]
    assert abs(accepted_miss - waited_miss) * 1000.0 > 1.0
    accepted.step(1.0, "maneuver", {"dv_m_s": 79.5, "direction": "prograde"})
    assert accepted.last_command_ok is False
    assert accepted.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(79.2)
    refused.step(1.0, "maneuver", {"dv_m_s": 79.5, "direction": "prograde"})
    assert refused.last_command_ok is True
    assert refused.spacecraft.state.propulsion.delta_v_remaining_m_s == pytest.approx(0.5)


def test_drag_makeup_schedule_refuses_a_poor_budget_and_blocks_imaging_on_its_burn():
    poor = drag_makeup_schedule(
        200_000.0,
        mass_kg=10.0,
        area_m2=1.0,
        drag_coefficient=2.2,
        horizon_sec=86_400.0,
        burns=4,
        dv_budget_m_s=1.0,
        mission_reserve_m_s=0.0,
    )
    assert poor["accepted"] is False
    assert poor["reason"] == "бюджет миссии"
    assert poor["burns"] == []
    assert poor["total_m_s"] > 1.0
    rich = drag_makeup_schedule(
        800_000.0,
        mass_kg=1_000.0,
        area_m2=0.01,
        drag_coefficient=2.2,
        horizon_sec=86_400.0,
        burns=4,
        dv_budget_m_s=1.0,
        mission_reserve_m_s=0.0,
    )
    assert rich["accepted"] is True
    assert rich["model"] == "exponential_scale_height"
    assert len(rich["burns"]) == 4
    assert sum(item["dv_m_s"] for item in rich["burns"]) == pytest.approx(rich["total_m_s"])
    assert rich["total_m_s"] < 1.0

    blocked = _sun_engine()
    kept = blocked.commit_drag_makeup(3_600.0, 3, 0.0)
    assert kept["accepted"] is True
    assert len(blocked.scenario.planned_events) == 3
    plain = _sun_engine()
    before = blocked.spacecraft.state.propulsion.delta_v_remaining_m_s
    blocked.step(900.0, "imaging")
    plain.step(900.0, "imaging")
    assert blocked._keeping_block is True
    assert blocked._imaging_count == 0
    assert plain._imaging_count == 1
    assert blocked.spacecraft.state.propulsion.delta_v_remaining_m_s < before - 1.0e-8
    denied = _sun_engine()
    empty = denied.commit_drag_makeup(3_600.0, 3, denied.spacecraft.state.propulsion.delta_v_remaining_m_s)
    assert empty["accepted"] is False
    assert denied.scenario.planned_events == []
