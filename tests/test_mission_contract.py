"""Контракт исполнения: прогноз, часы, антенна, квитанция и схема."""
import tempfile
from pathlib import Path

import yaml

from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PropulsionConfig, Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.events import EventKind, PlannedEvent
from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.tasking import TaskRequest, StationOutage, delivery_score, mission_kpis, run_policy
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

EPOCH = "2026-10-20T10:00:00Z"


def _engine(name: str = "a", planned=None) -> SimulationEngine:
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.9, 0.0, 0.0, 0.0, Time(EPOCH))
    craft = Spacecraft(config=SpacecraftConfig(name=name, propulsion=PropulsionConfig(delta_v_total_m_s=20.0)))
    craft.reset_state()
    return SimulationEngine(
        orbit,
        craft,
        scenario=ScenarioConfig(ground_stations=[], planned_events=list(planned or [])),
    )


def _world(engines=None, horizon: float = 60.0) -> MissionWorld:
    return MissionWorld(engines or {"a": _engine()}, epoch=EPOCH, horizon_sec=horizon, seed=7)


def test_forecast_keeps_the_burn_and_the_noise_seed():
    planned = PlannedEvent(
        kind=EventKind.MANEUVER,
        scheduled_at_sec=0.5,
        payload={"dv_m_s": 1.0, "direction": "prograde"},
    )
    world = _world({"a": _engine(planned=[planned])})
    before = world.engines["a"].spacecraft.state.propulsion.delta_v_remaining_m_s
    world.forecast("a", 1.0)
    after = world.engines["a"].spacecraft.state.propulsion.delta_v_remaining_m_s
    assert world.engines["a"].time_sec == 0.0
    assert after == before
    _snap, events = world.engines["a"].step(1.0, "wait")
    assert any(event.kind == EventKind.MANEUVER for event in events)
    fresh = _world()
    fresh.engines["a"].preload_bits(1000.0)
    snap = fresh.snapshot()
    first = fresh.estimate("a", noise=0.1)["stored_bits"]
    fresh.restore(snap)
    again = fresh.estimate("a", noise=0.1)["stored_bits"]
    assert first == again


def test_unknown_actor_and_split_clocks_do_not_move_the_world():
    world = _world()
    try:
        world.step(1.0, {"ghost": "imaging"})
        raised = False
    except KeyError:
        raised = True
    assert raised
    assert world.engines["a"].time_sec == 0.0
    early = _engine("b")
    early.step(1.0)
    try:
        _world({"a": _engine("a"), "b": early})
        clocks_ok = True
    except RuntimeError:
        clocks_ok = False
    assert clocks_ok is False


def test_runner_resumes_from_the_world_clock():
    world = _world(horizon=2.0)
    world.step(1.0)
    run_policy(world, [], step_sec=1.0)
    assert world.engines["a"].time_sec == 2.0


def test_duplicate_and_early_receipts_do_not_close_the_request():
    request = TaskRequest("r", 0.0, 0.0, 0.0, 10.0, 100.0, 1, "optical")
    receipt = {"product_id": "p", "request_id": "r", "destination": "x", "time_sec": 5.0, "offset_bits": 0.0, "bits": 60.0}
    assert delivery_score([request], [receipt], [])["on_time"] == []
    assert delivery_score([request], [receipt, dict(receipt)], [])["on_time"] == []
    overlap = dict(receipt, offset_bits=40.0, bits=60.0)
    assert delivery_score([request], [receipt, overlap], [])["on_time"] == ["r"]
    early = TaskRequest("later", 0.0, 0.0, 20.0, 30.0, 100.0, 1, "optical")
    premature = dict(receipt, request_id="later", bits=100.0, time_sec=1.0)
    assert delivery_score([early], [premature], [])["on_time"] == []


def test_safety_counts_a_real_constraint_event():
    world = _world()
    world.engines["a"].spacecraft.state.power.battery_soc = 0.01
    _snap, events = world.engines["a"].step(1.0)
    assert any(event.kind == EventKind.CONSTRAINT_VIOLATION for event in events)
    score = delivery_score([], [], [])
    kpis = mission_kpis(world, [], [], score, runtime_sec=0.0, replans=0, relay_bits=0.0)
    assert kpis["safety_violations"] >= 1


def test_invalid_scenario_is_rejected_before_propagation():
    baseline = {
        "version": 2,
        "epoch": EPOCH,
        "horizon_sec": 60,
        "instances": [{"id": "a", "model": "cubesat-6u"}],
        "stations": [],
        "requests": [],
    }
    bad = {
        "negative_horizon": dict(baseline, horizon_sec=-1),
        "nan_anomaly": dict(baseline, instances=[{"id": "a", "model": "cubesat-6u", "nu_deg": float("nan")}]),
        "bad_coordinates_and_deadline": dict(
            baseline,
            requests=[
                {
                    "id": "r",
                    "lat_deg": 120,
                    "lon_deg": 300,
                    "release_sec": 50,
                    "deadline_sec": 10,
                    "size_bits": -1,
                    "payload": "optical",
                }
            ],
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, config in bad.items():
            path = Path(tmp) / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            try:
                load_scenario_v2(path)
                accepted = True
            except ValueError:
                accepted = False
            assert accepted is False


def test_one_station_and_an_outage_are_enforced_by_the_world():
    def contacted(name: str) -> SimulationEngine:
        item = _engine(name)
        lat, lon = subsatellite_latlon(item.orbit.r, item.now())
        item.stations = [GroundStation(name="Shared", lat_deg=lat, lon_deg=lon, min_elevation_deg=5.0)]
        item._refresh_environment()
        item.accept_bits(100.0e6, origin="imaging", request_id=name)
        return item

    shared = _world({"a": contacted("a"), "b": contacted("b")})
    shared.step(1.0, {"a": "comms", "b": "comms"})
    delivered = {name: engine.delivered_bits for name, engine in shared.engines.items()}
    assert delivered["a"] > 0.0
    assert delivered["b"] == 0.0
    closed = _world({"a": contacted("a")}, horizon=3.0)
    report = run_policy(
        closed,
        [TaskRequest("a", 0.0, 0.0, 0.0, 10.0, 100.0e6, 1, "optical")],
        step_sec=1.0,
        outages=[StationOutage("Shared", 1.0, 3.0)],
    )
    assert report["receipts"]
    assert all(not (1.0 < float(item["time_sec"]) < 3.0) for item in report["receipts"])
    assert closed.engines["a"].delivered_bits < 2.0e6
