"""Cross-component episode and interval contracts, beyond smoke missions."""
import json

import gymnasium as gym
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.awareness import SensorModel, SensorStream
from space_env.mission.scenarios import ScenarioConfig
from space_env.rl.trajectories import rollout_episode
from space_env.simulation.commands import CommandBook, MissionCommand
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.parallel import ParallelFleet
from space_env.simulation.snapshot import decode, encode
from space_env.simulation.world import MissionWorld


def world():
    epoch = "2026-10-20T10:00:00Z"
    orbit = Orbit.from_keplerian(6_800_000., 0., .9, 0., 0., 0., Time(epoch))
    craft = Spacecraft(config=SpacecraftConfig(name="a"))
    craft.reset_state()
    engine = SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.stations = [GroundStation(name="Shared", lat_deg=lat, lon_deg=lon, min_elevation_deg=5)]
    engine._refresh_environment()
    engine.accept_bits(100e6, origin="imaging", request_id="frame")
    return MissionWorld({"a": engine}, epoch=epoch, horizon_sec=10)


@pytest.mark.parametrize("steps", [(10,), (1, 1, 8), (1, 3, 6)])
def test_duration_is_an_execution_boundary_not_a_post_step_label(steps):
    w = world()
    orders = w.accept([MissionCommand("pass", "a", "comms", duration_sec=2)])
    for index, seconds in enumerate(steps):
        w.step(seconds, orders if index == 0 else None)
        w.settle()
    slips = w.engines["a"].receipts
    assert slips
    assert max(row["end_sec"] for row in slips) <= 2 + 1e-9
    assert w.engines["a"].delivered_bits == pytest.approx(2e6)
    assert w.commands.get("pass")["status"] == "completed"
    assert w.now_sec() == 10


def test_failure_after_operation_boundary_rolls_back_whole_external_step(monkeypatch):
    w = world()
    orders = w.accept([MissionCommand("pass", "a", "comms", duration_sec=2)])
    real_step = w.engines["a"].step

    def fail_on_second_interval(dt, command=None):
        if w.engines["a"].time_sec >= 2:
            raise RuntimeError("injected")
        return real_step(dt, command)

    monkeypatch.setattr(w.engines["a"], "step", fail_on_second_interval)
    with pytest.raises(RuntimeError, match="injected"):
        w.step(10, orders)
    assert w.now_sec() == 0
    assert not w.engines["a"].receipts
    assert w.commands.get("pass")["status"] == "accepted"


def test_restore_rewinds_truth_and_sensor_rng_and_period_cache():
    w = world()
    stream = SensorStream(SensorModel(period_sec=2, sigma_soc=.1), np.random.default_rng(7))
    w.note_truth()
    w.measure("a", stream)
    snapshot = w.snapshot()
    w.step(3)
    expected = w.measure("a", stream)
    w.restore(snapshot)
    assert len(w._truth_log["a"]) == 1
    w.step(3)
    assert w.measure("a", stream) == expected


def test_parallel_episode_reset_validation_and_final_partial_step():
    env = ParallelFleet(world(), dt_sec=6)
    env.reset(seed=4)
    with pytest.raises(ValueError):
        env.step({"a": -1})
    assert env.world.now_sec() == 0
    env.step({"a": 2})
    _, _rewards, _, truncations, _ = env.step({"a": 0})
    assert truncations == {"a": True}
    assert env.world.now_sec() == 10
    assert env.agents == []
    assert env.step({}) == ({}, {}, {}, {}, {})
    env.reset(seed=4)
    assert env.world.now_sec() == 0
    assert env.world.engines["a"].delivered_bits == 0
    assert env.agents == ["a"]


def test_dict_episode_writes_structured_json_and_seeded_actions():
    def collect():
        env = gym.make("space_env/MissionDict-v0", max_episode_steps=3)
        try:
            return rollout_episode(env, seed=71)
        finally:
            env.close()

    first, second = collect(), collect()
    assert first == second
    encoded = json.loads(json.dumps(first, allow_nan=False))
    assert set(encoded[0]["observation"]) == {"clock", "soc", "contact", "open_requests"}
    assert encoded[-1]["truncated"]


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1])
def test_invalid_duration_does_not_write_command(duration):
    w = world()
    with pytest.raises(ValueError):
        w.accept([MissionCommand("bad", "a", "comms", duration_sec=duration)])
    assert w.commands.export() == {}


def test_command_checkpoint_is_immutable_across_all_lifecycle_mutations():
    book = CommandBook()
    for name in ("complete", "cancel", "fail", "reject"):
        book.submit(MissionCommand(name, "a", "comms", duration_sec=2))
    snapshot = book.checkpoint()
    expected = book.export()
    book.arm_window("complete", 0)
    book.mark_running("complete")
    book.complete("complete")
    book.cancel("cancel")
    book.fail("fail", "failure")
    book.reject("reject", "rejected")
    assert snapshot == expected
    book.restore_checkpoint(snapshot)
    book.complete("complete")
    assert snapshot == expected


def test_json_snapshot_rewinds_encounter_counters_and_subsystem_state():
    from space_env.mission.scenarios import ConjunctionInject

    w = world()
    engine = w.engines["a"]
    initial = w.snapshot()
    engine._start_conjunction(ConjunctionInject(at_sec=0, miss_distance_km=1, tca_in_sec=5))
    engine.step(1, "comms")
    saved = decode(json.loads(json.dumps(encode(w.snapshot()))))
    position = engine._encounter.r.copy()
    engine.step(1, "wait")
    w.restore(saved)
    np.testing.assert_array_equal(engine._encounter.r, position)
    assert engine.get_state_snapshot().comms_sessions == 1
    w.restore(initial)
    assert engine._encounter is None
    assert engine.get_state_snapshot().comms_sessions == 0


def test_tools_and_http_preserve_timed_command_duration(monkeypatch):
    from space_env.api import server
    from space_env.envs.mission_fleet_env import packaged_mission
    from space_env.mission.tools import MissionTools

    command = {"command_id": "bounded", "actor_id": "sentinel-a", "kind": "wait", "duration_sec": 2}
    tools = MissionTools(packaged_mission("family_nominal.yaml"))
    tools.submit([command])
    tools.advance(10, expected_revision=0)
    assert tools.session.world.commands.get("bounded")["until_sec"] == 2

    monkeypatch.setattr(server, "_SESSIONS", {})
    status, body, _kind = server.handle_post("/api/sessions", b'{"scenario":"family_nominal.yaml"}')
    assert status == 200
    sid = json.loads(body)["session_id"]
    status, _, _ = server.handle_post(
        f"/api/sessions/{sid}/advance",
        json.dumps({"expected_revision": 0, "dt_sec": 10, "commands": [command]}).encode(),
    )
    assert status == 200
    assert server._SESSIONS[sid].world.commands.get("bounded")["until_sec"] == 2


@pytest.mark.parametrize("status", ["complete", "cancel", "fail", "reject"])
@pytest.mark.parametrize("running", [False, True])
def test_active_indexes_follow_history_and_checkpoint_restore(status, running):
    book = CommandBook()
    for index in range(100):
        name = f"past-{index}"
        book.submit(MissionCommand(name, "old", "wait"))
        book.complete(name)
    book.submit(MissionCommand("live", "a", "comms", duration_sec=2))
    book.arm_window("live", 0)
    if running:
        book.mark_running("live")
    saved = book.checkpoint()
    if status in {"fail", "reject"}:
        getattr(book, status)("live", "failure")
    else:
        getattr(book, status)("live")
    assert book.running_orders() == {}
    assert book.active_windows() == []
    book.restore_checkpoint(saved)
    assert book.running_orders() == ({"a": "comms"} if running else {})
    assert [row["command_id"] for row in book.active_windows()] == ["live"]
    book.load(json.loads(json.dumps(book.export())))
    assert book.running_actors() == ({"a"} if running else set())
    assert book.submit(MissionCommand("live", "a", "comms", duration_sec=2))["status"] == (
        "running" if running else "accepted"
    )
