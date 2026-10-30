"""Causal tool dialogues, revision/sensor checkpoints and serial relay boundaries."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from space_env.data_generators.agent_run import (
    AgentRunRecorder,
    export_tool_dialogue,
    replay_agent_run,
)
from space_env.mission.agent_protocol import invoke_tool, tool_schemas
from space_env.mission.awareness import SensorModel, SensorStream
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.session import MissionSession
from space_env.mission.tools import MissionTools
from space_env.simulation.fleet import Fleet
from space_env.simulation.relay import relay_step
from space_env.simulation.snapshot import canonical_json, digest, world_payload
from space_env.simulation.world import MissionWorld
from tests.test_execution_integrity import world


def _tools(*, budget=0):
    return MissionTools.from_session(MissionSession.from_world(world()), seed=7, call_budget=budget)


def _relay_world(*, separation=.02, horizon=60):
    fleet = Fleet.string_of_pearls(2, separation_deg=separation,
                                  epoch="2026-10-20T10:00:00Z", scenario=ScenarioConfig(ground_stations=[]))
    w = MissionWorld(fleet.vehicles, epoch="2026-10-20T10:00:00Z", horizon_sec=horizon)
    w.engines["SC-1"].accept_bits(5e6, origin="imaging", request_id="scene")
    return w


def _rewrite(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    path.write_text(canonical_json(data))


def test_session_and_tools_checkpoint_preserve_revision_pending_sensor_and_budget():
    tools = _tools(budget=10)
    tools.stream = SensorStream(SensorModel(sigma_soc=.01, period_sec=2), np.random.default_rng(42))
    tools.observe("a")
    tools.submit([{"command_id": "p", "actor_id": "a", "kind": "comms", "duration_sec": 2}])
    state = json.loads(json.dumps(tools.capture()))
    restored = MissionTools.from_snapshot(state)
    assert restored.stream is restored.session.world._sensor_streams[0]
    assert restored.calls == tools.calls == 2
    assert restored.session.session_id == tools.session.session_id
    assert restored.advance(3, expected_revision=0) == tools.advance(3, expected_revision=0)
    assert restored.observe("a") == tools.observe("a")
    assert restored.session.world.engines["a"].receipts == tools.session.world.engines["a"].receipts
    assert restored.revision == 1
    assert restored.session.report()["track"] == tools.session.report()["track"]


def test_recorded_failures_and_successes_replay_and_export_without_verifier_truth(tmp_path):
    tools = _tools(budget=5)
    recorder = AgentRunRecorder(tools, task="Deliver the onboard scene", lineage="dialogue-family",
                                provenance={"teacher": "manual-test", "expert": False})
    assert recorder.invoke("observe", {})["ok"]
    assert not recorder.invoke("submit", {"commands": [], "unknown": 1})["ok"]
    assert recorder.invoke("submit", {"commands": [{"command_id": "pass", "actor_id": "a", "kind": "comms", "duration_sec": 2}]})["ok"]
    assert not recorder.invoke("advance", {"dt_sec": 3, "expected_revision": 9})["ok"]
    assert recorder.invoke("advance", {"dt_sec": 3, "expected_revision": 0})["ok"]
    assert recorder.invoke("observe", {})["ok"]
    assert not recorder.invoke("observe", {})["ok"]  # Budget exceeded, recorded as a failure.
    p = tmp_path / "agent.json"
    recorder.save(p)
    assert replay_agent_run(p)["ok"]
    result = subprocess.check_output([sys.executable, "-c", "import json,sys; from space_env.data_generators.agent_run import replay_agent_run; print(json.dumps(replay_agent_run(sys.argv[1])))", str(p)], cwd="/tmp", text=True)
    assert json.loads(result)["ok"]
    target = tmp_path / "dialogues.jsonl"
    report = export_tool_dialogue(p, target)
    row = json.loads(target.read_text())
    assert report["events"] == 7
    assert len(row["messages"]) == 15
    assert row["provenance"]["expert"] is False
    assert row["messages"][0]["role"] == "user"
    assert all(message["role"] in {"user", "assistant", "tool"} for message in row["messages"])
    assert "initial" not in row and "final" not in row
    assert "after" not in row and "rng" not in row
    assert row["messages"][1]["tool_calls"][0]["id"] == row["messages"][2]["tool_call_id"]
    assert report["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,args", [("advance", {"dt_sec": float('inf'), "expected_revision": 0}),
                                       ("forecast", {"actor": "a", "dt_sec": -1}),
                                       ("advance", {"dt_sec": 1, "expected_revision": True}),
                                       ("missing", {})])
def test_schema_failures_never_invoke_executor(name, args):
    tools = _tools()
    before = tools.capture()
    assert invoke_tool(tools, name, args)["ok"] is False
    assert tools.capture() == before


def test_forecast_and_failed_revision_do_not_mutate_physical_world():
    tools = _tools()
    before = world_payload(tools.session.world)
    assert invoke_tool(tools, "forecast", {"actor": "a", "dt_sec": 2})["ok"]
    assert world_payload(tools.session.world) == before
    assert invoke_tool(tools, "advance", {"dt_sec": 1, "expected_revision": 9})["ok"] is False
    assert tools.revision == 0
    assert world_payload(tools.session.world) == before
    assert tools.calls == 2


def test_session_reporting_error_rolls_back_revision_frames_and_world(monkeypatch):
    tools = _tools()
    before = tools.session.capture()

    def fail():
        raise OSError("report failed")

    monkeypatch.setattr(tools.session, "report", fail)
    with pytest.raises(OSError):
        tools.advance(1, expected_revision=0)
    assert tools.session.capture() == before


def test_relay_advances_all_clocks_before_receiving_and_preserves_data(monkeypatch):
    w = _relay_world()
    receiver = w.engines["SC-2"]
    original = receiver.accept_custody
    moments = []

    def receive(records):
        moments.append((w.now_sec(), [row["available_time_sec"] for row in records]))
        return original(records)

    monkeypatch.setattr(receiver, "accept_custody", receive)
    result = relay_step(w, "SC-1", "SC-2", 5)
    assert result["bits"] == pytest.approx(5e6)
    assert w.now_sec() == 5
    assert moments == [(5, [5])]
    assert w.engines["SC-1"].stored_bits == 0
    assert receiver.stored_bits == pytest.approx(5e6)
    assert w.relay_used["SC-1"] == [(0, 5)]
    assert all(row["available_time_sec"] >= 5 for row in receiver.capture()["products"])
    assert sum(engine.stored_bits for engine in w.engines.values()) == pytest.approx(5e6)


def test_relay_profile_handles_blocked_link_without_teleporting():
    w = _relay_world(separation=180)
    result = relay_step(w, "SC-1", "SC-2", 5)
    assert result["status"] == "no_transfer"
    assert w.now_sec() == 5
    assert w.engines["SC-1"].stored_bits == pytest.approx(5e6)
    assert w.engines["SC-2"].stored_bits == 0


@pytest.mark.parametrize("mode", ["processing", "expiry"])
def test_relay_does_not_borrow_not_ready_or_expired_inventory(mode):
    w = _relay_world()
    sender = w.engines["SC-1"]
    if mode == "processing":
        sender._products[0]["available_time_sec"] = 3
    else:
        sender.spacecraft.config.comms.ttl_sec = 3
    result = relay_step(w, "SC-1", "SC-2", 5)
    assert result["bits"] == 0
    assert w.engines["SC-2"].stored_bits == 0


def test_relay_failure_after_time_advance_is_atomic(monkeypatch):
    w = _relay_world()
    before = world_payload(w)

    def fail(records):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(w.engines["SC-2"], "accept_custody", fail)
    with pytest.raises(RuntimeError):
        relay_step(w, "SC-1", "SC-2", 5)
    assert world_payload(w) == before


def test_agent_relay_dialogue_is_replayable(tmp_path):
    tools = MissionTools.from_session(MissionSession.from_world(_relay_world()), call_budget=6)
    recorder = AgentRunRecorder(tools, task="Relay the scene to SC-2", lineage="relay-family")
    assert recorder.invoke("observe", {})["ok"]
    result = recorder.invoke("relay", {"source": "SC-1", "destination": "SC-2", "dt_sec": 5, "expected_revision": 0})
    assert result["ok"] and result["result"]["bits"] > 0
    assert tools.revision == 1
    assert recorder.invoke("observe", {})["ok"]
    p = tmp_path / "relay-run.json"
    recorder.save(p)
    assert replay_agent_run(p)["ok"]


def test_changed_response_or_runtime_blocks_dialogue_export(tmp_path):
    recorder = AgentRunRecorder(_tools(), task="Observe", lineage="tamper")
    recorder.invoke("observe", {})
    p = tmp_path / "agent.json"
    recorder.save(p)
    _rewrite(p, lambda row: row["events"][0]["response"]["result"].update(now_sec=99.))
    assert replay_agent_run(p)["ok"] is False
    output = tmp_path / "old.jsonl"
    output.write_text("previous data")
    with pytest.raises(ValueError):
        export_tool_dialogue(p, output)
    assert output.read_text() == "previous data"


def test_tool_schemas_match_supported_methods():
    schemas = tool_schemas()
    assert {row["function"]["name"] for row in schemas} == {"observe", "forecast", "submit", "cancel", "advance", "relay"}
    assert all(row["function"]["parameters"]["additionalProperties"] is False for row in schemas)


def test_advance_protocol_does_not_leak_ui_truth_or_bypass_sensor():
    tools = _tools()
    tools.stream = SensorStream(SensorModel(bias_soc=-.4), np.random.default_rng(4))
    observed = invoke_tool(tools, "observe", {})["result"]["actors"]["a"]
    assert observed["soc"] == pytest.approx(.6)
    result = invoke_tool(tools, "advance", {"dt_sec": 1, "expected_revision": 0})["result"]
    assert set(result) == {"revision", "now_sec", "horizon_sec", "statuses"}
    assert "vehicles" not in result and "episode" not in result


def test_each_actor_has_a_separate_period_cache_and_portable_rng():
    w = _relay_world()
    w.engines["SC-1"].spacecraft.state.power.battery_soc = .3
    w.engines["SC-2"].spacecraft.state.power.battery_soc = .8
    tools = MissionTools.from_session(MissionSession.from_world(w), seed=5)
    tools.stream = SensorStream(SensorModel(period_sec=10), np.random.default_rng(5))
    readings = tools.observe()["actors"]
    assert readings["SC-1"]["soc"] == pytest.approx(.3)
    assert readings["SC-2"]["soc"] == pytest.approx(.8)
    restored = MissionTools.from_snapshot(json.loads(json.dumps(tools.capture())))
    assert restored.observe() == tools.observe()
    assert restored._channels["SC-2"] is restored.session.world._sensor_streams[1]


def test_cli_tool_dialogue_survives_deletion_of_source_files(tmp_path, capsys):
    from space_env.cli import main

    scenario = tmp_path / "mission.yaml"
    scenario.write_text("version: 2\nepoch: '2026-10-20T10:00:00Z'\nhorizon_sec: 10\ninstances:\n  - {id: a, model: cubesat-6u}\nstations: []\nrequests: []\n")
    script = tmp_path / "calls.json"
    script.write_text(json.dumps({"task": "Observe then wait", "calls": [
        {"name": "observe", "arguments": {}},
        {"name": "advance", "arguments": {"dt_sec": 2, "expected_revision": 0}},
    ]}))
    run = tmp_path / "tools.json"
    main(["record-tools", str(scenario), str(script), str(run), "--lineage", "cli-tools"])
    assert json.loads(capsys.readouterr().out)["events"] == 2
    scenario.unlink()
    script.unlink()
    output = subprocess.check_output([sys.executable, "-m", "space_env.cli", "replay-tools", str(run)], cwd="/tmp", text=True)
    assert json.loads(output)["ok"]
    main(["export-dialogue", str(run), str(tmp_path / "sft.jsonl")])
    assert json.loads(capsys.readouterr().out)["messages"] == 5


@pytest.mark.parametrize("kind", ["busy", "unknown", "self", "horizon", "negative"])
def test_invalid_relay_is_physically_atomic(kind):
    w = _relay_world()
    if kind == "busy":
        from space_env.simulation.commands import MissionCommand

        w.accept([MissionCommand("busy", "SC-1", "wait", duration_sec=10)])
    source, destination, duration = "SC-1", "SC-2", 5
    if kind == "unknown":
        source = "missing"
    elif kind == "self":
        destination = source
    elif kind == "horizon":
        duration = 61
    elif kind == "negative":
        duration = -1
    before = world_payload(w)
    with pytest.raises((ValueError, KeyError)):
        relay_step(w, source, destination, duration)
    assert world_payload(w) == before


def test_tool_cancel_is_recorded_and_stops_running_downlink(tmp_path):
    tools = _tools()
    recorder = AgentRunRecorder(tools, task="Stop transmission after one second", lineage="cancel-family")
    recorder.invoke("submit", {"commands": [{"command_id": "p", "actor_id": "a", "kind": "comms", "duration_sec": 8}]})
    recorder.invoke("advance", {"dt_sec": 1, "expected_revision": 0})
    assert recorder.invoke("cancel", {"command_id": "p"})["ok"]
    recorder.invoke("advance", {"dt_sec": 9, "expected_revision": 1})
    assert tools.session.world.engines["a"].delivered_bits == pytest.approx(1e6)
    p = tmp_path / "cancel.json"
    recorder.save(p)
    assert replay_agent_run(p)["ok"]


def test_unexpected_execution_error_is_not_a_training_label(monkeypatch):
    tools = _tools()
    recorder = AgentRunRecorder(tools, task="Observe", lineage="programming-failure")

    def fail(*args, **kwargs):
        raise OSError("unexpected storage failure")

    monkeypatch.setattr(tools, "observe", fail)
    with pytest.raises(OSError):
        recorder.invoke("observe", {})
    assert recorder.events == []


def test_dialogue_excludes_hidden_fault_cause(tmp_path):
    tools = _tools()
    tools.session.world.engines["a"]._hidden_cause = "SECRET_FAULT_CAUSE_FOR_VERIFIER"
    recorder = AgentRunRecorder(tools, task="Monitor telemetry", lineage="hidden-fault")
    recorder.invoke("observe", {})
    recorder.invoke("advance", {"dt_sec": 1, "expected_revision": 0})
    p = tmp_path / "agent.json"
    recorder.save(p)
    assert "SECRET_FAULT_CAUSE_FOR_VERIFIER" in p.read_text()
    target = tmp_path / "dialogue.jsonl"
    export_tool_dialogue(p, target)
    assert "SECRET_FAULT_CAUSE_FOR_VERIFIER" not in target.read_text()


def test_replay_keeps_actor_order_for_sensor_channel_identity(tmp_path):
    w = _relay_world()
    w.engines = dict(reversed(list(w.engines.items())))
    tools = MissionTools.from_session(MissionSession.from_world(w))
    tools.stream = SensorStream(SensorModel(period_sec=10), np.random.default_rng(3))
    recorder = AgentRunRecorder(tools, task="Observe fleet", lineage="reverse-order")
    recorder.invoke("observe", {})
    recorder.invoke("advance", {"dt_sec": 1, "expected_revision": 0})
    recorder.invoke("observe", {})
    p = tmp_path / "run.json"
    recorder.save(p)
    assert replay_agent_run(p)["ok"]


def test_forged_declared_mission_context_is_rejected(tmp_path):
    recorder = AgentRunRecorder(_tools(), task="Observe", lineage="context-test")
    recorder.invoke("observe", {})
    p = tmp_path / "run.json"
    recorder.save(p)
    _rewrite(p, lambda row: row["context"].update(actors=["nonexistent"]))
    with pytest.raises(ValueError, match="context differs"):
        replay_agent_run(p)


def test_relay_receiver_can_downlink_only_in_subsequent_interval():
    from space_env.core.access import subsatellite_latlon
    from space_env.core.ground import GroundStation

    w = _relay_world()
    receiver = w.engines["SC-2"]
    lat, lon = subsatellite_latlon(receiver.orbit.r, receiver.now())
    receiver.stations = [GroundStation(name="Downlink", lat_deg=lat, lon_deg=lon, min_elevation_deg=5)]
    receiver._refresh_environment()
    relay_step(w, "SC-1", "SC-2", 5)
    assert not receiver.receipts
    w.step(1, {"SC-2": "comms"})
    assert receiver.receipts
    assert min(row["start_sec"] for row in receiver.receipts) >= 5


@pytest.mark.parametrize("mode", ["radio", "room", "receiver-expiry", "event", "reserved"])
def test_relay_checks_destination_resources_and_boundaries(mode):
    from space_env.core.spacecraft import SubsystemHealth
    from space_env.mission.scenarios import AnomalyInject

    w = _relay_world()
    receiver = w.engines["SC-2"]
    if mode == "radio":
        receiver.spacecraft.state.health["comms"] = SubsystemHealth.FAILED
    elif mode == "room":
        receiver.spacecraft.config.payload.memory_bits = 1e6
    elif mode == "receiver-expiry":
        receiver.spacecraft.config.comms.ttl_sec = 3
    elif mode == "event":
        receiver.scenario.anomaly_injects.append(AnomalyInject(subsystem="comms", at_sec=2))
    else:
        receiver._terminal_used = [(0, 5)]
    before = world_payload(w)
    if mode in {"event", "reserved"}:
        with pytest.raises(ValueError):
            relay_step(w, "SC-1", "SC-2", 5)
        assert world_payload(w) == before
    else:
        result = relay_step(w, "SC-1", "SC-2", 5)
        assert result["bits"] == (1e6 if mode == "room" else 0)
        assert sum(engine.stored_bits for engine in w.engines.values()) == pytest.approx(5e6)


def test_recorded_heuristic_uses_tools_and_delivers_real_data(tmp_path):
    from space_env.data_generators.agent_run import record_baseline

    tools = _tools()
    p = tmp_path / "baseline.json"
    report = record_baseline(tools, p, lineage="teacher-family", step_sec=5, max_steps=2)
    assert report["now_sec"] == 10
    assert report["events"] == 6
    assert tools.session.world.engines["a"].delivered_bits == pytest.approx(10e6)
    assert replay_agent_run(p)["ok"]
    row = json.loads(p.read_text())
    assert row["provenance"]["expert"] is False
    assert [event["name"] for event in row["events"]] == ["observe", "submit", "advance"] * 2


def test_recorded_heuristic_does_not_bypass_budget_failure(tmp_path):
    from space_env.data_generators.agent_run import record_baseline

    tools = _tools(budget=1)
    p = tmp_path / "budget.json"
    result = record_baseline(tools, p, lineage="budget-teacher", step_sec=5)
    assert result["failed_calls"] == 1
    assert tools.session.world.now_sec() == 0
    assert replay_agent_run(p)["ok"]


def test_protocol_forecast_does_not_expose_battery_or_injected_future_truth():
    from space_env.mission.scenarios import AnomalyInject

    left, right = _tools(), _tools()
    right.session.world.engines["a"].spacecraft.state.power.battery_soc = .1
    right.session.world.engines["a"].scenario.anomaly_injects.append(
        AnomalyInject(subsystem="comms", at_sec=1, hidden=True, cause="secret"))
    before = world_payload(right.session.world)
    result = invoke_tool(right, "forecast", {"actor": "a", "dt_sec": 2})
    assert result == invoke_tool(left, "forecast", {"actor": "a", "dt_sec": 2})
    assert result["result"]["kind"] == "nominal_geometry_forecast"
    assert "soc" not in result["result"] and "stored_bits" not in result["result"]
    assert world_payload(right.session.world) == before
