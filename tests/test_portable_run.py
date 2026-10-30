"""Portable checkpoints and replay must survive loss of the source scenario."""
import hashlib
import json
import subprocess
import sys

import pytest

from space_env.data_generators.portable_run import (
    RunRecorder,
    export_transitions,
    replay_run,
)
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import (
    canonical_json,
    digest,
    load_world,
    save_world,
    world_payload,
)
from tests.test_execution_integrity import world


def _record(path, *, lineage="same-mission"):
    w = world()
    w.set_outages([("Shared", 2, 4)])
    recorder = RunRecorder(w, lineage=lineage, provenance={"teacher": "manual-test", "expert": False})
    recorder.step(6, [MissionCommand("c", "a", "comms", duration_sec=5)])
    recorder.step(4, [])
    recorder.save(path)
    return w


def _rewrite(path, edit):
    payload = json.loads(path.read_text())
    edit(payload)
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    path.write_text(canonical_json(payload))


def test_portable_checkpoint_restores_custom_configuration_history_and_next_step(tmp_path, monkeypatch):
    w = world()
    w.engines["a"].spacecraft.config.power.battery_capacity_Wh = 987
    w.engines["a"].constraints.battery_soc_min = .3
    w.engines["a"].mission_goal.data_downlink_Mb = 42
    w.step(2, {"a": "comms"})
    p = tmp_path / "snapshot.json"
    # The obsolete argument is deliberately a nonexistent path.
    save_world(p, w, tmp_path / "deleted.yaml")

    def forbidden(*args, **kwargs):
        raise AssertionError("catalog must not be consulted")

    monkeypatch.setattr("space_env.catalog.build.make_engine", forbidden)
    restored = load_world(p)
    assert restored.engines["a"].spacecraft.config.power.battery_capacity_Wh == 987
    assert restored.engines["a"].constraints.battery_soc_min == .3
    assert restored.engines["a"].mission_goal.data_downlink_Mb == 42
    assert len(restored.engines["a"].log) == len(w.engines["a"].log)
    assert restored.engines["a"].log[-1][1] == w.engines["a"].log[-1][1]
    w.step(3, {"a": "comms"})
    restored.step(3, {"a": "comms"})
    assert restored.engines["a"].delivered_bits == w.engines["a"].delivered_bits
    assert restored.engines["a"].get_state_snapshot() == w.engines["a"].get_state_snapshot()
    assert restored.engines["a"].receipts == w.engines["a"].receipts


def test_portable_experiment_replays_in_another_process_and_exports(tmp_path):
    p = tmp_path / "run.json"
    _record(p)
    script = "import json,sys; from space_env.data_generators.portable_run import replay_run; print(json.dumps(replay_run(sys.argv[1])))"
    output = subprocess.check_output([sys.executable, "-c", script, str(p)], cwd="/tmp", text=True)
    assert json.loads(output)["ok"]
    data = tmp_path / "transitions.jsonl"
    report = export_transitions(p, data)
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    assert report["rows"] == len(rows) == 2
    assert report["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert all(row["replay_verified"] for row in rows)
    assert rows[0]["action"]["commands"][0]["duration_sec"] == 5
    assert rows[0]["observation"]["a"]["time_sec"] == 0
    assert rows[0]["next_observation"]["a"]["time_sec"] == 6
    assert rows[0]["provenance"]["expert"] is False
    assert "hidden_cause" not in rows[0]["observation"]["a"]


@pytest.mark.parametrize("field", ["observation", "next_observation", "outcome"])
def test_replay_detects_changed_transitions_even_with_recomputed_checksum(tmp_path, field):
    p = tmp_path / "run.json"
    _record(p)

    def edit(record):
        if field == "outcome":
            record["steps"][0][field]["time_sec"] = 99.
        else:
            record["steps"][0][field]["a"]["time_sec"] = 99.

    _rewrite(p, edit)
    assert replay_run(p)["ok"] is False
    target = tmp_path / "existing.jsonl"
    target.write_text("previous complete dataset\n")
    with pytest.raises(ValueError, match="replay failed"):
        export_transitions(p, target)
    assert target.read_text() == "previous complete dataset\n"


def test_corruption_and_runtime_change_are_not_certified(tmp_path):
    p = tmp_path / "run.json"
    _record(p)
    _rewrite(p, lambda record: record["runtime"].update(python="different"))
    assert replay_run(p)["reason"] == "runtime_mismatch"
    payload = json.loads(p.read_text())
    payload["lineage"] = "unsealed mutation"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupt"):
        replay_run(p)


def test_failed_step_does_not_append_or_leave_accepted_commands(tmp_path):
    w = world()
    recorder = RunRecorder(w, lineage="failure")
    before = world_payload(w)
    with pytest.raises(ValueError):
        recorder.step(11, [MissionCommand("bad", "a", "comms")])
    assert not recorder.steps
    assert world_payload(w) == before


def test_external_unrecorded_mutation_is_detected(tmp_path):
    w = world()
    recorder = RunRecorder(w, lineage="mutated")
    w.engines["a"].accept_bits(10, origin="imaging", request_id="outside")
    recorder.step(1, [])
    p = tmp_path / "run.json"
    recorder.save(p)
    assert replay_run(p)["ok"] is False


def test_lineage_groups_variants_into_same_split(tmp_path):
    reports = []
    for variant in range(2):
        p = tmp_path / f"run{variant}.json"
        _record(p, lineage="mission-family")
        reports.append(export_transitions(p, tmp_path / f"data{variant}.jsonl"))
    assert reports[0]["split"] == reports[1]["split"]


def test_snapshot_checksum_and_atomic_write(tmp_path, monkeypatch):
    p = tmp_path / "world.json"
    save_world(p, world())
    original = p.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected replacement failure")

    monkeypatch.setattr("space_env.simulation.snapshot.os.replace", fail)
    with pytest.raises(OSError):
        save_world(p, world())
    assert p.read_bytes() == original
    assert list(tmp_path.iterdir()) == [p]
    payload = json.loads(p.read_text())
    payload["horizon_sec"] = 999
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupt"):
        load_world(p)


def test_cli_records_then_replays_after_yaml_and_plan_are_deleted(tmp_path, capsys):
    from space_env.cli import main

    scenario = tmp_path / "mission.yaml"
    scenario.write_text("version: 2\nepoch: '2026-10-20T10:00:00Z'\nhorizon_sec: 5\ninstances:\n  - {id: a, model: cubesat-6u}\nstations: []\nrequests: []\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"steps": [{"dt_sec": 5, "commands": []}]}))
    artifact = tmp_path / "run.json"
    main(["record-commands", str(scenario), str(plan), str(artifact), "--lineage", "cli-family"])
    capsys.readouterr()
    scenario.unlink()
    plan.unlink()
    result = subprocess.check_output([sys.executable, "-m", "space_env.cli", "replay-run", str(artifact)], cwd="/tmp", text=True)
    assert json.loads(result)["ok"]
    main(["replay-run", str(artifact)])
    assert json.loads(capsys.readouterr().out)["ok"]
    main(["export-transitions", str(artifact), str(tmp_path / "data.jsonl")])
    assert json.loads(capsys.readouterr().out)["rows"] == 1


def test_multisatellite_commands_and_shared_station_replay(tmp_path):
    from space_env.simulation.snapshot import world_from_payload
    from space_env.simulation.world import MissionWorld

    first = world()
    second = world_from_payload(world_payload(first))
    second.engines["a"].spacecraft.config.name = "b"
    fleet = MissionWorld({"a": first.engines["a"], "b": second.engines["a"]},
                         epoch=first.epoch, horizon_sec=10)
    recorder = RunRecorder(fleet, lineage="shared-station")
    recorder.step(10, [MissionCommand("a-pass", "a", "comms"), MissionCommand("b-pass", "b", "comms")])
    assert fleet.engines["a"].delivered_bits > 0
    assert fleet.engines["b"].delivered_bits == 0
    p = tmp_path / "fleet.json"
    recorder.save(p)
    assert replay_run(p)["ok"]


def test_checkpoint_continuation_keeps_event_times_and_history(tmp_path):
    from space_env.mission.events import EventKind, MissionEvent
    from space_env.simulation.snapshot import journal

    w = world()
    w.step(3)
    event = MissionEvent(kind=EventKind.RECOVERY, time=w.engines["a"].now(), time_sec=3,
                         payload={"action": "test-history"})
    w.engines["a"].log[-1][2].append(event)
    p = tmp_path / "midway.json"
    save_world(p, w)
    restored = load_world(p)
    assert journal(restored.engines["a"]) == journal(w.engines["a"])
    recorder = RunRecorder(restored, lineage="continuation")
    recorder.step(7, [MissionCommand("remaining", "a", "comms", duration_sec=2)])
    run = tmp_path / "continuation.json"
    recorder.save(run)
    assert replay_run(run)["ok"]


def test_legacy_path_snapshot_remains_readable(tmp_path):
    from space_env.mission.scenario_v2 import world_from_scenario
    from space_env.simulation.snapshot import encode

    scenario = tmp_path / "old.yaml"
    scenario.write_text("version: 2\nhorizon_sec: 10\ninstances:\n  - {id: a, model: cubesat-6u}\nstations: []\nrequests: []\n")
    w = world_from_scenario(scenario)
    w.step(2)
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"schema": 1, "scenario": str(scenario), "snapshot": encode(w.snapshot())}))
    assert load_world(p).now_sec() == 2
