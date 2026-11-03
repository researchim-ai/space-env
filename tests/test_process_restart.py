"""Real process boundaries, bound checkpoint handoff and retained failures."""

from __future__ import annotations

import json
import os

import pytest

from space_env.scaling import restart
from space_env.scaling.benchmark import Workload, _execute_workload, run_scale
from space_env.scaling.ensemble import (
    EnsembleQualificationSpec,
    prepare_ensemble_qualification,
    run_ensemble_qualification,
    verify_ensemble_qualification,
)
from space_env.scaling.worker import measure
from space_env.simulation.snapshot import canonical_json, digest
from tests.test_ensemble_qualification import spec as base_spec


@pytest.fixture
def spec():
    value = base_spec.__wrapped__()
    value["sensitivity"]["samples"] = 1
    value["resume_in_new_process"] = True
    return value


def workload(spec):
    return Workload.model_validate(
        {**spec["workload"], "resume_after_decisions": 1, "resume_in_new_process": True}
    )


def seal(packet):
    packet.pop("sha256", None)
    packet["sha256"] = digest(packet)
    return packet


@pytest.fixture
def packet(spec, tmp_path):
    w = workload(spec)
    path = tmp_path / "packet.json"
    restart.write_prefix(w.model_dump(mode="json"), path, "1" * 32)
    return w, json.loads(path.read_text())


def test_new_flags_omit_false_and_preserve_legacy_serialized_contract(spec):
    spec.pop("resume_in_new_process")
    default = EnsembleQualificationSpec.model_validate(spec).model_dump(mode="json")
    explicit = EnsembleQualificationSpec.model_validate(
        {**spec, "resume_in_new_process": False}
    ).model_dump(mode="json")
    assert default == explicit and "resume_in_new_process" not in default
    w = Workload.model_validate(spec["workload"])
    assert "resume_in_new_process" not in w.model_dump(mode="json")


def test_only_restored_workloads_use_new_process(spec):
    p = prepare_ensemble_qualification(spec)
    assert len(p["scale_manifest"]["workloads"]) == 4
    assert (
        sum(
            w.get("resume_in_new_process", False)
            for w in p["scale_manifest"]["workloads"]
        )
        == 2
    )
    assert all(
        w.get("resume_in_new_process", False) == w["id"].endswith("-cold")
        for w in p["scale_manifest"]["workloads"]
    )


@pytest.mark.parametrize("mutation", ["missing_point", "bool", "base_override"])
def test_invalid_new_process_spec(spec, mutation):
    if mutation == "missing_point":
        spec["resume_after_decisions"] = None
    if mutation == "bool":
        spec["resume_in_new_process"] = 1
    if mutation == "base_override":
        spec["workload"].update(resume_in_new_process=True, resume_after_decisions=1)
    with pytest.raises(ValueError):
        prepare_ensemble_qualification(spec)


def test_workload_requires_resume_point(spec):
    with pytest.raises(ValueError):
        Workload.model_validate({**spec["workload"], "resume_in_new_process": True})


def test_valid_packet_checks_workload_build_and_process_identity(packet):
    w, p = packet
    record = restart.validate_packet(
        p, w, "1" * 32, producer_pid=os.getpid(), consumer_pid=os.getpid() + 1
    )
    assert record["sha256"] == p["checkpoint_sha256"]
    assert p["build"] == restart.build_identity()
    assert p["prefix_decisions"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "nonce",
        "build",
        "workload",
        "producer",
        "prefix_count",
        "checkpoint_hash",
        "trace",
        "seed",
        "policy",
        "task",
        "operator",
        "episode",
        "negative_time",
        "bool_peak",
    ],
)
def test_invalid_or_resealed_prefix_rejected(packet, mutation):
    w, p = packet
    if mutation == "checksum":
        p["prefix_decisions"] = 2
    else:
        if mutation == "nonce":
            p["nonce"] = "2" * 32
        if mutation == "build":
            p["build"]["package_version"] = "foreign"
        if mutation == "workload":
            p["workload_sha256"] = digest("foreign")
        if mutation == "producer":
            p["producer_pid"] = False
        if mutation == "prefix_count":
            p["prefix_decisions"] = True
        if mutation == "checkpoint_hash":
            p["checkpoint_sha256"] = digest("foreign")
        if mutation == "negative_time":
            p["prefix_elapsed_sec"] = -1.0
        if mutation == "bool_peak":
            p["prefix_peak_rss_bytes"] = True
        if mutation in {"trace", "seed", "policy", "task", "operator", "episode"}:
            cp = p["checkpoint"]
            if mutation == "trace":
                cp["trace"] = None
            if mutation == "seed":
                cp["episode"]["episode_seed"] += 1
            if mutation == "policy":
                cp["policy"]["config"]["seed"] += 1
            if mutation == "task":
                cp["episode"]["task"]["id"] = "different"
            if mutation == "operator":
                cp["operator_enabled"] = True
            if mutation == "episode":
                cp["episode"] = []
            seal(cp)
            p["checkpoint_sha256"] = cp["sha256"]
        seal(p)
    with pytest.raises((ValueError, TypeError)):
        restart.validate_packet(p, w, "1" * 32, consumer_pid=os.getpid() + 1)


def test_same_process_and_wrong_observed_producer_rejected(packet):
    w, p = packet
    with pytest.raises(ValueError, match="same-process"):
        restart.validate_packet(p, w, "1" * 32, consumer_pid=os.getpid())
    with pytest.raises(ValueError, match="identity"):
        restart.validate_packet(p, w, "1" * 32, producer_pid=os.getpid() + 1)


def test_process_worker_cannot_fall_back_to_local_restore(packet, tmp_path):
    w, p = packet
    with pytest.raises(ValueError, match="requires"):
        measure(w.model_dump(mode="json"))
    path = tmp_path / "packet.json"
    path.write_text(canonical_json(p))
    with pytest.raises(ValueError, match="same-process"):
        measure(w.model_dump(mode="json"), restart_packet=path, restart_nonce="1" * 32)


@pytest.mark.parametrize("kind", ["symlink", "oversize", "truncated"])
def test_packet_io_is_bounded_and_no_symlinks(tmp_path, kind):
    path = tmp_path / "packet.json"
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("{}")
        path.symlink_to(target)
    elif kind == "oversize":
        path.write_bytes(b" " * 101)
    else:
        path.write_text("{")
    with pytest.raises(ValueError):
        restart.read_json(path, 100)


def test_prefix_budget_and_invalid_nonce_do_not_publish_packet(spec, tmp_path):
    w = workload(spec).model_copy(update={"max_checkpoint_bytes": 1})
    target = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="byte budget"):
        restart.write_prefix(w.model_dump(mode="json"), target, "1" * 32)
    assert not target.exists()
    with pytest.raises(ValueError, match="nonce"):
        restart.write_prefix(workload(spec).model_dump(mode="json"), target, "bad")
    assert not target.exists()


def test_real_two_process_restore_matches_uninterrupted_and_audits(spec, tmp_path):
    r = run_ensemble_qualification(spec, tmp_path / "run")
    assert r["passed"], r["summary"]
    assert r["summary"]["planned_process_launches"] == 6
    native = json.loads((tmp_path / "run" / "native" / "result.json").read_text())
    resumed = [row for row in native["rows"] if "process_restart" in row]
    assert len(resumed) == 2
    for row in resumed:
        assert row["restart_protocol"]["status"] == "completed"
        assert (
            row["process_restart"]["producer_pid"]
            != row["process_restart"]["consumer_pid"]
        )
        assert row["gates"]["process_restart"]
        restart.validate_observed_restart(
            row,
            Workload.model_validate(
                next(
                    w
                    for w in prepare_ensemble_qualification(spec)["scale_manifest"][
                        "workloads"
                    ]
                    if w["id"] == row["id"]
                )
            ),
        )
    assert verify_ensemble_qualification(tmp_path / "run")["verified"]
    packet_path = next((tmp_path / "run" / "native" / "workers").glob("*/restart.json"))
    packet_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_ensemble_qualification(tmp_path / "run")


def test_prefix_corruption_never_starts_continuation(spec, tmp_path, monkeypatch):
    original = restart.read_json

    def corrupted(path, limit):
        value = original(path, limit)
        if str(path).endswith("restart.json"):
            value["nonce"] = "2" * 32
        return value

    monkeypatch.setattr(restart, "read_json", corrupted)
    row = _execute_workload(workload(spec), 0, progress_path=tmp_path / "progress.json")
    assert not row["passed"] and row["error"]["type"] == "ValueError"
    assert len(row["restart_protocol"]["phases"]) == 1
    assert row["restart_protocol"]["phases"][0]["phase"] == "prefix"


def test_actual_shared_timeout_retains_failure_and_reaps_children(spec, tmp_path):
    w = workload(spec).model_copy(update={"timeout_sec": 1})
    row = _execute_workload(w, 0, progress_path=tmp_path / "progress.json")
    assert not row["passed"] and row["error"]["type"] == "TimeoutExpired"
    assert row["restart_protocol"]["status"] == "failed"
    for phase in row["restart_protocol"]["phases"]:
        if "pid" in phase:
            with pytest.raises(ProcessLookupError):
                os.kill(phase["pid"], 0)


def test_prefix_size_failure_is_a_retained_native_row(spec, tmp_path):
    w = workload(spec).model_dump(mode="json")
    w["max_checkpoint_bytes"] = 1
    r = run_scale({"id": "prefix-budget", "workloads": [w]}, tmp_path / "run")
    assert not r["ok"] and len(r["rows"]) == 1
    assert r["rows"][0]["restart_protocol"]["status"] == "failed"
    assert r["rows"][0]["error"] is not None
    assert not list((tmp_path / "run").rglob("restart.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        "same_pid",
        "consumer",
        "nonce",
        "exit",
        "order",
        "gate",
        "prefix_bool",
        "phase_type",
    ],
)
def test_observed_protocol_cannot_contradict_success(spec, mutation):
    w = workload(spec)
    row = {
        "worker_environment": {"pid": 200},
        "gates": {"process_restart": True},
        "process_restart": {
            "producer_pid": 100,
            "consumer_pid": 200,
            "nonce": "1" * 32,
            "packet_sha256": "a" * 64,
            "prefix_decisions": 1,
        },
        "restart_protocol": {
            "status": "completed",
            "nonce": "1" * 32,
            "packet_sha256": "a" * 64,
            "phases": [
                {"phase": "prefix", "pid": 100, "status": "exited", "returncode": 0},
                {
                    "phase": "continuation",
                    "pid": 200,
                    "status": "exited",
                    "returncode": 0,
                },
            ],
        },
    }
    restart.validate_observed_restart(row, w)
    if mutation == "same_pid":
        row["restart_protocol"]["phases"][1]["pid"] = 100
    if mutation == "consumer":
        row["worker_environment"]["pid"] = 300
    if mutation == "nonce":
        row["process_restart"]["nonce"] = "2" * 32
    if mutation == "exit":
        row["restart_protocol"]["phases"][0]["returncode"] = 1
    if mutation == "order":
        row["restart_protocol"]["phases"].reverse()
    if mutation == "gate":
        row["gates"]["process_restart"] = False
    if mutation == "prefix_bool":
        row["process_restart"]["prefix_decisions"] = True
    if mutation == "phase_type":
        row["restart_protocol"]["phases"] = [None, None]
    with pytest.raises((ValueError, TypeError)):
        restart.validate_observed_restart(row, w)


def test_prefix_at_actual_episode_end_is_not_silently_moved(spec, tmp_path):
    w = workload(spec).model_dump(mode="json")
    w["scenario"]["definition"]["horizon_sec"] = 0.25
    for request in w["scenario"]["definition"]["lifecycle"]["requests"]:
        request["deadline_sec"] = 0.25
    with pytest.raises(ValueError, match="terminated"):
        restart.write_prefix(w, tmp_path / "absent.json", "1" * 32)
    assert not (tmp_path / "absent.json").exists()
