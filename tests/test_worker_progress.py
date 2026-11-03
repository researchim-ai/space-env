"""Diagnostics survive timeout; never stand in for native qualification."""

import json
import subprocess
from pathlib import Path

import pytest

from space_env.scaling.benchmark import Workload, _execute_workload, run_scale
from space_env.scaling.progress import MAX_PROGRESS_BYTES, StageReporter, read_progress
from space_env.scaling.worker import measure
from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import digest


def small_workload():
    return fleet_workload(
        {
            "id": "diagnostic",
            "actors": 2,
            "products": 2,
            "planes": 1,
            "horizon_sec": 4.0,
            "repeats": 1,
        }
    )


def test_atomic_stages_throttle_and_binding(tmp_path):
    path = tmp_path / "progress.json"
    reporter = StageReporter(path, "input")
    reporter.emit("initialize")
    first = read_progress(path, "input")
    reporter.emit("execution", decisions=0)
    second = read_progress(path, "input")
    reporter.emit("execution", decisions=1, physical_sec=2.0)
    assert read_progress(path, "input") == second
    reporter.emit("replay", decisions=2, physical_sec=4.0)
    reporter.emit("replay", status="completed")
    final = read_progress(path, "input")
    assert first["sequence"] < final["sequence"]
    assert final["status"] == "completed" and final["decisions"] == 2
    assert [s["stage"] for s in final["history"]] == ["initialize", "execution"]
    assert all(s["started_sec"] <= s["ended_sec"] for s in final["history"])
    assert read_progress(path, "wrong") is None
    assert path.stat().st_size <= MAX_PROGRESS_BYTES
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "content", [b"{", b"[]", b"x" * (MAX_PROGRESS_BYTES + 1), b'{"x":NaN}']
)
def test_torn_or_invalid_progress_is_not_a_result(tmp_path, content):
    path = tmp_path / "progress.json"
    path.write_bytes(content)
    assert read_progress(path, "input") is None


def test_write_failure_does_not_publish_partial_json(tmp_path, monkeypatch):
    from space_env.simulation import snapshot

    path = tmp_path / "progress.json"
    reporter = StageReporter(path, "input")
    reporter.emit("initialize")
    before = path.read_bytes()

    def fail(*args):
        raise OSError("disk failed")

    monkeypatch.setattr(snapshot.os, "replace", fail)
    with pytest.raises(OSError):
        reporter.emit("execution")
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_timeout_retains_last_stage_but_remains_failed(tmp_path, monkeypatch):
    from space_env.scaling import benchmark

    w = Workload.model_validate(small_workload())
    progress = tmp_path / "progress.json"

    def timeout(argv, **kwargs):
        StageReporter(progress, digest(w.model_dump(mode="json"))).emit(
            "replay", decisions=2, physical_sec=4.0
        )
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    row = _execute_workload(w, 0, progress_path=progress)
    assert not row["passed"] and row["error"]["type"] == "TimeoutExpired"
    assert row["worker_progress"]["stage"] == "replay"
    assert row["worker_progress"]["status"] == "running"
    assert "metrics" not in row and "native_backend_sha256" not in row


def test_exception_stage_keeps_phase_and_bounded_error(tmp_path):
    reporter = StageReporter(tmp_path / "progress.json", "input")
    reporter.emit("checkpoint")
    reporter.emit("checkpoint", status="failed", error=ValueError("x" * 2000))
    result = read_progress(reporter.path, "input")
    assert result["stage"] == "checkpoint" and result["status"] == "failed"
    assert len(result["error"]["message"]) == 512


def test_telemetry_does_not_change_native_or_controller_semantics(tmp_path):
    w = small_workload()
    control = measure(w)
    path = tmp_path / "progress.json"
    reporter = StageReporter(path, digest(w))
    observed = measure(w, progress=reporter)
    assert observed["native_backend_sha256"] == control["native_backend_sha256"]
    assert observed["trace_sha256"] == control["trace_sha256"]
    assert observed["gates"] == control["gates"]
    assert read_progress(path, digest(w))["stage"] == "export"


def test_parallel_failures_keep_manifest_order_and_denominators(tmp_path, monkeypatch):
    from space_env.scaling import benchmark

    w = small_workload()
    w["repeats"] = 2
    original_run = benchmark.subprocess.run

    def timeout(argv, **kwargs):
        if "space_env.scaling.worker" not in argv:
            return original_run(argv, **kwargs)
        StageReporter(argv[-1], digest(json.loads(Path(argv[-3]).read_text()))).emit(
            "initialize"
        )
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    result = run_scale(
        {"id": "timeouts", "workers": 2, "workloads": [w]}, tmp_path / "scale"
    )
    assert not result["ok"]
    assert [r["repeat"] for r in result["rows"]] == [0, 1]
    assert result["summary"][w["id"]]["failed_repeats"] == 2
    assert all(r["worker_progress"]["stage"] == "initialize" for r in result["rows"])
    assert len(list((tmp_path / "scale/workers").glob("*/progress.json"))) == 2


def test_completed_diagnostic_without_native_result_cannot_pass(tmp_path, monkeypatch):
    from space_env.scaling import benchmark

    w = Workload.model_validate(small_workload())
    progress = tmp_path / "progress.json"

    def missing_result(argv, **kwargs):
        reporter = StageReporter(progress, digest(w.model_dump(mode="json")))
        reporter.emit("export", status="completed")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(benchmark.subprocess, "run", missing_result)
    row = _execute_workload(w, 0, progress_path=progress)
    assert not row["passed"] and row["error"] is not None
    assert row["worker_progress"]["status"] == "completed"
