"""Real fresh-process sequence execution and adversarial receipt binding."""

import copy

import pytest

from space_env.scaling import sequence as s


def test_frozen_profile():
    rows = s.workloads()
    assert [(r["actors"], r["requests"], r["horizon_sec"]) for r in rows] == [
        (1, 1, 60),
        (2, 3, 150),
        (3, 3, 180),
        (4, 8, 600),
        (4, 16, 1200),
    ]
    assert rows == s.workloads()
    for row in rows[3:]:
        requests = row["configuration"]["mission"]["requests"]
        assert max(r["release_sec"] for r in requests) == row["horizon_sec"] - 180
        assert max(r["deadline_sec"] for r in requests) > 180
        assert all(
            r["release_sec"] < r["deadline_sec"] <= row["horizon_sec"] for r in requests
        )
    rows[0]["configuration"]["mission"]["name"] = "mutated"
    assert rows != s.workloads()


@pytest.mark.parametrize(
    "selected", [[], ["tiny", "tiny"], ["../../x"], [True], "tiny"]
)
def test_invalid_selection(tmp_path, selected):
    with pytest.raises(ValueError, match="selection"):
        s.run(tmp_path / "out", selected=selected)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("timeout", [True, 0, -1, 301, float("nan")])
def test_invalid_budget(tmp_path, timeout):
    with pytest.raises(ValueError):
        s.run(tmp_path / "out", timeout_sec=timeout)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp("sequence-scale") / "run"
    report = s.run(root, selected=["tiny", "reference"], timeout_sec=60)
    assert [r["status"] for r in report["rows"]] == ["completed", "completed"]
    assert len({r["pid"] for r in report["rows"]}) == 2
    assert s.audit(root)["qualified"]
    return root


def test_real_native_reference(native):
    receipt = s._read(native / "reference/receipt.json")
    assert receipt["metrics"]["fulfilled"] == receipt["planned"] == 3
    assert receipt["search_nodes"] <= 50000
    assert all(
        p["wall_sec"] >= 0 and p["cumulative_peak_rss_bytes"] > 0
        for p in receipt["phases"].values()
    )
    assert s.audit(native)["full_matrix"] is False


@pytest.mark.parametrize(
    "fault",
    ["source", "selection", "duplicate", "configuration", "metric", "phase", "path"],
)
def test_resealed_tampering(native, tmp_path, fault):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(native, root)
    path = root / "report.json"
    report = s._read(path)
    if fault == "source":
        report["sourcehashfinal"] = "0" * 64
    elif fault == "selection":
        report["rows"] = report["rows"][:1]
    elif fault == "duplicate":
        report["rows"][1] = copy.deepcopy(report["rows"][0])
    elif fault == "path":
        report["rows"][0]["case"] = "../../outside"
    else:
        path = root / "reference/receipt.json"
        value = s._read(path)
        if fault == "metric":
            value["planned"] += 1
        elif fault == "phase":
            value["phases"]["compile"]["wall_sec"] = -1
        else:
            episode_path = root / "reference/episode.json"
            episode = s._read(episode_path)
            episode["configuration"]["group"] = "wrong"
            s._write(
                episode_path,
                s.seal({k: v for k, v in episode.items() if k != "sha256"}),
            )
        value = s.seal({k: v for k, v in value.items() if k != "sha256"})
        s._write(path, value)
        report["rows"][1]["receipt_sha256"] = value["sha256"]
    s._write(
        root / "report.json", s.seal({k: v for k, v in report.items() if k != "sha256"})
    )
    with pytest.raises(ValueError):
        s.audit(root)


def test_deadline_keeps_failure_denominator(tmp_path):
    root = tmp_path / "timeout"
    report = s.run(root, selected=["reference"], timeout_sec=0.001)
    assert report["rows"][0]["status"] == "timeout"
    proof = s.audit(root)
    assert proof["selected"] == 1 and proof["replayed"] == 0 and not proof["qualified"]


def test_changed_source_is_rejected(native, monkeypatch):
    monkeypatch.setattr(s, "sources", dict)
    with pytest.raises(ValueError, match="Source"):
        s.audit(native)


def test_worker_failure_preserves_diagnostics(tmp_path):
    import os
    import subprocess
    import sys

    root = tmp_path / "failed"
    (root / "tiny").mkdir(parents=True)
    s._write(root / "plan.json", s.seal({"selected": ["tiny"], "model_sources": {}}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "space_env.scaling.sequence",
            "--worker",
            str(root),
            "tiny",
        ],
        env=dict(os.environ, PYTHONPATH=str(s.Path(s.__file__).resolve().parents[2])),
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0 and b"source/workload mismatch" in result.stderr
    error = s._read(root / "tiny/error.json")
    assert error["status"] == "error" and error["error_type"] == "ValueError"
    assert not (root / "tiny/receipt.json").exists()


@pytest.mark.parametrize("workers", [True, False, 0, -1, 5, 2.0, "2", None])
def test_invalid_workers(tmp_path, workers):
    with pytest.raises(ValueError, match="Workers"):
        s.run(tmp_path / "out", workers=workers)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_supported_workers(workers):
    s._limits(60, 1_000_000_000, workers)


def test_real_parallel_native_parity(native, tmp_path):
    root = tmp_path / "parallel"
    serial = s._read(native / "report.json")
    parallel = s.run(root, selected=["tiny", "reference"], workers=2, timeout_sec=60)
    assert [r["case"] for r in parallel["rows"]] == ["tiny", "reference"]
    assert all(r["status"] == "completed" for r in parallel["rows"])
    assert len({r["pid"] for r in serial["rows"] + parallel["rows"]}) == 4
    a, b = parallel["rows"]
    assert max(a["started_offset_sec"], b["started_offset_sec"]) < min(
        a["finished_offset_sec"], b["finished_offset_sec"]
    )
    for name in ("tiny", "reference"):
        before = s._read(native / name / "receipt.json")
        after = s._read(root / name / "receipt.json")
        # The episode seal binds native checkpoint, trace, dialogue and metrics;
        # timing/PIDs live outside it and must differ across fresh processes.
        assert before["episode_sha256"] == after["episode_sha256"]
        assert before["metrics"] == after["metrics"]
    proof = s.audit(root)
    assert proof["qualified"] and proof["replayed"] == 2 and proof["workers"] == 2
    assert proof["per_worker_rss_budget_bytes"] == 1_000_000_000
    assert proof["aggregate_potential_rss_bytes"] == 2_000_000_000


def test_parallel_timeouts_keep_order_and_denominator(tmp_path):
    root = tmp_path / "timeouts"
    report = s.run(root, selected=["reference", "tiny"], workers=2, timeout_sec=0.001)
    assert [r["case"] for r in report["rows"]] == ["reference", "tiny"]
    assert [r["status"] for r in report["rows"]] == ["timeout", "timeout"]
    proof = s.audit(root)
    assert proof["selected"] == 2 and proof["replayed"] == 0 and not proof["qualified"]


def test_parallel_launch_errors_keep_denominator(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected launch failure")

    monkeypatch.setattr(s.subprocess, "Popen", fail)
    root = tmp_path / "launch-errors"
    report = s.run(root, selected=["tiny", "reference"], workers=2)
    assert [r["status"] for r in report["rows"]] == ["error", "error"]
    for name in ("tiny", "reference"):
        assert s._read(root / name / "error.json")["error_type"] == "OSError"
        assert "injected launch failure" in (root / name / "stderr.txt").read_text()
    proof = s.audit(root)
    assert proof["selected"] == 2 and proof["passed"] == 0 and not proof["qualified"]


def test_resealed_aggregate_budget_rejected(tmp_path):
    root = tmp_path / "budget"
    report = s.run(root, selected=["tiny"], workers=2, timeout_sec=0.001)
    plan = s._read(root / "plan.json")
    plan["aggregate_potential_rss_bytes"] = 1
    plan = s.seal({k: v for k, v in plan.items() if k != "sha256"})
    s._write(root / "plan.json", plan)
    report["plan_sha256"] = plan["sha256"]
    s._write(
        root / "report.json", s.seal({k: v for k, v in report.items() if k != "sha256"})
    )
    with pytest.raises(ValueError, match="binding"):
        s.audit(root)
