"""Fixed denominators, crash recovery and native source-bound study checkpoints."""

import copy
import json

import pytest

from space_env.llm import qualification as q
from space_env.llm.episode import seal
from space_env.simulation.snapshot import canonical_json


def error_rows(plan):
    return [
        {
            "case": c["id"],
            "method": m,
            "split": c["split"],
            "family": c["family"],
            "status": "error",
            "metrics": None,
        }
        for c in plan["suite"]["cases"]
        for m in plan["methods"]
    ]


def test_fixed_denominators_keep_pending_and_errors_unknown():
    plan = q._plan()
    empty = q.summarize(plan, [])
    assert empty["expected_cells"] == 72 and empty["paired"]["unknown"] == 36
    for row in empty["aggregates"]:
        assert (
            row["pending"] == row["cases"]
            and row["value_lower"] == 0
            and row["value_upper"] == 1
        )
    failed = q.summarize(plan, error_rows(plan))
    assert failed["runtime_errors"] == 72 and failed["replayed"] == 0
    assert all(
        r["value_lower"] == 0 and r["value_upper"] == 1 and r["pending"] == 0
        for r in failed["aggregates"]
    )
    assert failed["geometry_groups"] == 6 and not failed["blind_holdout_claim"]


@pytest.mark.parametrize(
    "fault", ["duplicate", "unknown", "split", "family", "metrics"]
)
def test_invalid_partial_cells_are_rejected(fault):
    plan = q._plan()
    rows = error_rows(plan)
    if fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "unknown":
        rows[0]["case"] = "invented"
    elif fault in {"split", "family"}:
        rows[0][fault] = "invented"
    else:
        rows[0]["metrics"] = {"weighted_fulfilled": 0}
    with pytest.raises(ValueError):
        q.summarize(plan, rows)


def test_cross_split_aggregation_uses_all_frozen_cases():
    plan = q._plan()
    rows = error_rows(plan)
    c = plan["suite"]["cases"][0]
    rows[1].update(
        status="completed",
        planned=3,
        metrics={"success": True, "weighted_fulfilled": 18, "fulfilled": 3, "total": 3},
    )
    summary = q.summarize(plan, rows)
    aggregate = next(
        r
        for r in summary["aggregates"]
        if r["axis"] == "overall" and r["method"] == "sequence_lanes"
    )
    assert aggregate["cases"] == 36 and aggregate["value_lower"] == pytest.approx(
        1 / 36
    )
    assert summary["paired"]["unknown"] == 36  # baseline error is not a measured zero
    assert c["split"] == "train"


@pytest.fixture(scope="module")
def native_prefix(tmp_path_factory):
    root = tmp_path_factory.mktemp("qualification") / "study"
    first = q.run_study(root, max_cases=1)
    assert not first["complete"] and first["summary"]["recorded_cells"] == 2
    assert q.audit_study(root, require_complete=False)["replayed"] == 2
    original = copy.deepcopy(json.loads((root / "journal.json").read_text()))
    # Clean resume must reuse the first case unchanged and add the next one.
    second = q.run_study(root, resume=True, max_cases=1)
    journal = json.loads((root / "journal.json").read_text())
    assert journal["attempts"][0] == original["attempts"][0]
    assert second["completed_cases"] == 2 and not second["complete"]
    return root


def test_native_prefix_is_not_full_qualification(native_prefix):
    proof = q.audit_study(native_prefix, require_complete=False)
    assert proof["replayed"] == 4 and not proof["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        q.audit_study(native_prefix)
    assert not (native_prefix / "report.json").exists()


@pytest.mark.parametrize(
    "fault", ["source", "aggregate", "child", "hidden-completed", "path"]
)
def test_resealed_corrupt_checkpoint_is_rejected(native_prefix, tmp_path, fault):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(native_prefix, root)
    if fault == "source":
        p = root / "study.json"
        value = json.loads(p.read_text())
        value["model_sources"] = {}
    elif fault == "aggregate":
        p = root / "progress.json"
        value = json.loads(p.read_text())
        value["summary"]["replayed"] = 72
    else:
        p = root / "journal.json"
        value = json.loads(p.read_text())
        if fault == "child":
            value["attempts"][0]["report_sha256"] = "0" * 64
        elif fault == "hidden-completed":
            value["attempts"][0]["status"] = "interrupted"
        else:
            value["attempts"][0]["case"] = "../../outside"
    p.write_text(
        canonical_json(seal({k: v for k, v in value.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        q.audit_study(root, require_complete=False)


def test_writer_lease_serializes_processes(tmp_path):
    with (
        q._lease(tmp_path),
        pytest.raises(ValueError, match="busy"),
        q._lease(tmp_path),
    ):
        pytest.fail("second writer acquired the same lease")


def test_crash_before_attempt_directory_creation_is_recorded_and_resumable(
    tmp_path, monkeypatch
):
    original = q.run_case

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(q, "run_case", interrupt)
    root = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt):
        q.run_study(root, max_cases=1)
    assert (
        json.loads((root / "journal.json").read_text())["attempts"][0]["status"]
        == "running"
    )
    with pytest.raises(ValueError, match="resume"):
        q.audit_study(root, require_complete=False)
    monkeypatch.setattr(q, "run_case", original)
    result = q.run_study(root, resume=True, max_cases=1)
    assert result["interrupted_attempts"] == 1 and result["completed_cases"] == 1
    assert q.audit_study(root, require_complete=False)["replayed"] == 2


def test_resume_rejects_source_revision_mismatch_before_running(
    native_prefix, tmp_path, monkeypatch
):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(native_prefix, root)
    expected = q._plan()
    expected["scope"] = "different-source"
    expected = seal({k: v for k, v in expected.items() if k != "sha256"})
    monkeypatch.setattr(q, "_plan", lambda: expected)
    with pytest.raises(ValueError, match="different suite/source"):
        q.run_study(root, resume=True)


def test_finished_error_cells_are_never_retried_for_a_better_score(
    tmp_path, monkeypatch
):
    from space_env.workspace import MissionWorkspace

    original = MissionWorkspace.network

    def fail(*args, **kwargs):
        raise RuntimeError("injected native setup failure")

    monkeypatch.setattr(MissionWorkspace, "network", fail)
    root = tmp_path / "study"
    first = q.run_study(root, max_cases=1)
    assert first["summary"]["runtime_errors"] == 2 and first["completed_cases"] == 1
    first_report = json.loads(
        (
            root
            / "cases"
            / q._plan()["suite"]["cases"][0]["id"]
            / "attempt-01/report.json"
        ).read_text()
    )
    assert all(r["status"] == "error" for r in first_report["rows"])
    monkeypatch.setattr(MissionWorkspace, "network", original)
    result = q.run_study(root, resume=True, max_cases=1)
    assert result["summary"]["runtime_errors"] == 2
    journal = json.loads((root / "journal.json").read_text())
    assert (
        sum(e["case"] == first_report["selected"][0] for e in journal["attempts"]) == 1
    )


def test_crash_after_child_report_is_recovered_without_rerun(tmp_path, monkeypatch):
    original = q.run_case

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("injected crash after durable child report")

    monkeypatch.setattr(q, "run_case", interrupt)
    root = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt):
        q.run_study(root, max_cases=1)
    journal = json.loads((root / "journal.json").read_text())
    assert journal["attempts"][0]["status"] == "running"
    monkeypatch.setattr(q, "run_case", original)
    result = q.run_study(root, resume=True, max_cases=1)
    assert result["completed_cases"] == 2 and result["interrupted_attempts"] == 0
    assert len(json.loads((root / "journal.json").read_text())["attempts"]) == 2


def test_unjournaled_attempt_is_not_silently_ignored(native_prefix, tmp_path):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(native_prefix, root)
    name = q._plan()["suite"]["cases"][0]["id"]
    (root / "cases" / name / "attempt-03").mkdir()
    with pytest.raises(ValueError, match="inventory"):
        q.audit_study(root, require_complete=False)
