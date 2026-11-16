"""Frozen denominators, faithful baselines and honest incomplete-track receipts."""

import copy
import json

import pytest

from space_env.llm.episode import seal
from space_env.validation import evidence_benchmark as evidence


def test_frozen_full_matrix_and_no_ranking():
    full = evidence.benchmark_manifest("full")
    assert len(full["missions"]) == 28
    assert len(full["policies"]) == 3
    assert full["scale"]["actors"] == [2, 4, 8, 16]
    assert full["scale"]["repeats"] == 2
    assert len(evidence.benchmark_manifest("smoke")["missions"]) == 6
    with pytest.raises(ValueError, match="profile"):
        evidence.benchmark_manifest("invented")


@pytest.fixture(scope="module")
def small_report(tmp_path_factory):
    from space_env.tasks.catalog import mission_recipe

    directory = tmp_path_factory.mktemp("evidence") / "run"
    original_recipes = evidence._recipes
    original_scale = evidence._scaling
    from space_env.validation import ai_reference_suite

    original_physics = ai_reference_suite.run_suite
    case = mission_recipe("finite/finite-j2")

    def unavailable(*args, **kwargs):
        raise RuntimeError("deliberate test failure retained")

    evidence._recipes = lambda: {"finite/finite-j2": copy.deepcopy(case)}
    evidence._scaling = unavailable
    ai_reference_suite.run_suite = unavailable
    try:
        report = evidence.run(directory)
        yield directory, report
    finally:
        evidence._recipes = original_recipes
        evidence._scaling = original_scale
        ai_reference_suite.run_suite = original_physics


def test_native_matrix_and_failures_survive_audit(small_report):
    directory, report = small_report
    assert report["tracks"]["physics"]["status"] == "error"
    assert report["tracks"]["scaling"]["status"] == "error"
    assert report["tracks"]["missions"]["status"] == "completed"
    assert not report["superiority_established"]
    assert report["learning"]["status"] == "not_measured"
    audit = evidence.audit(directory)
    assert audit["native_episodes_replayed"] == 3
    assert not list(directory.rglob("*.jsonl"))
    with pytest.raises(ValueError, match="new"):
        evidence.run(directory)


def test_resealed_missing_denominator_is_rejected(small_report):
    directory, _ = small_report
    missions_path = directory / "missions.json"
    report_path = directory / "report.json"
    original_missions = missions_path.read_text()
    original_report = report_path.read_text()
    missions = json.loads(original_missions)
    missions["rows"].pop()
    report = json.loads(original_report)
    from space_env.simulation.snapshot import digest

    report["tracks"]["missions"]["sha256"] = digest(missions)
    missions_path.write_text(json.dumps(missions))
    report_path.write_text(
        json.dumps(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    try:
        with pytest.raises(ValueError, match="denominator"):
            evidence.audit(directory)
    finally:
        missions_path.write_text(original_missions)
        report_path.write_text(original_report)


def test_resealed_superiority_claim_is_rejected(small_report):
    directory, _ = small_report
    path = directory / "report.json"
    original = path.read_text()
    report = json.loads(original)
    report["superiority_established"] = True
    path.write_text(
        json.dumps(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    try:
        with pytest.raises(ValueError, match="claims"):
            evidence.audit(directory)
    finally:
        path.write_text(original)


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        evidence._read(tmp_path, "../outside.json")
