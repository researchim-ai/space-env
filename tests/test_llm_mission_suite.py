"""Frozen workload families and replay-bound public/privileged comparisons."""

import copy
import json

import pytest

from space_env.llm.benchmark import POLICIES
from space_env.llm.episode import seal
from space_env.llm.mission_suite import (
    FAMILIES,
    audit,
    compile_suite,
    run_suite,
    summarize,
    validate_suite,
)
from space_env.planning.network_benchmark import geometry_key
from space_env.simulation.snapshot import canonical_json


def error_rows(suite, selected):
    rows = []
    for c in suite["cases"]:
        if c["id"] not in selected:
            continue
        for policy in POLICIES:
            row = {k: c[k] for k in ("group", "split", "family", "geometry_sha256")}
            row.update(
                case=c["id"],
                policy=policy,
                status="error",
                metrics=None,
                request_total=len(c["configuration"]["mission"]["requests"]),
                priority_total=sum(
                    r["priority"] for r in c["configuration"]["mission"]["requests"]
                ),
            )
            rows.append(row)
    return rows


def test_literal_families_are_repeatable_and_geometry_is_split_disjoint():
    suite = compile_suite()
    assert suite == compile_suite() == validate_suite(suite)
    assert len(suite["cases"]) == 36
    assert {c["family"] for c in suite["cases"]} == set(FAMILIES)
    by_split = {
        split: {c["geometry_sha256"] for c in suite["cases"] if c["split"] == split}
        for split in ("train", "development", "test")
    }
    assert all(len(values) == 2 for values in by_split.values())
    assert not (
        by_split["train"] & by_split["test"]
        or by_split["train"] & by_split["development"]
        or by_split["development"] & by_split["test"]
    )
    assert all(
        c["geometry_sha256"] == geometry_key(c["configuration"]) for c in suite["cases"]
    )
    assert "not blind holdout" in suite["scope"]


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_changes_its_declared_physical_mechanism(family):
    c = next(c for c in compile_suite()["cases"] if c["family"] == family)
    cfg = c["configuration"]
    m = cfg["mission"]
    if family == "relay-chain":
        assert len(cfg["links"]) == 2 and cfg["max_hops"] == 2
    elif family == "cpu-backlog":
        assert [r["processing_sec"] for r in m["requests"]] == [35.0, 20.0, 45.0]
    elif family == "memory-pressure":
        assert m["memory_mbit"] == 1.2
    elif family == "station-blackout":
        assert m["outages"][0]["end_sec"] == 120.0
    elif family == "link-blackout":
        assert cfg["links"][0]["outages"] == [[0.0, 90.0]]
    else:
        assert [r["deadline_sec"] for r in m["requests"]] == [30.0, 45.0, 60.0]


def test_failures_stay_unknown_and_denominators_include_them():
    suite = compile_suite()
    selected = [c["id"] for c in suite["cases"]]
    result = summarize(suite, selected, error_rows(suite, selected))
    assert len(result["aggregates"]) == 54
    assert all(
        a["cases"] == 2 and a["value_lower"] == 0 and a["value_upper"] == 1
        for a in result["aggregates"]
    )
    assert not result["blind_holdout_claim"] and not result["external_llm_measured"]
    assert all(
        a["privileged"] == (a["policy"] == "reference_schedule")
        for a in result["aggregates"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "split",
        "group",
        "priority",
        "requests",
        "nan",
        "negative",
        "condition",
    ],
)
def test_bad_comparison_cells_are_rejected(fault):
    suite = compile_suite()
    selected = [suite["cases"][0]["id"]]
    rows = error_rows(suite, selected)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault in {"split", "group"}:
        rows[0][fault] = "wrong"
    elif fault == "priority":
        rows[0]["priority_total"] = 1
    elif fault == "requests":
        rows[0]["request_total"] = 999
    elif fault == "condition":
        rows[0]["family"] = "fiction"
    else:
        rows[0].update(
            status="completed",
            metrics={
                "weighted_fulfilled": float("nan") if fault == "nan" else -1,
                "fulfilled": 0,
                "total": 3,
                "success": False,
            },
        )
    with pytest.raises(ValueError):
        summarize(suite, selected, rows)


def test_modified_but_resealed_suite_is_not_new_version():
    suite = compile_suite()
    suite["cases"][0]["configuration"]["mission"]["memory_mbit"] = 1000.0
    suite = seal({k: v for k, v in suite.items() if k != "sha256"})
    with pytest.raises(ValueError, match="Frozen"):
        validate_suite(suite)


@pytest.fixture(scope="module")
def native_comparison(tmp_path_factory):
    root = tmp_path_factory.mktemp("family-run") / "run"
    selected = [compile_suite()["cases"][0]["id"]]
    report = run_suite(root, selected=selected)
    assert all(r["status"] == "completed" for r in report["rows"])
    return root, report


def test_actual_native_comparison_and_independent_replay(native_comparison):
    root, report = native_comparison
    assert len(report["rows"]) == 3
    assert len({r["initial_checkpoint_sha256"] for r in report["rows"]}) == 1
    assert (
        next(r for r in report["rows"] if r["policy"] == "wait")["metrics"]["fulfilled"]
        == 0
    )
    assert audit(root)["replayed"] == 3


@pytest.mark.parametrize(
    "fault", ["metric", "artifact", "split", "origin", "suite-file", "checkpoint"]
)
def test_resealed_report_cannot_invent_native_results(
    native_comparison, tmp_path, fault
):
    import shutil

    source, report = native_comparison
    root = tmp_path / "run"
    shutil.copytree(source, root)
    report = copy.deepcopy(report)
    if fault == "metric":
        report["rows"][0]["metrics"]["weighted_fulfilled"] = 1
    elif fault == "artifact":
        report["rows"][0]["artifact"] = "../elsewhere.json"
    elif fault == "split":
        report["rows"][0]["split"] = "test"
    elif fault == "checkpoint":
        report["rows"][0]["initial_checkpoint_sha256"] = "0" * 64
    elif fault == "suite-file":
        (root / "suite.json").write_text("{}")
    else:
        p = root / report["rows"][0]["artifact"]
        record = json.loads(p.read_text())
        record["configuration"]["policy_origin"] = "external_llm"
        record = seal({k: v for k, v in record.items() if k != "sha256"})
        p.write_text(canonical_json(record))
    (root / "report.json").write_text(
        canonical_json(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        audit(root)


def test_bad_selection_never_creates_output(tmp_path):
    for selected in ([], ["unknown"], [compile_suite()["cases"][0]["id"]] * 2):
        with pytest.raises(ValueError):
            run_suite(tmp_path / "not-created", selected=selected)
        assert not (tmp_path / "not-created").exists()


def test_boolean_is_not_equal_to_integer_in_frozen_config():
    suite = compile_suite()
    assert suite["cases"][0]["configuration"]["mission"]["seed"] == 0
    suite["cases"][0]["configuration"]["mission"]["seed"] = False
    # A dictionary equality alone treats False == 0; the canonical contract must not.
    with pytest.raises(ValueError, match="Frozen"):
        validate_suite(suite)
