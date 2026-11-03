"""Native ensemble denominator, restart, audit and failure-retention contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from space_env.scaling import ensemble as module
from space_env.scaling.benchmark import ScaleSpec, compare_semantics, summarize
from space_env.scaling.ensemble import (
    prepare_ensemble_qualification,
    run_ensemble_qualification,
    summarize_ensemble_qualification,
    verify_ensemble_qualification,
)
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from tests.test_calibration import ensemble_spec, parameter
from tests.test_calibration import scenario as host


@pytest.fixture
def spec():
    return {
        "id": "test-native-ensemble",
        "workers": 1,
        "resume_after_decisions": 1,
        "workload": {
            "id": "host",
            "family": "synthetic-coast",
            "source": "integration fixture, no service claim",
            "scenario": host.__wrapped__(),
            "task": {
                "id": "coast",
                "roles": {"a": {"actors": ["a"]}},
                "decision_sec": 0.25,
                "max_decision_sec": 0.25,
                "max_decisions": 2,
                "runtime_audit": "compact",
            },
            "policy": {"name": "wait"},
            "min_completion_fraction": 0.0,
        },
        "sensitivity": ensemble_spec(
            samples=2, parameters=[parameter(target="battery_capacity_scale")]
        ),
    }


def seal(value):
    value.pop("sha256", None)
    value["sha256"] = digest(value)
    return value


def scale_result(
    prepared, *, missing=False, error=False, different=False, gate_failure=False
):
    scale = ScaleSpec.model_validate(prepared["scale_manifest"])
    rows = []
    for w in scale.workloads:
        for repeat in range(w.repeats):
            if missing and w.id == "m000-cold":
                continue
            gates = dict.fromkeys(
                [
                    "native_replay",
                    "quality",
                    "step_latency_budget",
                    "execution_budget",
                    "checkpoint_budget",
                    "peak_rss_budget",
                    "completion",
                ],
                True,
            )
            if gate_failure and w.id.startswith("m000"):
                gates["completion"] = False
            row = {
                "id": w.id,
                "repeat": repeat,
                "error": None,
                "passed": all(gates.values()),
                "gates": gates,
                "metrics": {"completion_fraction": 0.0 if gate_failure else 1.0},
                "execution_sec": 1.0,
                "peak_rss_bytes": 1000,
                "native_backend_sha256": digest(w.comparison_group or w.id),
                "trace_sha256": digest(w.comparison_group or w.id),
            }
            if different and w.id == "m000-cold":
                row["trace_sha256"] = digest("different-continuation")
            if error and w.id == "m000-cold":
                row = {
                    "id": w.id,
                    "repeat": repeat,
                    "passed": False,
                    "error": {
                        "type": "TimeoutExpired",
                        "message": "retained timeout fixture",
                    },
                }
            rows.append(row)
    summary, comparisons = summarize(scale, rows), compare_semantics(scale, rows)
    return seal(
        {
            "manifest_sha256": digest(scale.model_dump(mode="json", by_alias=True)),
            "rows": rows,
            "summary": summary,
            "semantic_comparisons": comparisons,
            "ok": all(r["passed"] for r in rows)
            and all(v["passed"] for v in summary.values())
            and all(v["passed"] for v in comparisons.values()),
        }
    )


def test_preparation_owns_inputs_and_only_scenarios_change(spec):
    before = copy.deepcopy(spec)
    prepared = prepare_ensemble_qualification(spec)
    assert spec == before
    assert len(prepared["scale_manifest"]["workloads"]) == 6
    for w in prepared["scale_manifest"]["workloads"]:
        assert w["task"] == spec["workload"]["task"]
        assert w["scenario"]["seed"] == spec["workload"]["scenario"]["seed"]
        assert "draws" not in w and "source_spec" not in w
    prepared["scale_manifest"]["workloads"][0]["task"]["id"] = "mutated"
    assert spec == before


@pytest.mark.parametrize(
    "update",
    [
        {"workers": True},
        {"workers": 5},
        {"resume_after_decisions": 2},
        {"resume_after_decisions": True},
        {"id": "../escape"},
    ],
)
def test_invalid_run_contract(spec, update):
    spec.update(update)
    with pytest.raises(ValueError):
        prepare_ensemble_qualification(spec)


@pytest.mark.parametrize(
    "field,value", [("resume_after_decisions", 1), ("comparison_group", "foreign")]
)
def test_base_cannot_override_pair_bindings(spec, field, value):
    spec["workload"][field] = value
    with pytest.raises(ValueError, match="owns"):
        prepare_ensemble_qualification(spec)


@pytest.mark.parametrize(
    "samples,repeats,restart", [(32, 1, 1), (64, 1, None), (8, 10, 1)]
)
def test_declared_budget_counts_baseline_restart_repeats_and_rejections(
    spec, samples, repeats, restart
):
    spec["sensitivity"]["samples"] = samples
    spec["workload"]["repeats"] = repeats
    spec["resume_after_decisions"] = restart
    with pytest.raises(ValueError, match="execution slots"):
        prepare_ensemble_qualification(spec)


def test_unpaired_mode_and_repeats(spec):
    spec["resume_after_decisions"] = None
    spec["workload"]["repeats"] = 2
    p = prepare_ensemble_qualification(spec)
    r = summarize_ensemble_qualification(p, scale_result(p))
    assert r["passed"] and r["expected_submitted_executions"] == 6
    assert all(m["restart_semantics"] is None for m in r["members"])


@pytest.mark.parametrize("mode", ["missing", "error", "different", "gate_failure"])
def test_all_native_failures_remain_in_denominator(spec, mode):
    p = prepare_ensemble_qualification(spec)
    result = summarize_ensemble_qualification(p, scale_result(p, **{mode: True}))
    assert not result["passed"]
    assert result["attempted_variants"] == 2
    assert result["counts"] == {
        "passed": 1,
        "native_failed": 1,
        "structural_rejected": 0,
    }
    assert result["variant_pass_fraction_all_attempts"] == 0.5
    assert result["members"][0]["expected_executions"] == 2


def test_structural_failures_not_filtered_or_executed(spec):
    spec["sensitivity"]["parameters"] = [
        parameter(target="solar_efficiency_scale", low=100.0, high=101.0)
    ]
    p = prepare_ensemble_qualification(spec)
    assert len(p["scale_manifest"]["workloads"]) == 2
    result = summarize_ensemble_qualification(p, scale_result(p))
    assert not result["passed"] and result["baseline"]["status"] == "passed"
    assert result["counts"]["structural_rejected"] == 2
    assert result["variant_pass_fraction_all_attempts"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "duplicate",
        "unknown",
        "bool_repeat",
        "signature",
        "missing_gate",
        "summary",
        "manifest",
        "overall",
    ],
)
def test_tampered_or_contradictory_evidence_rejected(spec, mutation):
    p = prepare_ensemble_qualification(spec)
    r = scale_result(p)
    if mutation == "checksum":
        r["ok"] = False
    else:
        if mutation == "duplicate":
            r["rows"].append(copy.deepcopy(r["rows"][0]))
        if mutation == "unknown":
            r["rows"][0]["id"] = "not-submitted"
        if mutation == "bool_repeat":
            r["rows"][0]["repeat"] = False
        if mutation == "signature":
            r["rows"][0]["native_backend_sha256"] = "invalid"
        if mutation == "missing_gate":
            r["rows"][0]["gates"].pop("quality")
        if mutation == "summary":
            r["summary"]["baseline-warm"]["passed"] = False
        if mutation == "manifest":
            r["manifest_sha256"] = digest("foreign")
        if mutation == "overall":
            r["ok"] = False
        seal(r)
    with pytest.raises(ValueError):
        summarize_ensemble_qualification(p, r)


def test_prepared_resealed_modification_still_rejected(spec):
    p = prepare_ensemble_qualification(spec)
    r = scale_result(p)
    p["compiled"]["members"][0]["draws"]["battery"] = 0.001
    seal(p)
    with pytest.raises(ValueError, match="compilation"):
        summarize_ensemble_qualification(p, r)


def fake_runner(prepared):
    def run(manifest, output):
        output = Path(output)
        output.mkdir()
        value = scale_result(prepared)
        for name, data in [
            ("manifest.json", {"manifest": manifest, "sha256": digest(manifest)}),
            ("result.json", value),
            ("preflight.json", {"manifest_sha256": digest(manifest)}),
            (
                "progress.json",
                {
                    "manifest_sha256": digest(manifest),
                    "rows": value["rows"],
                    "finished_repeats": len(value["rows"]),
                    "expected_repeats": sum(
                        w["repeats"] for w in manifest["workloads"]
                    ),
                },
            ),
        ]:
            atomic_write(output / name, canonical_json(data))
        return value

    return run


def test_artifact_audit_does_not_claim_native_reexecution(spec, tmp_path, monkeypatch):
    p = prepare_ensemble_qualification(spec)
    monkeypatch.setattr(module, "run_scale", fake_runner(p))
    r = run_ensemble_qualification(spec, tmp_path / "run")
    audit = verify_ensemble_qualification(tmp_path / "run")
    assert r["passed"] and audit["verified"] and audit["qualification_passed"]
    assert not audit["native_reexecuted"]
    (tmp_path / "run" / "native" / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_ensemble_qualification(tmp_path / "run")


@pytest.mark.parametrize("unsafe", ["../outside", "/tmp/outside", "native/link"])
def test_unsafe_artifact_inventory_rejected(spec, tmp_path, monkeypatch, unsafe):
    p = prepare_ensemble_qualification(spec)
    monkeypatch.setattr(module, "run_scale", fake_runner(p))
    run_ensemble_qualification(spec, tmp_path / "run")
    path = tmp_path / "run" / "result.json"
    r = json.loads(path.read_text())
    if unsafe == "native/link":
        (tmp_path / "run" / "native" / "link").symlink_to(
            tmp_path / "run" / "prepared.json"
        )
    r["artifacts_sha256"][unsafe] = digest("x")
    atomic_write(path, canonical_json(seal(r)))
    with pytest.raises(ValueError, match="unsafe"):
        verify_ensemble_qualification(tmp_path / "run")


def test_nonempty_output_and_invalid_contract_leave_files_untouched(spec, tmp_path):
    (tmp_path / "sentinel").write_text("original")
    with pytest.raises(ValueError, match="empty"):
        run_ensemble_qualification(spec, tmp_path)
    assert (tmp_path / "sentinel").read_text() == "original"
    spec["sensitivity"]["samples"] = 64
    with pytest.raises(ValueError):
        run_ensemble_qualification(spec, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_runner_interruption_keeps_prepared_but_no_final_success(
    spec, tmp_path, monkeypatch
):
    def broken(*args):
        raise OSError("intentional retained runner interruption")

    monkeypatch.setattr(module, "run_scale", broken)
    with pytest.raises(OSError):
        run_ensemble_qualification(spec, tmp_path / "run")
    assert (tmp_path / "run" / "prepared.json").exists()
    assert not (tmp_path / "run" / "result.json").exists()


def test_cli_input_budget_and_error_exit(tmp_path, capsys):
    from space_env.cli import main

    source = tmp_path / "too-large.json"
    source.write_bytes(b" " * 2_000_001)
    with pytest.raises(SystemExit) as exc:
        main(["qualify-ensemble", str(source), str(tmp_path / "run")])
    assert exc.value.code == 2
    assert (
        json.loads(capsys.readouterr().out)["error"]["code"]
        == "ensemble_contract_error"
    )
    assert not (tmp_path / "run").exists()


def test_completion_gate_must_be_explicit(spec):
    spec["workload"].pop("min_completion_fraction")
    with pytest.raises(ValueError, match="explicitly declared"):
        prepare_ensemble_qualification(spec)


@pytest.mark.parametrize(
    "name", ["native/manifest.json", "native/progress.json", "native/preflight.json"]
)
def test_resealed_conflicting_native_artifacts_rejected(
    spec, tmp_path, monkeypatch, name
):
    p = prepare_ensemble_qualification(spec)
    monkeypatch.setattr(module, "run_scale", fake_runner(p))
    run_ensemble_qualification(spec, tmp_path / "run")
    target = tmp_path / "run" / name
    target.write_text("{}")
    receipt = tmp_path / "run" / "result.json"
    r = json.loads(receipt.read_text())
    import hashlib

    r["artifacts_sha256"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_write(receipt, canonical_json(seal(r)))
    with pytest.raises(ValueError, match="manifest|progress|preflight"):
        verify_ensemble_qualification(tmp_path / "run")


def test_real_native_coast_workers_restore_and_verified_exports(spec, tmp_path):
    spec["sensitivity"]["samples"] = 1
    result = run_ensemble_qualification(spec, tmp_path / "native")
    assert result["passed"], result["summary"]
    assert result["summary"]["observed_executions"] == 4
    assert result["summary"]["members"][0]["restart_semantics"]["passed"]
    assert all(
        row["gates"]["native_replay"] and row["verified_export_rows"] > 0
        for row in result["summary"]["members"][0]["execution_rows"]
    )
    assert verify_ensemble_qualification(tmp_path / "native")["verified"]
