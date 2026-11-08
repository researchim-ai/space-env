import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from space_env.validation import network_evidence as evidence


def test_linear_rf_oracle_gates_thresholds_and_retains_full_case_set():
    r = evidence.rf_evidence()
    assert r["passed"] and len(r["cases"]) == 16
    assert not r["hardware_calibrated"]
    rows = {c["id"]: c for c in r["cases"]}
    assert rows["threshold-above"]["actual"]["selected_mode"] == "250000"
    for name in (
        "threshold-below",
        "doppler-outside",
        "narrow-bandwidth",
        "rate-cap",
        "spectrum-outside",
    ):
        assert rows[name]["actual"]["selected_mode"] is None
    assert max(max(c["absolute_errors"].values()) for c in r["cases"]) < 1e-10


def test_linux_memory_measurement_uses_post_exec_mm_peak(monkeypatch):
    monkeypatch.setattr(evidence.sys, "platform", "linux")
    monkeypatch.setattr(
        evidence.Path, "read_text", lambda self: "VmRSS:\t 100 kB\nVmHWM:\t 512 kB\n"
    )
    assert evidence.peak_rss() == (512 * 1024, "linux-proc-VmHWM-after-exec")
    monkeypatch.setattr(evidence.Path, "read_text", lambda self: "VmRSS:\t 100 kB\n")
    with pytest.raises(RuntimeError):
        evidence.peak_rss()


def test_rf_evidence_detects_a_production_budget_regression(monkeypatch):
    original = evidence.rf_budget

    def faulty(*args, **kwargs):
        row = original(*args, **kwargs)
        row["cn0_db_hz"] += 0.25
        return row

    monkeypatch.setattr(evidence, "rf_budget", faulty)
    report = evidence.rf_evidence()
    assert not report["passed"]
    assert all(not c["passed"] for c in report["cases"])


def test_workloads_are_fixed_bounded_and_not_identical_repeats():
    from space_env.api.network_workspace import NetworkConfiguration

    cases = evidence.workloads()
    assert [len(c["configuration"]["mission"]["requests"]) for c in cases] == [
        1,
        4,
        16,
        1,
    ]
    assert len({c["id"] for c in cases}) == 4
    for c in cases:
        NetworkConfiguration.model_validate(c["configuration"])


@pytest.fixture(scope="module")
def native_row():
    case = evidence.workloads()[0]
    row = evidence.run_scale_case(case, timeout_sec=60)
    assert row["status"] == "completed", row
    return case, row


def test_fresh_process_native_measurement_and_input_source_binding(native_row):
    case, row = native_row
    assert evidence.validate_row(row, case) == row
    assert row["native_fulfilled"] == row["total"] == 1
    assert row["sdk_steps"] == 30
    assert row["environment"]["threads"]["OMP_NUM_THREADS"] == "1"


@pytest.mark.parametrize(
    "fault", ["seal", "identity", "source", "denominator", "timing", "count"]
)
def test_tampered_worker_rows_are_rejected(native_row, fault):
    case, row = copy.deepcopy(native_row)
    if fault == "seal":
        row["sha256"] = "0" * 64
    else:
        if fault == "identity":
            row["input_sha256"] = "0" * 64
        elif fault == "source":
            row["source_sha256"] = "0" * 64
        elif fault == "denominator":
            row["total"] += 1
        elif fault == "timing":
            row["step_sec"]["p95"] = -1.0
        else:
            row["native_fulfilled"] = True
        row = evidence.seal({k: v for k, v in row.items() if k != "sha256"})
    with pytest.raises(ValueError):
        evidence.validate_row(row, case)


@pytest.mark.parametrize(
    "failure", ["timeout", "exit", "malformed", "non_object", "os"]
)
def test_worker_failures_are_rows_in_the_full_denominator(monkeypatch, failure):
    case = evidence.workloads()[2]

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("worker", 1)
        if failure == "os":
            raise OSError("cannot spawn")
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0,
            stdout="[]" if failure == "non_object" else "not json",
            stderr="worker failed",
        )

    monkeypatch.setattr(evidence.subprocess, "run", run)
    row = evidence.run_scale_case(case, timeout_sec=1)
    assert row["status"] == ("timeout" if failure == "timeout" else "error")
    assert row["total"] == 16 and row["native_fulfilled"] is None


def test_subprocess_uses_importable_module_and_validates_timeout(monkeypatch):
    def run(argv, **kw):
        assert argv[2] == "space_env.validation.network_evidence"
        assert kw["env"]["OPENBLAS_NUM_THREADS"] == "1"
        json.loads(kw["input"])
        return SimpleNamespace(returncode=1, stderr="test only", stdout="")

    monkeypatch.setattr(evidence.subprocess, "run", run)
    assert evidence.run_scale_case(evidence.workloads()[0])["status"] == "error"
    for timeout in (True, 0, 601):
        with pytest.raises(ValueError):
            evidence.run_scale_case(evidence.workloads()[0], timeout_sec=timeout)


def artifact_set(tmp_path, native_row):
    case, row = copy.deepcopy(native_row)
    inputs = evidence.seal({"cases": [case], "repeats": 1})
    rf = evidence.seal({"passed": True})
    analytical = evidence.seal({"passed": True})
    report = evidence.seal(
        {
            "schema": "space-env.network-evidence/1",
            "inputs_sha256": inputs["sha256"],
            "rf_sha256": rf["sha256"],
            "analytical_sha256": analytical["sha256"],
            "model_sources": evidence.source_manifest(),
            "rows": [{"repeat": 0, "result": row}],
            "passed": True,
            "external_forces_included": False,
            "external_forces_sha256": None,
        }
    )
    data = {
        "inputs.json": inputs,
        "rf.json": rf,
        "analytical.json": analytical,
        "report.json": report,
        case["id"] + "-0.json": row,
    }
    for name, obj in data.items():
        (tmp_path / name).write_text(json.dumps(obj))
    return data


def test_saved_evidence_audit_binds_all_rows_and_sources(tmp_path, native_row):
    artifact_set(tmp_path, native_row)
    result = evidence.audit_evidence(tmp_path)
    assert result["consistent"] and result["source_matches_current"]
    assert result["rows"] == 1
    assert "not native replay" in result["scope"]


@pytest.mark.parametrize(
    "fault", ["missing_row", "duplicate_row", "rf_tamper", "outcome", "row_file"]
)
def test_audit_rejects_missing_denominator_and_artifact_tampering(
    tmp_path, native_row, fault
):
    data = artifact_set(tmp_path, native_row)
    report = data["report.json"]
    if fault == "missing_row":
        report["rows"] = []
    elif fault == "duplicate_row":
        report["rows"] *= 2
    elif fault == "outcome":
        report["passed"] = False
    elif fault == "rf_tamper":
        (tmp_path / "rf.json").write_text(
            json.dumps({**data["rf.json"], "passed": False})
        )
    else:
        row = data[report["rows"][0]["result"]["case_id"] + "-0.json"]
        row["native_fulfilled"] = 0
        (tmp_path / (row["case_id"] + "-0.json")).write_text(
            json.dumps(evidence.seal({k: v for k, v in row.items() if k != "sha256"}))
        )
    (tmp_path / "report.json").write_text(
        json.dumps(evidence.seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        evidence.audit_evidence(tmp_path)
