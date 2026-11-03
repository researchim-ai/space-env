"""Frozen oracle separation, negative numerical evidence and bounded report scope."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from space_env.simulation.snapshot import digest
from space_env.validation import envelopes as env

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads(env.FIXTURE.read_text())


def resign(payload):
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    return payload


@pytest.fixture(scope="module")
def measured():
    return env.analytic_envelope_report()


def test_complete_suite_reports_measured_points_not_general_readiness(measured):
    from space_env import __version__

    assert measured["passed"]
    assert measured["build"]["package_version"] == __version__
    assert len(measured["analytic"]["checks"]) == 103
    assert len(measured["analytic"]["evaluated_cases"]) == 10
    assert measured["qualification_status"] == "needs_external_evidence"
    assert measured["industrial_readiness"] is False
    assert not measured["analytic"]["execution_errors"]
    for row in measured["analytic"]["checks"]:
        assert row["measured"] <= row["tolerance"]
    orbit = [c for c in measured["analytic"]["evaluated_cases"] if c["kind"] == "orbit"]
    assert all(
        c["evaluated_times_sec"] == [0.0, 30.0, 60.0, 330.0, 600.0] for c in orbit
    )
    assert all(
        c["deferred_times_sec"] == [3630.0, 86400.0, 259170.0, 259200.0] for c in orbit
    )
    assert set(measured["unqualified"]) == {
        "orbit",
        "attitude",
        "power",
        "thermal",
        "payload",
        "rf",
        "dtn",
        "coupled_mission",
    }
    assert "external_formula_shared_inputs" not in measured
    assert (
        digest({k: v for k, v in measured.items() if k != "sha256"})
        == measured["sha256"]
    )


def test_interpolation_and_nodes_use_different_predeclared_budgets(measured):
    checks = measured["analytic"]["checks"]
    assert {c["solver"] for c in checks if c["kind"] == "orbit"} == {
        "DOP853",
        "Hermite/60s",
    }
    assert {c["solver"] for c in checks if c["kind"] == "conduction"} == {
        "DOP853",
        "Radau",
    }
    assert env.BUDGETS["orbit"]["position_m"] == 2.0
    assert env.BUDGETS["orbit"]["interpolated_position_m"] == 20.0


def test_frozen_generator_reproduces_bytes_without_simulator(tmp_path):
    script = ROOT / "scripts/generate_analytic_vectors.py"
    tree = ast.parse(script.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(n.name.split(".")[0] for n in node.names)
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "argparse", "hashlib", "json", "math", "pathlib"}
    output = tmp_path / "vectors.json"
    subprocess.run(
        [sys.executable, "-I", str(script), str(output)], cwd=tmp_path, check=True
    )
    assert output.read_bytes() == env.FIXTURE.read_bytes()
    assert (
        fixture()["provenance"]["generator_sha256"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )


def corrupt(payload, mutation):
    case = payload["cases"][0]
    if mutation == "schema":
        payload["schema"] += "invalid"
    elif mutation == "extra_root":
        payload["tolerance"] = 1000.0
    elif mutation == "missing_case":
        payload["cases"].pop()
    elif mutation == "duplicate_case":
        payload["cases"][1] = copy.deepcopy(case)
    elif mutation == "unknown_case":
        case["id"] = "unknown"
    elif mutation == "unknown_kind":
        case["kind"] = "unknown"
    elif mutation == "time_domain":
        case["times_sec"][1] = 1e10
    elif mutation == "constant":
        case["inputs"]["mu_m3_s2"] *= 1.001
    elif mutation == "solver_control":
        case["inputs"]["sample_sec"] = 120.0
    elif mutation == "inflated_budget":
        case["tolerance"] = 1e10
    elif mutation == "missing_expected":
        case["expected"].pop()
    elif mutation == "wrong_width":
        case["expected"][0].pop()
    elif mutation == "boolean":
        case["expected"][0][0] = True
    elif mutation == "missing_provenance":
        payload["provenance"].pop("scope")
    elif mutation == "false_independence":
        payload["provenance"]["independence"] = "independent-flight-truth"
    elif mutation == "empty_shared_inputs":
        payload["provenance"]["shared_inputs"] = []
    elif mutation == "bad_generator_hash":
        payload["provenance"]["generator_sha256"] = "z" * 64
    elif mutation == "too_many_vectors":
        case["expected"] *= 129
    else:
        raise AssertionError(mutation)
    return resign(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "extra_root",
        "missing_case",
        "duplicate_case",
        "unknown_case",
        "unknown_kind",
        "time_domain",
        "constant",
        "solver_control",
        "inflated_budget",
        "missing_expected",
        "wrong_width",
        "boolean",
        "missing_provenance",
        "false_independence",
        "empty_shared_inputs",
        "bad_generator_hash",
        "too_many_vectors",
    ],
)
def test_resigned_bad_domain_rejected_before_solver(monkeypatch, mutation):
    data = corrupt(fixture(), mutation)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid fixture reached solver")

    monkeypatch.setattr(env, "_orbit", forbidden)
    with pytest.raises(ValueError):
        env.verify_analytic_fixture(data)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), None, 10**1000]
)
def test_nonfinite_and_unbounded_scalars_rejected(value):
    data = fixture()
    data["cases"][0]["expected"][0][0] = value
    with pytest.raises(ValueError):
        env.validate_analytic_fixture(data)


@pytest.mark.parametrize("kind", ["orbit", "conduction", "radiation", "rigid", "rf"])
def test_resigned_numerically_wrong_vector_fails(kind):
    data = fixture()
    case = next(c for c in data["cases"] if c["kind"] == kind)
    case["expected"][0][0] += 100.0
    resign(data)
    env.validate_analytic_fixture(
        data
    )  # integrity alone deliberately cannot certify physics
    handlers = {
        "orbit": lambda c: env._orbit(c, False),
        "conduction": env._thermal,
        "radiation": env._thermal,
        "rigid": env._rigid,
        "rf": env._rf,
    }
    assert any(not c["passed"] for c in handlers[kind](case))


def test_quaternion_sign_equivalence():
    data = fixture()
    case = next(c for c in data["cases"] if c["id"] == "rigid-wheel")
    for row in case["expected"]:
        row[:4] = [-v for v in row[:4]]
    assert all(row["passed"] for row in env._rigid(case))


def test_solver_failure_is_failed_evidence_not_missing_case(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("injected solver failure")

    for name in ("_orbit", "_thermal", "_rigid", "_rf"):
        monkeypatch.setattr(env, name, broken)
    result = env.verify_analytic_fixture(fixture())
    assert not result["passed"] and not result["checks"]
    assert len(result["execution_errors"]) == 10
    assert all(
        c["status"] == "execution_error" and not c["evaluated_times_sec"]
        for c in result["evaluated_cases"]
    )


@pytest.mark.parametrize("kwargs", [{"long_arcs": 1}, {"external": "yes"}])
def test_report_flags_are_strict(kwargs):
    with pytest.raises(ValueError):
        env.analytic_envelope_report(**kwargs)


def test_cli_writes_failed_report_and_returns_nonzero(monkeypatch, tmp_path, capsys):
    from space_env.cli import main

    output = tmp_path / "failed.json"
    failed = {
        "schema": "space-env.physics-envelopes/1",
        "passed": False,
        "qualification_status": "needs_external_evidence",
        "analytic": {"checks": []},
        "sha256": "test",
    }
    monkeypatch.setattr(env, "analytic_envelope_report", lambda **_kw: failed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "space-env",
            "reference-qualify-envelopes",
            str(output),
            "--long",
            "--external",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert json.loads(output.read_text()) == failed
    assert not json.loads(capsys.readouterr().out)["passed"]


def test_modified_vector_without_resigning_fails_integrity():
    data = fixture()
    data["cases"][0]["expected"][0][0] += 1.0
    with pytest.raises(ValueError, match="corrupt"):
        env.validate_analytic_fixture(data)


def test_full_orbits_cover_late_interpolation_and_nodes():
    for case in fixture()["cases"][:2]:
        rows = env._orbit(case, True)
        assert len(rows) == 18
        assert {r["time_sec"] for r in rows} == set(case["times_sec"])
        assert all(r["passed"] for r in rows)
        assert any(
            r["time_sec"] == 259170.0 and r["solver"] == "Hermite/60s" for r in rows
        )


def test_external_failure_cannot_be_hidden_by_analytic_pass(monkeypatch, measured):
    from space_env.validation import gcrs_report, orbit_fixtures

    monkeypatch.setattr(
        env,
        "verify_analytic_fixture",
        lambda *_a, **_kw: copy.deepcopy(measured["analytic"]),
    )
    monkeypatch.setattr(
        orbit_fixtures,
        "verify_packaged_orbit_fixtures",
        lambda: {"passed": False, "cases": []},
    )
    monkeypatch.setattr(
        gcrs_report,
        "reference_physics_report",
        lambda **_kw: {"passed": True, "checks": [{"passed": True}]},
    )
    result = env.analytic_envelope_report(external=True)
    assert not result["passed"]
    assert result["evidence_registry"][1]["status"] == "failed"
    assert (
        result["evidence_registry"][2]["independence"]
        == "same-backend-upstream-regression"
    )
    assert result["evidence_registry"][3]["status"] == "not_run"


def test_report_owns_its_budget_and_gap_maps(monkeypatch, measured):
    monkeypatch.setattr(
        env,
        "verify_analytic_fixture",
        lambda *_a, **_kw: copy.deepcopy(measured["analytic"]),
    )
    first = env.analytic_envelope_report()
    first["budgets"]["orbit"]["position_m"] = 1e9
    first["unqualified"]["power"].clear()
    second = env.analytic_envelope_report()
    assert second["budgets"]["orbit"]["position_m"] == 2.0
    assert second["unqualified"]["power"]
    assert all(r["status"] == "not_run" for r in second["evidence_registry"][1:])


def test_source_version_is_not_replaced_by_stale_distribution_metadata(
    monkeypatch, measured
):
    from space_env import __version__

    original = env.version
    monkeypatch.setattr(
        env,
        "verify_analytic_fixture",
        lambda *_a, **_kw: copy.deepcopy(measured["analytic"]),
    )
    monkeypatch.setattr(
        env, "version", lambda name: "0.0.0" if name == "space-env" else original(name)
    )
    report = env.analytic_envelope_report()
    assert report["build"]["package_version"] == __version__
    assert report["build"]["installed_distribution_version"] == "0.0.0"
    assert not report["build"]["distribution_matches_source_version"]


def test_source_without_distribution_metadata_is_explicit(monkeypatch, measured):
    original = env.version

    def metadata(name):
        if name == "space-env":
            raise env.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(
        env,
        "verify_analytic_fixture",
        lambda *_a, **_kw: copy.deepcopy(measured["analytic"]),
    )
    monkeypatch.setattr(env, "version", metadata)
    report = env.analytic_envelope_report()
    assert report["build"]["installed_distribution_version"] is None
    assert not report["build"]["distribution_matches_source_version"]
