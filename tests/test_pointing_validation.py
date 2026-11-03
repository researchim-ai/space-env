"""Off-axis oracle separation, bounded contracts and detectable physical errors."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from space_env.simulation.snapshot import digest
from space_env.validation import pointing as pv

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads(pv.FIXTURE.read_text())


def resign(data):
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    return data


@pytest.fixture(scope="module")
def measured():
    return pv.pointing_validation_report()


def test_all_frozen_cases_measured_without_readiness_inflation(measured):
    from space_env import __version__

    assert measured["passed"]
    assert len(measured["cases"]) == 16
    assert len(measured["checks"]) == 612
    assert not measured["execution_errors"]
    assert all(c["status"] == "passed" for c in measured["cases"])
    assert measured["qualification_status"] == "needs_external_evidence"
    assert not measured["industrial_readiness"]
    assert measured["build"]["package_version"] == __version__
    assert "shared-erfa-inputs" in measured["provenance"]["independence"]
    assert len(measured["unqualified"]) >= 7
    assert (
        digest({k: v for k, v in measured.items() if k != "sha256"})
        == measured["sha256"]
    )


def test_fixed_sites_include_real_passes_and_no_contact(measured):
    windows = [r for r in measured["checks"] if r["metric"] == "window_count_error"]
    assert len(windows) == 8
    assert sum(len(c["expected_windows"]) for c in windows) == 7
    assert sum(not c["actual_windows"] for c in windows) == 4
    for case in windows:
        if case["expected_windows"]:
            np.testing.assert_allclose(
                case["actual_windows"], case["expected_windows"], rtol=0, atol=0.02
            )
        else:
            assert case["actual_windows"] == []
    cases = [c for c in measured["cases"] if c["kind"] == "contact"]
    assert {
        (c["inputs"]["station"]["lat_deg"], c["inputs"]["station"]["lon_deg"])
        for c in cases
    } == {(0.0, 0.0), (40.0, -3.0), (-35.0, 149.0), (78.0, 15.0)}


def test_capacity_timing_is_separate_from_attitude_error(measured):
    caps = [r for r in measured["checks"] if r["budget_group"] == "capacity"]
    assert caps and all(c["passed"] for c in caps)
    assert {
        c["tolerance"] for c in caps if c["metric"] == "saturation_duration_sec"
    } == {0.010001}
    refinement = [r for r in measured["checks"] if r["metric"] == "fine_error_excess"]
    assert len(refinement) == 3
    assert all(r["fine_error"] <= 0.6 * r["coarse_error"] + 1e-10 for r in refinement)


def test_generator_is_separate_and_reproduces_frozen_bytes(tmp_path):
    script = ROOT / "scripts/generate_pointing_vectors.py"
    tree = ast.parse(script.read_text())
    modules = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            modules.update(x.name.split(".")[0] for x in n.names)
        elif isinstance(n, ast.ImportFrom):
            modules.add(n.module.split(".")[0])
    assert modules <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "erfa",
    }
    assert "scipy" not in modules and "space_env" not in modules
    output = tmp_path / "vectors.json"
    subprocess.run(
        [sys.executable, "-I", str(script), str(output)], cwd=tmp_path, check=True
    )
    assert output.read_bytes() == pv.FIXTURE.read_bytes()
    assert (
        fixture()["provenance"]["generator_sha256"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )


def generator():
    spec = importlib.util.spec_from_file_location(
        "independent_pointing_generator", ROOT / "scripts/generate_pointing_vectors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["offaxis-rotating-burn", "offaxis-offset-burn"])
def test_independent_burn_oracle_step_refinement(name):
    case = next(c for c in fixture()["cases"] if c["id"] == name)
    p = copy.deepcopy(case["inputs"])
    p["oracle_step_sec"] /= 2
    fine = generator().burn(p)
    coarse = case["expected"]
    delta = np.asarray(fine["states_si_mass"]) - coarse["states_si_mass"]
    assert max(np.linalg.norm(delta[:, :3], axis=1)) < 5e-6
    assert max(np.linalg.norm(delta[:, 3:6], axis=1)) < 1e-7
    assert np.linalg.norm(np.array(fine["impulse_ns"]) - coarse["impulse_ns"]) < 1e-8
    np.testing.assert_allclose(
        fine["attitude_samples"], coarse["attitude_samples"], rtol=0, atol=1e-9
    )


def mutate(data, name):
    c = data["cases"][0]
    if name == "schema":
        data["schema"] = "unknown"
    elif name == "root_budget":
        data["budgets"] = {"attitude": 1e9}
    elif name == "duplicate":
        data["cases"][1] = copy.deepcopy(c)
    elif name == "missing_case":
        data["cases"].pop()
    elif name == "unknown_id":
        c["id"] = "unknown"
    elif name == "kind":
        c["kind"] = "unknown"
    elif name == "time":
        c["times_sec"][0] = 10.0
    elif name == "inertia":
        c["inputs"]["inertia_kg_m2"][1] = 2.01
    elif name == "torque":
        c["inputs"]["motor_z_nm"] = 100.0
    elif name == "step":
        c["inputs"]["step_sec"] = 0.1
    elif name == "case_budget":
        c["tolerance"] = 1e10
    elif name == "vector_width":
        c["expected"][0].pop()
    elif name == "vector_count":
        c["expected"].pop()
    elif name == "boolean":
        c["expected"][0][0] = True
    elif name == "missing_provenance":
        data["provenance"].pop("scope")
    elif name == "empty_text":
        data["provenance"]["scope"] = " "
    elif name == "numeric_provenance":
        data["provenance"]["erfa_version"] = 1.0
    elif name == "bad_hash":
        data["provenance"]["generator_sha256"] = " " * 64
    elif name == "shared_inputs":
        data["provenance"]["shared_inputs"] = []
    elif name == "independence":
        data["provenance"]["independence"] = "independent-flight-truth"
    elif name == "too_many":
        c["expected"] *= 129
    elif name == "missing_burn_field":
        next(c for c in data["cases"] if c["kind"] == "burn")["expected"].pop(
            "impulse_ns"
        )
    elif name == "burn_width":
        next(c for c in data["cases"] if c["kind"] == "burn")["expected"][
            "attitude_samples"
        ][0].pop()
    elif name == "look_width":
        next(c for c in data["cases"] if c["kind"] == "contact")["expected"]["looks"][
            0
        ].pop()
    elif name == "reverse_window":
        next(c for c in data["cases"] if c["id"] == "plane-a-iberia")["expected"][
            "windows"
        ][0] = [2.0, 1.0]
    elif name == "outside_window":
        next(c for c in data["cases"] if c["id"] == "plane-a-iberia")["expected"][
            "windows"
        ][0] = [0.0, 1e6]
    else:
        raise AssertionError(name)
    return resign(data)


@pytest.mark.parametrize(
    "name",
    [
        "schema",
        "root_budget",
        "duplicate",
        "missing_case",
        "unknown_id",
        "kind",
        "time",
        "inertia",
        "torque",
        "step",
        "case_budget",
        "vector_width",
        "vector_count",
        "boolean",
        "missing_provenance",
        "empty_text",
        "numeric_provenance",
        "bad_hash",
        "shared_inputs",
        "independence",
        "too_many",
        "missing_burn_field",
        "burn_width",
        "look_width",
        "reverse_window",
        "outside_window",
    ],
)
def test_resigned_contract_corruption_rejected_before_any_solver(monkeypatch, name):
    def forbidden(*_a, **_kw):
        pytest.fail("invalid fixture reached numerical solver")

    for attr in ("attitude_checks", "burn_checks", "contact_checks"):
        monkeypatch.setattr(pv, attr, forbidden)
    with pytest.raises(ValueError):
        pv.verify_pointing_fixture(mutate(fixture(), name))


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), None, 10**1000]
)
def test_nonfinite_or_unbounded_expected_values_rejected(value):
    data = fixture()
    data["cases"][0]["expected"][0][0] = value
    with pytest.raises(ValueError):
        pv.validate_pointing_fixture(data)


def test_unsigned_expected_change_rejected():
    data = fixture()
    data["cases"][0]["expected"][0][0] += 0.01
    with pytest.raises(ValueError, match="corrupt"):
        pv.validate_pointing_fixture(data)


@pytest.mark.parametrize(
    "name",
    [
        "offaxis-coast",
        "offaxis-balanced-unloading",
        "offaxis-rotating-burn",
        "offaxis-offset-burn",
        "plane-a-iberia",
        "plane-b-equatorial",
    ],
)
def test_signed_numerical_oracle_errors_still_fail(name):
    data = fixture()
    c = next(c for c in data["cases"] if c["id"] == name)
    if c["kind"] in {"attitude", "forced_attitude"}:
        c["expected"][0][0] += 0.01
    elif c["kind"] == "burn":
        c["expected"]["states_si_mass"][0][0] += 1.0
    elif c["expected"]["windows"]:
        c["expected"]["windows"] = []
    else:
        c["expected"]["windows"] = [[1.0, 2.0]]
    pv.validate_pointing_fixture(resign(data))
    handler = (
        pv.attitude_checks
        if c["kind"] in {"attitude", "forced_attitude"}
        else pv.burn_checks
        if c["kind"] == "burn"
        else pv.contact_checks
    )
    assert any(not row["passed"] for row in handler(c))


def test_frozen_inertial_force_cannot_pass_rotating_burn(monkeypatch):
    from space_env.reference.thrust import RotatingThrust

    case = next(c for c in fixture()["cases"] if c["id"] == "offaxis-offset-burn")
    monkeypatch.setattr(
        RotatingThrust, "direction_at", lambda self, _t: np.asarray(self.direction_gcrs)
    )
    rows = pv.burn_checks(case)
    assert any(not r["passed"] for r in rows if r["metric"] == "direction_chord")
    assert any(not r["passed"] for r in rows if r["metric"] == "impulse_ns")
    assert any(not r["passed"] for r in rows if r["metric"] == "velocity_m_s")


def test_wrong_pointing_goal_fails_independent_line_of_sight(monkeypatch):
    from space_env.reference import spacecraft_bus

    case = next(c for c in fixture()["cases"] if c["id"] == "plane-a-iberia")
    monkeypatch.setattr(
        spacecraft_bus, "pointing_quaternion", lambda *_a: (1.0, 0.0, 0.0, 0.0)
    )
    rows = pv.contact_checks(case)
    assert any(not r["passed"] for r in rows if r["metric"] == "pointing_chord")
    assert all(r["passed"] for r in rows if r["metric"] == "boundary_sec")


def test_solver_failure_retains_all_denominators_and_causes_failure(monkeypatch):
    def broken(*_a, **_kw):
        raise RuntimeError("injected physical solver failure")

    for name in ("attitude_checks", "burn_checks", "contact_checks"):
        monkeypatch.setattr(pv, name, broken)
    result = pv.verify_pointing_fixture(fixture())
    assert not result["passed"] and not result["checks"]
    assert len(result["execution_errors"]) == 16
    assert all(
        c["status"] == "execution_error" and not c["evaluated_times_sec"]
        for c in result["cases"]
    )


@pytest.mark.parametrize(
    "name",
    ["offaxis-positive-cap", "offaxis-negative-cap", "offaxis-balanced-unloading"],
)
def test_restore_continuation_and_unsplit_physics_budget(name):
    from space_env.reference.rigid_attitude import (
        RigidAttitudeState,
        advance_rigid_attitude,
    )

    c = next(c for c in fixture()["cases"] if c["id"] == name)
    p = c["inputs"]
    model, state = pv._rigid_config(p)
    kwargs = {
        "motor_torque_nm": (0.0, 0.0, p["motor_z_nm"]),
        "external_torque_body_nm": tuple(p.get("external_body_nm", [0.0, 0.0, 0.0])),
    }
    total = c["times_sec"][-1]
    half, _ = advance_rigid_attitude(model, state, 3.507, **kwargs)
    restored = RigidAttitudeState.model_validate_json(half.model_dump_json())
    warm, _ = advance_rigid_attitude(model, half, total - 3.507, **kwargs)
    cold, _ = advance_rigid_attitude(model, restored, total - 3.507, **kwargs)
    assert warm.model_dump() == cold.model_dump()
    whole, _ = advance_rigid_attitude(model, state, total, **kwargs)
    np.testing.assert_allclose(
        pv._rotation(whole.quaternion_wxyz),
        pv._rotation(cold.quaternion_wxyz),
        rtol=0,
        atol=1e-6,
    )


def test_report_budgets_do_not_alias_global_state(monkeypatch, measured):
    monkeypatch.setattr(
        pv, "verify_pointing_fixture", lambda _data: copy.deepcopy(measured)
    )
    first = pv.pointing_validation_report()
    first["budgets"]["burn"]["position_m"] = 1e9
    assert pv.pointing_validation_report()["budgets"]["burn"]["position_m"] == 0.002


def test_cli_persists_failed_evidence_with_nonzero_exit(monkeypatch, tmp_path, capsys):
    from space_env.cli import main

    path = tmp_path / "failed.json"
    failed = {
        "schema": "space-env.pointing-validation/1",
        "passed": False,
        "qualification_status": "needs_external_evidence",
        "cases": [],
        "checks": [],
        "sha256": "test",
    }
    monkeypatch.setattr(pv, "pointing_validation_report", lambda: failed)
    monkeypatch.setattr(
        sys, "argv", ["space-env", "reference-qualify-pointing", str(path)]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert json.loads(path.read_text()) == failed
    assert not json.loads(capsys.readouterr().out)["passed"]
