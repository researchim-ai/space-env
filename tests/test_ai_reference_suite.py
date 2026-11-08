"""Reference agreement, failure sensitivity, and the local report contract."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from space_env.validation import ai_reference_suite as suite


@pytest.fixture(scope="module")
def report():
    return suite.run_suite()


def test_fixed_reference_cases_pass_and_are_explicit(report):
    assert report["passed"] is True
    assert len(report["checks"]) == 32
    assert len({row["name"] for row in report["checks"]}) == 32
    for row in report["checks"]:
        assert row["passed"] is True, row
        assert 0 <= row["max_absolute_error"] <= row["tolerance"]
        assert row["units"] and row["applicability"] and row["reference"]
    rows = {row["name"]: row for row in report["checks"]}
    assert rows["kepler_e0.2_position"]["tolerance"] == 1.0
    assert rows["kepler_e0.2_velocity"]["tolerance"] == 0.001
    assert rows["kepler_e0.2_position"]["sample_count"] > 1000
    assert rows["friis_loss"]["tolerance"] == 0.01
    assert rows["rocket_mass_10000.0"]["expected"] == 100.0
    assert report["external_engine_comparison"]["performed"] is False
    assert "external-engine agreement" in report["not_qualified"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), -float("inf"), 1.01])
def test_nonfinite_and_out_of_tolerance_fail(actual):
    row = suite._check("bad", actual, 0.0, 1.0, "m", "test", "test")
    assert row["passed"] is False
    json.dumps(row, allow_nan=False)


def test_inclusive_boundary_and_shape_validation():
    assert suite._check("edge", 1.0, 0.0, 1.0, "m", "test", "test")["passed"]
    with pytest.raises(ValueError, match="shapes"):
        suite._check("bad", [1.0, 2.0], 0.0, 1.0, "m", "test", "test")
    with pytest.raises(ValueError, match="tolerance"):
        suite._check("bad", 0.0, 0.0, float("nan"), "m", "test", "test")


@pytest.mark.parametrize(
    "model",
    [
        "orbit",
        "gravity",
        "radio",
        "rocket",
        "thermal",
        "rigid_attitude",
        "rigid_energy",
    ],
)
def test_corrupted_production_model_fails_report(monkeypatch, model):
    if model == "orbit":
        real = suite.propagate_state

        def corrupt(*args, **kwargs):
            r, v, samples = real(*args, **kwargs)
            return (
                r,
                v,
                [
                    (t, pos + np.array([100.0, 0.0, 0.0]), vel)
                    for t, pos, vel in samples
                ],
            )

        monkeypatch.setattr(suite, "propagate_state", corrupt)
    elif model == "gravity":
        real = suite.gravity.acceleration_m_s2
        monkeypatch.setattr(
            suite.gravity, "acceleration_m_s2", lambda *a, **kw: real(*a, **kw) * 1.001
        )
    elif model == "radio":
        monkeypatch.setattr(suite.link, "free_space_path_loss_db", lambda *a: 0.0)
    elif model == "rocket":
        monkeypatch.setattr(
            suite.propulsion, "mass_after_thrust", lambda *a: (120.0, 0.0)
        )
    elif model.startswith("rigid"):
        real = suite.rigid_attitude.advance_rigid_attitude

        def corrupt(*args, **kwargs):
            result, detail = real(*args, **kwargs)
            key, value = (
                ("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))
                if model == "rigid_attitude"
                else ("dc_energy_j", result.dc_energy_j + 1.0)
            )
            return result.model_copy(update={key: value}), detail

        monkeypatch.setattr(suite.rigid_attitude, "advance_rigid_attitude", corrupt)
    else:
        monkeypatch.setattr(suite.thermal, "step_temperature_c", lambda *a, **kw: 0.0)
    report = suite.run_suite()
    assert report["passed"] is False
    assert any(not row["passed"] for row in report["checks"])


def test_local_runner_writes_report_and_returns_failure(monkeypatch, tmp_path, capsys):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/validation/ai_reference_suite.py"
    )
    spec = importlib.util.spec_from_file_location("ai_reference_example", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--output", str(output)])

    def fake_report(passed):
        sources = suite.source_manifest()
        return suite.seal_report(
            {
                "schema": "space-env.ai-reference-suite/2",
                "passed": passed,
                "checks": [],
                "sources": sources,
                "sources_sha256": suite.digest(sources),
            }
        )

    monkeypatch.setattr(runner, "run_suite", lambda: fake_report(False))
    assert runner.main() == 1
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)
    monkeypatch.setattr(runner, "run_suite", lambda: fake_report(True))
    assert runner.main() == 0


def test_source_manifest_and_native_seal(report, monkeypatch):
    assert suite.verify_report(report)
    assert report["sha256"] == suite.digest(
        {k: v for k, v in report.items() if k != "sha256"}
    )
    assert report["sources_sha256"] == suite.digest(report["sources"])
    files = report["sources"]["files"]
    root = Path(__file__).resolve().parents[1]
    for name in (
        "src/space_env/validation/ai_reference_suite.py",
        "src/space_env/reference/rigid_attitude.py",
        "src/space_env/reference/rigid_bus.py",
        "src/space_env/simulation/snapshot.py",
        "tests/test_ai_reference_suite.py",
        "examples/validation/ai_reference_suite.py",
        "docs/AI_PHYSICS_REFERENCE.md",
    ):
        assert files[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert not report["sources"]["unavailable_repository_artifacts"]
    tampered = copy.deepcopy(report)
    tampered["checks"][0]["tolerance"] *= 10
    assert not suite.verify_report(tampered, check_current_sources=False)
    changed = copy.deepcopy(report["sources"])
    changed["files"]["src/space_env/reference/rigid_attitude.py"] = "0" * 64
    monkeypatch.setattr(suite, "source_manifest", lambda: changed)
    assert not suite.verify_report(report)
    assert suite.verify_report(report, check_current_sources=False)
    assert not suite.verify_report({})


def test_source_change_during_run_is_rejected(monkeypatch):
    source = suite.source_manifest()
    changed = copy.deepcopy(source)
    changed["files"]["new.py"] = "0" * 64
    sequence = iter([source, changed])
    monkeypatch.setattr(suite, "source_manifest", lambda: next(sequence))
    with pytest.raises(RuntimeError, match="source files changed"):
        suite.run_suite()


def test_observed_runtime_and_reference_kernel_scope(report):
    runtime = report["runtime"]
    assert runtime["python"] and runtime["implementation"] and runtime["platform"]
    assert runtime["loaded_distributions"]["numpy"] == np.__version__
    assert "pydantic" in runtime["loaded_distributions"]
    rows = [row for row in report["checks"] if row["name"].startswith("reference_")]
    assert len(rows) == 16
    assert all("kernel" in row["applicability"] for row in rows)
    assert all(row["passed"] for row in rows)
    assert (
        "mission rigid controller, battery coupling and wheel saturation/regeneration"
        in report["not_qualified"]
    )
