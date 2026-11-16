import json

import pytest

from space_env.industrial import study
from space_env.simulation.snapshot import canonical_json, digest


def small_suite():
    return {
        "schema": "space-env.industrial-suite/1",
        "cases": [
            {
                "id": "proximity-one",
                "kind": "proximity",
                "configuration": {"horizon_steps": 1},
            }
        ],
        "policies": ["wait", "public"],
        "scope": "test only",
    }


def test_study_replays_matrix_and_rejects_missing_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "suite", small_suite)
    root = tmp_path / "study"
    report = study.run_study(root)
    assert study.audit(root)["replayed"] == 2
    report["rows"].pop()
    report["sha256"] = digest({k: v for k, v in report.items() if k != "sha256"})
    (root / "report.json").write_text(canonical_json(report))
    with pytest.raises(ValueError, match="matrix"):
        study.audit(root)


def test_study_keeps_infrastructure_errors_without_success_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "suite", small_suite)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(study, "run_case", fail)
    root = tmp_path / "study"
    report = study.run_study(root)
    assert len(report["rows"]) == 2 and all(
        r["status"] == "error" for r in report["rows"]
    )
    result = study.audit(root)
    assert (
        result["valid"] and result["replayed"] == 0 and result["errors_retained"] == 2
    )
    assert not json.loads((root / "report.json").read_text())["flight_qualification"]
