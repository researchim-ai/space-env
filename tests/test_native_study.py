import json

import pytest

from space_env.industrial import native_study as study
from space_env.industrial.native_proximity import native_proximity_recipe
from space_env.simulation.snapshot import canonical_json, digest


def tiny_suite():
    cfg = native_proximity_recipe()
    cfg["required_observations"] = 1
    return {
        "schema": "test-native-suite",
        "cases": [{"id": "tiny", "kind": "native_proximity", "configuration": cfg}],
        "policies": ["wait", "public"],
        "scope": "unit regression only",
    }


def test_native_matrix_replays_and_missing_cell_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "suite", tiny_suite)
    root = tmp_path / "study"
    report = study.run_study(root, workers=1)
    result = study.audit(root)
    assert (
        result["valid"]
        and result["policy_and_episode_replayed"] == 2
        and result["errors_retained"] == 0
    )
    report["rows"].pop()
    report["sha256"] = digest({k: v for k, v in report.items() if k != "sha256"})
    (root / "report.json").write_text(canonical_json(report))
    with pytest.raises(ValueError, match="matrix"):
        study.audit(root)


def test_unknown_executor_errors_remain_in_matrix(tmp_path, monkeypatch):
    def bad_suite():
        cfg = tiny_suite()
        cfg["cases"][0]["kind"] = "nonexistent"
        return cfg

    monkeypatch.setattr(study, "suite", bad_suite)
    root = tmp_path / "study"
    report = study.run_study(root, workers=1)
    assert len(report["rows"]) == 2 and all(
        r["status"] == "error" for r in report["rows"]
    )
    result = study.audit(root)
    assert (
        result["valid"]
        and result["errors_retained"] == 2
        and result["policy_and_episode_replayed"] == 0
    )
    assert not json.loads((root / "report.json").read_text())["flight_qualified"]
