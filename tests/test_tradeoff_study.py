"""Study accounting rejects scope inflation and retains failed native attempts."""

import json

import pytest

from space_env.llm.episode import seal
from space_env.simulation.snapshot import canonical_json
from space_env.validation import tradeoff_study as study


def test_failed_attempt_keeps_declared_identity(monkeypatch):
    def fail(config):
        raise RuntimeError("deliberate construction failure")

    monkeypatch.setattr(study, "heterogeneous_mission", fail)
    config = {"profile": "balanced", "longitude_shift_deg": -12, "catalogue_seed": 0}
    row = study._execute_native((config, "priority"))
    assert row["status"] == "error"
    assert row["configuration"] == config and row["policy"] == "priority"
    assert row["error_type"] == "RuntimeError"
    assert "metrics" not in row


def test_oracle_only_integrity_never_becomes_full_mission_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_native_jobs", list)
    root = tmp_path / "study"
    report = study.run_study(root)
    result = study.audit(root)
    assert result["valid"] and result["calendar_certificates_recomputed"] == 6
    assert result["native_replayed"] == 0
    with pytest.raises(ValueError, match="new"):
        study.run_study(root)
    report["full_mission_optimality"] = True
    (root / "report.json").write_text(
        canonical_json(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError, match="claim"):
        study.audit(root)


def test_missing_matrix_cell_is_not_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_native_jobs", list)
    root = tmp_path / "study"
    report = study.run_study(root)
    report["native_rows"] = [
        {"configuration": {}, "policy": "priority", "status": "error"}
    ]
    (root / "report.json").write_text(
        json.dumps(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError, match="matrix"):
        study.audit(root)
