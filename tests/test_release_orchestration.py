"""Aggregation cannot substitute different cases or falsify artifact accounting."""

import copy
import importlib.util
from pathlib import Path

import pytest

from space_env.planning.release_benchmark import Suite
from space_env.planning.release_cases import release_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

spec = importlib.util.spec_from_file_location(
    "release_orchestrator",
    Path(__file__).parents[1] / "scripts/run_release_qualification.py",
)
orchestrator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orchestrator)


def prepare(tmp_path):
    value = release_suite(smoke=True)
    value["cases"] = value["cases"][:1]
    frozen = Suite.model_validate(value).model_dump(mode="json", by_alias=True)
    atomic_write(tmp_path / "spec.json", canonical_json(frozen))
    directory = tmp_path / "parts/000/results"
    directory.mkdir(parents=True)
    expected = copy.deepcopy(frozen)
    expected["id"] += "-part-0"
    atomic_write(directory / "spec.json", canonical_json(expected))
    atomic_write(directory.parent / "input.json", canonical_json(expected))
    part = {
        "index": 0,
        "group": value["cases"][0]["group"],
        "exit_code": 0,
        "audit": None,
    }
    atomic_write(directory.parent / "status.json", canonical_json(part))
    body = {
        "schema": "space-env.release-qualification/1",
        "parts": [part],
        "spec_sha256": digest(frozen),
        "rows": [],
        "teacher": {},
        "all_native_rows_verified": True,
        "global_artifact_gate": True,
    }
    return body, directory


def save(root, body):
    body["files"] = orchestrator.inventory(root)
    body["artifact_bytes"] = sum(m["bytes"] for m in body["files"].values())
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    atomic_write(root / "report.json", canonical_json(body))


def test_resealed_wrong_shard_refused_before_native_audit(tmp_path, monkeypatch):
    body, directory = prepare(tmp_path)
    other = Suite.model_validate(release_suite(smoke=True)).model_dump(
        mode="json", by_alias=True
    )
    atomic_write(directory / "spec.json", canonical_json(other))
    save(tmp_path, body)
    monkeypatch.setattr(
        orchestrator,
        "verify_suite",
        lambda *a, **k: pytest.fail("should reject before audit"),
    )
    with pytest.raises(ValueError, match="frozen parent"):
        orchestrator.verify(tmp_path)


def test_resealed_artifact_count_refused(tmp_path):
    body, _ = prepare(tmp_path)
    save(tmp_path, body)
    body["artifact_bytes"] = 0
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    atomic_write(tmp_path / "report.json", canonical_json(body))
    with pytest.raises(ValueError, match="accounting"):
        orchestrator.verify(tmp_path)


def test_resealed_group_identity_refused(tmp_path):
    body, _ = prepare(tmp_path)
    body["parts"][0]["group"] = "foreign"
    save(tmp_path, body)
    with pytest.raises(ValueError, match="identity"):
        orchestrator.verify(tmp_path)


def test_freeze_capacity_profile_preserves_physical_cases(tmp_path):
    import json
    import subprocess
    import sys

    output = tmp_path / "profile.json"
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).parents[1] / "scripts/run_release_qualification.py"),
        "freeze",
        str(output),
        "--artifact-gib",
        "4",
        "--result-mb",
        "128",
    ]
    subprocess.run(command, check=True, capture_output=True)
    value = json.loads(output.read_text())
    original = Suite.model_validate(release_suite()).model_dump(
        mode="json", by_alias=True
    )
    assert value["cases"] == original["cases"] and value["gates"] == original["gates"]
    assert value["budget"]["max_result_bytes"] == 128000000
    assert value["max_total_artifact_bytes"] == 4294967296
    before = output.read_bytes()
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
    assert output.read_bytes() == before
