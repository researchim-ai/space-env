"""Explicit benchmark split authority and atomic/fail-closed SFT selection."""

import importlib.util
import json
from pathlib import Path

import pytest

from space_env.planning.release_benchmark import POLICIES, Suite
from space_env.planning.release_cases import release_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest

spec = importlib.util.spec_from_file_location(
    "release_data", Path(__file__).parents[1] / "scripts/export_release_data.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def context(tmp_path, promote=False):
    value = release_suite(smoke=True)
    value["cases"] = [value["cases"][0], value["cases"][-1]]
    spec = Suite.model_validate(value)
    root = tmp_path / "native"
    root.mkdir()
    report = {
        "sha256": "a" * 64,
        "teacher": {"passed_declared_gates": promote},
        "files": {},
    }
    paths = {}
    for i, c in enumerate(spec.cases):
        for policy in POLICIES:
            path = root / str(i) / policy
            path.mkdir(parents=True)
            paths[c.id, policy] = path
            for fmt in ("rl", "sft"):
                row = {
                    "schema": "space-env.task-dataset/1",
                    "split": "train",
                    "lineage": c.scenario["lineage"],
                    "index": 0,
                    "observation": {"public": True},
                    "format": fmt,
                }
                source = path / (fmt + ".jsonl")
                atomic_write(source, canonical_json(row) + "\n")
                report["files"][source.relative_to(root).as_posix()] = {
                    "sha256": file_digest(source)
                }
    return root, report, spec, paths


def test_explicit_test_group_never_becomes_train(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(exporter, "source", lambda *a, **k: ctx)
    output = tmp_path / "export"
    manifest = exporter.publish("unused", output)
    assert not manifest["sft_exported"]
    assert set(manifest["files"]) == {
        "train/rl.jsonl",
        "development/rl.jsonl",
        "test/rl.jsonl",
    }
    test = [json.loads(x) for x in (output / "test/rl.jsonl").read_text().splitlines()]
    assert len(test) == 3 and all(
        x["split"] == "test" and x["legacy_lineage_split"] == "train" for x in test
    )
    train = [
        json.loads(x) for x in (output / "train/rl.jsonl").read_text().splitlines()
    ]
    assert all(x["benchmark_group"] == "train-orbits" for x in train)
    assert exporter.verify("unused", output)["integrity_passed"]


def test_sft_only_from_promoted_training_group(tmp_path, monkeypatch):
    ctx = context(tmp_path, True)
    monkeypatch.setattr(exporter, "source", lambda *a, **k: ctx)
    output = tmp_path / "export"
    manifest = exporter.publish("unused", output)
    assert (
        manifest["sft_exported"] and manifest["files"]["train/sft.jsonl"]["rows"] == 3
    )
    assert not (output / "test/sft.jsonl").exists()
    assert exporter.verify("unused", output)["sft_exported"]


def test_resealed_split_tamper_refused(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(exporter, "source", lambda *a, **k: ctx)
    output = tmp_path / "export"
    manifest = exporter.publish("unused", output)
    path = output / "train/rl.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0] = json.loads((output / "test/rl.jsonl").read_text().splitlines()[0])
    rows[0]["split"] = "train"
    atomic_write(path, "".join(canonical_json(r) + "\n" for r in rows))
    m = manifest["files"]["train/rl.jsonl"]
    m.update(
        file_sha256=file_digest(path),
        bytes=path.stat().st_size,
        rows_sha256=digest(rows),
    )
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    atomic_write(output / "manifest.json", canonical_json(manifest))
    with pytest.raises(ValueError, match="source row"):
        exporter.verify("unused", output)


def test_changed_source_never_publishes_partial_dataset(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(exporter, "source", lambda *a, **k: ctx)
    path = next(iter(ctx[3].values())) / "rl.jsonl"
    path.write_bytes(path.read_bytes() + path.read_bytes())
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="changed during export"):
        exporter.publish("unused", output)
    assert not output.exists() and not list(tmp_path.glob(".export.*"))
