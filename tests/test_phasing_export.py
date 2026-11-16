"""Reject partial matrices and record path traversal before emitting datasets."""

import json
import runpy
from pathlib import Path

import pytest

from space_env.llm.episode import sources
from space_env.simulation.snapshot import digest

EXPORT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/modeling/export_phasing.py")
)["export_corpus"]


def sealed(value):
    return {**value, "sha256": digest(value)}


def fixture_corpus(root, rows):
    root.mkdir()
    manifest = sealed(
        {
            "sources": sources(),
            "cases": [
                {
                    "id": "train-0",
                    "configuration": {
                        "actors": 2,
                        "generation": {"split": "train", "seed": 0},
                    },
                }
            ],
        }
    )
    report = sealed({"manifest_sha256": manifest["sha256"], "rows": rows})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "report.json").write_text(json.dumps(report))


def test_partial_matrix_cannot_be_exported(tmp_path):
    root = tmp_path / "corpus"
    out = tmp_path / "output"
    fixture_corpus(root, [{"case": "train-0", "policy": "wait"}])
    with pytest.raises(ValueError, match="incomplete"):
        EXPORT(root, out)
    assert not out.exists()


def test_complete_but_path_traversing_matrix_cannot_read_external_records(tmp_path):
    root = tmp_path / "corpus"
    out = tmp_path / "output"
    rows = [
        {
            "case": "train-0",
            "policy": p,
            "status": "completed",
            "record": "../outside.json",
        }
        for p in ("wait", "public")
    ]
    fixture_corpus(root, rows)
    with pytest.raises(ValueError, match="outside corpus"):
        EXPORT(root, out)
    assert not out.exists()


def test_record_split_cannot_be_relabelled_as_train(tmp_path):
    root = tmp_path / "corpus"
    out = tmp_path / "output"
    rows = [
        {
            "case": "train-0",
            "policy": p,
            "status": "completed",
            "record": "record.json",
            "sha256": "same",
            "metrics": {},
        }
        for p in ("wait", "public")
    ]
    fixture_corpus(root, rows)
    record = {
        "sha256": "same",
        "initial": {
            "configuration": {"actors": 2, "generation": {"split": "train", "seed": 0}}
        },
        "split": "test",
        "metrics": {},
    }
    (root / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity mismatch"):
        EXPORT(root, out)
    assert not out.exists()


def test_case_identifiers_cannot_escape_output_directory(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    body = {"sources": sources(), "cases": [{"id": "../escape", "configuration": {}}]}
    manifest = sealed(body)
    report = sealed({"manifest_sha256": manifest["sha256"], "rows": []})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="case identifier"):
        EXPORT(root, tmp_path / "output")
    assert not (tmp_path / "output").exists()
