"""Corpus bridge contracts. Study-audit dependency is stubbed, episodes are native.

These unit fixtures are not qualification scores or a replacement for the full
72-cell native acceptance run.
"""

import json

import pytest

from examples.llm import qualified_corpus as bridge
from space_env.llm.corpus import build_corpus
from space_env.llm.episode import LLMEpisode
from space_env.simulation.snapshot import canonical_json, digest
from tests.test_llm_episode import session


def make_fixture(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    record = LLMEpisode(
        session(), task_text="Unit fixture only.", group="unit"
    ).record()
    attempt = root / "cases" / "unit" / "attempt-01"
    attempt.mkdir(parents=True)
    (attempt / "episode.json").write_text(canonical_json(record))
    (attempt / "report.json").write_text(
        canonical_json(
            {
                "rows": [
                    {
                        "method": "sequence_lanes",
                        "status": "completed",
                        "artifact": "episode.json",
                        "episode_sha256": record["sha256"],
                    }
                ]
            }
        )
    )
    (root / "study.json").write_text(
        canonical_json({"suite": {"cases": [{"id": "unit"}]}})
    )
    (root / "journal.json").write_text(
        canonical_json(
            {"attempts": [{"case": "unit", "attempt": 1, "status": "completed"}]}
        )
    )
    monkeypatch.setattr(
        bridge, "audit_study", lambda _: {"report_sha256": digest("unit audit stub")}
    )
    return root, attempt, record


def test_bridge_build_and_resume_are_native_verified_and_idempotent(
    tmp_path, monkeypatch
):
    root, _, record = make_fixture(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    registry = tmp_path / "registry.sqlite"
    first = bridge.build(root, output, registry)
    second = bridge.build(root, output, registry, resume=True)
    assert first == second and first["registration"]["registered"]
    assert first["summary"]["rows"] == {"rl": 1, "sft": 0, "wire-sft": 0}
    assert (
        json.loads((output / "corpus.json").read_text())["entries"][0]["episode_sha256"]
        == record["sha256"]
    )


def test_resume_cannot_attach_an_unrelated_valid_corpus(tmp_path, monkeypatch):
    root, _, _ = make_fixture(tmp_path, monkeypatch)
    foreign = LLMEpisode(
        session(), task_text="Different native dialogue.", group="other"
    ).record()
    path = tmp_path / "foreign.json"
    path.write_text(canonical_json(foreign))
    output = tmp_path / "corpus"
    build_corpus([path], output)
    registry = tmp_path / "registry.sqlite"
    with pytest.raises(ValueError, match="not bound"):
        bridge.build(root, output, registry, resume=True)
    assert not registry.exists()


def test_incomplete_study_is_rejected_before_any_output(tmp_path, monkeypatch):
    def reject(_):
        raise ValueError("Qualification matrix is incomplete")

    monkeypatch.setattr(bridge, "audit_study", reject)
    out = tmp_path / "output"
    registry = tmp_path / "registry.sqlite"
    with pytest.raises(ValueError, match="incomplete"):
        bridge.build(tmp_path / "study", out, registry)
    assert not out.exists() and not registry.exists()


def test_corpus_cannot_modify_the_study_tree(tmp_path, monkeypatch):
    root, _, _ = make_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside qualification"):
        bridge.build(root, root / "new-corpus", tmp_path / "registry.sqlite")
    assert not (root / "new-corpus").exists()
