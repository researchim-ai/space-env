"""Comparison example rejects unsafe identifiers before writing output."""

import json
import runpy
from pathlib import Path

import pytest

from space_env.llm.episode import sources
from space_env.simulation.snapshot import digest

EXAMPLE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1] / "examples/modeling/utility_comparisons.py"
    )
)


def manifest(root, ids):
    root.mkdir()
    body = {
        "competition": True,
        "utility": True,
        "sources": sources(),
        "cases": [{"id": i} for i in ids],
    }
    (root / "manifest.json").write_text(json.dumps({**body, "sha256": digest(body)}))


@pytest.mark.parametrize(
    "ids,message",
    [(["train-0000", "train-0000"], "duplicate"), (["../outside"], "identifier")],
)
def test_comparison_cannot_overwrite_or_escape_with_case_names(tmp_path, ids, message):
    root = tmp_path / "corpus"
    output = tmp_path / "output"
    manifest(root, ids)
    with pytest.raises(ValueError, match=message):
        EXAMPLE["compare"](root, output)
    assert not output.exists()


def test_unknown_comparison_policy_cannot_fall_back_to_public():
    with pytest.raises(ValueError, match="unknown comparison policy"):
        EXAMPLE["run"]({}, "invented")
