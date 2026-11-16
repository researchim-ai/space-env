"""Reject conflicting/reserved/path-like input IDs before writing comparisons."""

import json
import runpy
from pathlib import Path

import pytest

from space_env.llm.episode import sources
from space_env.simulation.snapshot import digest

EXAMPLE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples/modeling/contract_comparisons.py"
    )
)


@pytest.mark.parametrize(
    "ids", [["train-0000", "train-0000"], ["../escape"], ["four-mandatory"]]
)
def test_invalid_comparison_identifiers_cannot_overwrite_or_escape(tmp_path, ids):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "output"
    body = {
        "sources": sources(),
        "acceptance": "mandatory",
        "competition": True,
        "utility": True,
        "cases": [{"id": name} for name in ids],
    }
    (corpus / "manifest.json").write_text(json.dumps({**body, "sha256": digest(body)}))
    with pytest.raises(ValueError, match="identifiers"):
        EXAMPLE["compare"](corpus, output)
    assert not output.exists()


def test_unknown_policy_cannot_be_treated_as_public():
    with pytest.raises(ValueError, match="unknown"):
        EXAMPLE["run"]({}, "invented")
