"""First-decision preference payload excludes post-decision/private material."""

import copy
import json
import runpy
from pathlib import Path

import pytest

HELPERS = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples/modeling/export_utility_training.py"
    )
)
BUILD = HELPERS["_preference_payload"]


def records():
    prefix = [
        {"role": "system", "content": "Use public observations."},
        {"role": "user", "content": "public goal"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "0", "function": {"arguments": "wait"}}],
        },
        {"role": "tool", "content": "current public observation"},
    ]

    def record(choice, reward):
        return {
            "split": "train",
            "initial": {"configuration": {"private_config": "DO_NOT_EXPORT"}},
            "metrics": {"terminal": True, "utility": {"violations": []}},
            "call_budget_exhausted": False,
            "events": [{"response": {"reward": reward}}],
            "sha256": choice,
            "messages": prefix
            + [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "1", "function": {"arguments": choice}}],
                },
                {"role": "tool", "content": "SECRET_AFTER_BRANCH"},
            ],
        }

    return record("high", 0.9), record("low", 0.1)


def test_preference_contains_only_shared_prefix_and_first_tool_decision():
    high, low = records()
    row = BUILD(high, low)
    assert len(row["prompt"]) == 4 and row["prompt"][-1]["role"] == "tool"
    assert len(row["chosen"]) == len(row["rejected"]) == 1
    assert row["outcomes"] == {"chosen_return": 0.9, "rejected_return": 0.1}
    assert "SECRET_AFTER_BRANCH" not in json.dumps(
        row
    ) and "DO_NOT_EXPORT" not in json.dumps(row)


@pytest.mark.parametrize(
    "change", ["heldout", "different_configuration", "unsafe", "budget", "nonterminal"]
)
def test_preference_requires_matching_safe_train_episodes(change):
    high, low = records()
    if change == "heldout":
        low["split"] = "test"
    elif change == "different_configuration":
        low["initial"]["configuration"] = {}
    elif change == "unsafe":
        low["metrics"]["utility"]["violations"] = ["fuel"]
    elif change == "budget":
        low["call_budget_exhausted"] = True
    else:
        low["metrics"]["terminal"] = False
    with pytest.raises(ValueError):
        BUILD(high, low)


def test_different_prompt_or_tool_feedback_is_not_a_preference_pair():
    high, low = records()
    low["messages"] = copy.deepcopy(low["messages"])
    low["messages"][1]["content"] = "different problem"
    with pytest.raises(ValueError, match="first divergence"):
        BUILD(high, low)


def test_equal_returns_cannot_create_a_preferred_answer():
    high, low = records()
    low["events"][0]["response"]["reward"] = 0.9
    with pytest.raises(ValueError, match="strict"):
        BUILD(high, low)


def test_export_utility_case_name_cannot_escape_output(tmp_path):
    from space_env.llm.episode import sources
    from space_env.simulation.snapshot import digest

    root, other, output = (
        tmp_path / "corpus",
        tmp_path / "comparisons",
        tmp_path / "output",
    )
    root.mkdir()
    other.mkdir()

    def seal(value):
        return {**value, "sha256": digest(value)}

    manifest = seal({"sources": sources(), "cases": [{"id": "../escape"}]})
    report = seal({"manifest_sha256": manifest["sha256"], "rows": []})
    setup = {"sources": sources(), "competition_manifest_sha256": manifest["sha256"]}
    comparison = seal({"manifest_sha256": digest(setup), "rows": []})
    for path, value in (
        (root / "manifest.json", manifest),
        (root / "report.json", report),
        (other / "manifest.json", setup),
        (other / "report.json", comparison),
    ):
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="case identifier"):
        HELPERS["export_utility_training"](root, other, output)
    assert not output.exists()
