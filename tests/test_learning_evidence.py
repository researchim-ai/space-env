"""Pairing and statistical gates, not synthetic claims of LLM training benefit."""

import copy

import pytest

from space_env.llm.benchmark import public_policy, run_agent
from space_env.llm.episode import LLMEpisode
from space_env.navigation.task_suite import compile_suite
from space_env.tasks import TaskSession
from space_env.validation.learning_evidence import (
    learning_comparison,
    paired_statistics,
)


def disclosure():
    return {
        "training_groups": ["train-only"],
        "training_dataset_sha256": "0" * 64,
        "before_weights_sha256": "1" * 64,
        "after_weights_sha256": "2" * 64,
        "before_inference_trace_sha256": "3" * 64,
        "after_inference_trace_sha256": "4" * 64,
    }


def test_exact_paired_statistics_and_low_sample_gate():
    before = {str(i): False for i in range(20)}
    after = dict.fromkeys(before, True)
    result = paired_statistics(before, after)
    assert result["paired_improvement_detected"]
    assert result["exact_two_sided_mcnemar_p"] == 2 / 2**20
    assert result["success_rate_difference"] == 1.0
    assert not paired_statistics({"one": False}, {"one": True})[
        "paired_improvement_detected"
    ]
    assert paired_statistics(after, after)["exact_two_sided_mcnemar_p"] == 1.0
    assert not paired_statistics(after, before)["paired_improvement_detected"]


@pytest.mark.parametrize(
    "before,after", [({}, {}), ({"a": True}, {"b": True}), ({"a": 1}, {"a": True})]
)
def test_invalid_pairing(before, after):
    with pytest.raises(ValueError):
        paired_statistics(before, after)


@pytest.fixture(scope="module")
def record():
    case = compile_suite()[0]
    ep = LLMEpisode(
        TaskSession(case["scenario"], case["task"]),
        task_text="Synthetic replay fixture",
        group="held-out-one",
        split="test",
        max_calls=8,
        policy_origin="external_llm",
    )
    # Synthetic test transport only. Not actual inference and not reported as data.
    return run_agent(ep, public_policy("wait", 1.0))


def test_native_outcomes_do_not_attest_training(record):
    result = learning_comparison([record], [record], disclosure())
    assert result["native_replay_verified"]
    assert not result["training_proven"]
    assert not result["statistics"]["paired_improvement_detected"]


def test_training_leak_duplicate_and_weights_guard(record):
    declared = disclosure()
    declared["training_groups"] = ["held-out-one"]
    with pytest.raises(ValueError, match="leaking"):
        learning_comparison([record], [record], declared)
    with pytest.raises(ValueError, match="duplicate"):
        learning_comparison([record, record], [record, record], disclosure())
    declared = disclosure()
    declared["after_weights_sha256"] = declared["before_weights_sha256"]
    with pytest.raises(ValueError, match="identities"):
        learning_comparison([record], [record], declared)


def test_scripted_records_not_ranked_as_llm(record):
    from space_env.llm.episode import seal

    scripted = copy.deepcopy(record)
    scripted["configuration"]["policy_origin"] = "scripted_public"
    scripted = seal({k: v for k, v in scripted.items() if k != "sha256"})
    with pytest.raises(ValueError, match="external LLM"):
        learning_comparison([scripted], [scripted], disclosure())
