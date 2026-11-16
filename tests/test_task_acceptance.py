"""Declared acceptance, dual labels, mandatory incentives and budget finality."""

import copy
import json

import pytest

from space_env.industrial.acceptance import (
    TaskAcceptanceSpec,
    acceptance_bound,
    evaluate_acceptance,
)
from space_env.industrial.agent import (
    export_training,
    verify_episode,
)
from space_env.industrial.missions import make_env, make_episode, run_mission
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.simulation.snapshot import canonical_json, digest


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, float("nan"), float("inf"), True])
def test_acceptance_threshold_is_positive_finite_and_bounded(value):
    with pytest.raises(ValueError):
        TaskAcceptanceSpec(minimum_fraction=value)


def test_mandatory_ids_and_utility_requirement_validated():
    for ids in (["a", "a"], [" "]):
        with pytest.raises(ValueError):
            TaskAcceptanceSpec(minimum_fraction=0.9, mandatory=ids)
    with pytest.raises(ValueError, match="utility"):
        OrbitSession({"acceptance": {"minimum_fraction": 0.9}})
    with pytest.raises(ValueError, match="declared goal"):
        OrbitSession(
            {
                "utility": {},
                "acceptance": {"minimum_fraction": 0.9, "mandatory": ["missing"]},
            }
        )


def test_acceptance_needs_natural_safe_terminal_and_mandatory_conditions():
    spec = TaskAcceptanceSpec(minimum_fraction=0.9, mandatory=["high"])
    goal = {"conditions": [{"id": "high", "met": True}, {"id": "low", "met": False}]}
    utility = {"score": 0.9, "completion_fraction": 0.9, "violations": []}
    assert not evaluate_acceptance(spec, goal, utility, finished=False)["accepted"]
    assert not evaluate_acceptance(spec, goal, utility, finished=True, truncated=True)[
        "accepted"
    ]
    assert evaluate_acceptance(spec, goal, utility, finished=True)["accepted"]
    unsafe = {**utility, "score": -2, "violations": ["fuel"]}
    assert not evaluate_acceptance(spec, goal, unsafe, finished=True)["accepted"]


def test_bound_is_sufficient_and_conditional_on_current_dispatch_rules():
    weights = {"high": 9, "low": 1}

    def bound(required, threshold, limit):
        return acceptance_bound(
            TaskAcceptanceSpec(minimum_fraction=threshold, mandatory=required),
            weights,
            {"maximum_completed_conditions": limit},
        )

    assert (
        bound(["high", "low"], 0.9, 1)["reason"]
        == "mandatory_count_exceeds_dispatch_bound"
    )
    assert bound(["low"], 0.9, 1)["reason"] == "threshold_exceeds_dispatch_bound"
    assert bound(["low"], 0.1, 1)["status"] == "not_ruled_out"
    assert bound(["high", "low"], 1.0, 2)["status"] == "not_ruled_out"
    assert bound([], 0.1, 0)["status"] == "infeasible_under_current_dispatch_rules"


def run_priority_only():
    episode = make_episode(
        "orbit_and_deployment/contract_mandatory_train", policy_origin="scripted_public"
    )
    index = 0
    while not episode.done:
        obs = episode.observe()
        view = copy.deepcopy(obs)
        view.pop("task_objective")
        episode.call(
            str(index),
            canonical_json(
                {
                    "action": orbit_policy(view),
                    "expected_revision": obs["revision"],
                    "idempotency_key": str(index),
                }
            ),
        )
        index += 1
    return episode.record()


def test_mandatory_low_value_target_beats_optional_high_reward_and_sft_is_explicit(
    tmp_path,
):
    accepted = run_mission(
        "orbit_and_deployment/contract_mandatory_train", policy="public"
    )
    wrong = run_priority_only()
    assert (
        verify_episode(accepted)["task_accepted"] and not accepted["metrics"]["success"]
    )
    assert accepted["metrics"]["utility"]["decision_return"] == pytest.approx(0.1)
    assert not verify_episode(wrong)["task_accepted"]
    assert wrong["metrics"]["utility"]["score"] == 0.9
    assert wrong["metrics"]["utility"]["decision_return"] == pytest.approx(-0.1)
    for record in (accepted, wrong):
        for event in record["events"]:
            r = event["response"]
            parts = r["reward_breakdown"]
            assert r["reward"] == pytest.approx(
                parts["utility_delta"]
                + parts["mandatory_penalty"]
                + parts["rejection_penalty"]
            )
    with pytest.raises(ValueError, match="SFT"):
        export_training(accepted, tmp_path / "native.sft.jsonl", format="sft")
    export_training(
        accepted, tmp_path / "declared.sft.jsonl", format="sft", objective="declared"
    )
    row = json.loads((tmp_path / "declared.sft.jsonl").read_text())
    assert row["schema"] == "space-env.industrial-objective-training/1"
    assert (
        row["success"]
        and not row["native_success"]
        and row["success_basis"] == "declared_task_acceptance"
    )
    prompt = json.loads(row["messages"][1]["content"])
    assert prompt["observation"]["task_objective"]["mandatory"] == ["slot-sat-01"]
    assert "task_objective" in prompt["task_text"]
    assert prompt["episode_budget"]["incomplete_penalty"] == 1.0
    with pytest.raises(ValueError, match="SFT"):
        export_training(
            wrong, tmp_path / "wrong.sft.jsonl", format="sft", objective="declared"
        )


def test_no_declared_contract_cannot_be_inferred_after_execution(tmp_path):
    legacy = run_mission("orbit_and_deployment/priority_train", policy="public")
    with pytest.raises(ValueError, match="configured"):
        export_training(
            legacy, tmp_path / "retroactive.jsonl", format="rl", objective="declared"
        )


def test_budget_exhaustion_does_not_avoid_task_failure_cost_or_advance_physics():
    episode = make_episode(
        "orbit_and_deployment/contract_priority_train",
        max_calls=1,
        policy_origin="scripted_public",
    )
    result = episode.call("bad", "{}")
    assert (
        result["reward"] == pytest.approx(-1.1)
        and result["episode_penalty"]["amount"] == -1
    )
    assert episode.observe()["time_sec"] == 0.0
    record = episode.record()
    assert verify_episode(record)["valid"]
    assert (
        record["call_budget_exhausted"]
        and not record["metrics"]["task_acceptance"]["accepted"]
    )
    assert record["metrics"]["utility"]["decision_return"] == 0.0
    with pytest.raises(ValueError, match="finished"):
        episode.call("again", "{}")
    env = make_env("orbit_and_deployment/contract_priority_test", max_calls=1)
    observation, _ = env.reset()
    assert json.loads(observation)["episode_budget"]["remaining_calls"] == 1
    value, reward, terminated, truncated, _ = env.step("{}")
    assert reward == pytest.approx(-1.1) and truncated and not terminated
    assert json.loads(value)["episode_budget"]["remaining_calls"] == 0
    env.close()


def test_infeasible_request_is_retained_and_forged_acceptance_rejected():
    record = run_mission(
        "orbit_and_deployment/contract_infeasible_train", policy="public"
    )
    assert (
        verify_episode(record)["valid"]
        and not record["metrics"]["task_acceptance"]["accepted"]
    )
    initial = json.loads(record["messages"][1]["content"])["observation"]
    assert (
        initial["acceptance_bound"]["status"]
        == "infeasible_under_current_dispatch_rules"
    )
    forged = copy.deepcopy(record)
    forged["metrics"]["task_acceptance"]["accepted"] = True
    forged["sha256"] = digest({k: v for k, v in forged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay"):
        verify_episode(forged)


def test_hidden_capacity_increase_changes_only_published_bound_and_can_enable_task():
    base = {
        "actors": 2,
        "generation": {
            "split": "train",
            "seed": 0,
            "windows": True,
            "competition": True,
        },
        "utility": {"priorities": {"slot-sat-00": 9, "slot-sat-01": 1}},
        "acceptance": {
            "minimum_fraction": 1.0,
            "mandatory": ["slot-sat-00", "slot-sat-01"],
        },
    }
    plain = OrbitSession(base)
    future = OrbitSession(
        {
            **base,
            "dispatch_events": [{"at_sec": 5.0, "kind": "capacity", "capacity": 2}],
        }
    )
    assert plain.observe() == future.observe()
    assert (
        future.observe()["acceptance_bound"]["status"]
        == "infeasible_under_current_dispatch_rules"
    )
    future.step({}, expected_revision=0, idempotency_key="wait")
    assert future.observe()["dispatch_bound"]["maximum_completed_conditions"] == 2
    assert future.observe()["acceptance_bound"]["status"] == "not_ruled_out"
    while not future._done:
        future.step(
            orbit_policy(future.observe()),
            expected_revision=future.revision,
            idempotency_key=str(future.revision),
        )
    assert (
        future.metrics()["success"] and future.metrics()["task_acceptance"]["accepted"]
    )


def test_acceptance_config_is_owned_and_public_from_start():
    configuration = {
        "utility": {},
        "acceptance": {"minimum_fraction": 1.0, "mandatory": ["slot-sat-00"]},
    }
    session = OrbitSession(configuration)
    configuration["acceptance"]["mandatory"].clear()
    assert session.observe()["task_objective"]["mandatory"] == ["slot-sat-00"]
