"""Bounded priorities, declared-safety gating, native equivalence and exports."""

import copy
import json
from itertools import pairwise

import pytest

from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.missions import make_episode
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.industrial.phasing_corpus import run_case
from space_env.industrial.utility import (
    GoalUtilitySpec,
    evaluate_utility,
    resolve_weights,
)
from space_env.simulation.snapshot import canonical_json


def status(a=False, b=False):
    return {
        "conditions": [{"id": "a", "met": a}, {"id": "b", "met": b}],
        "failures": [],
        "resources": [
            {
                "actor": "sat",
                "recipe": {
                    "min_propellant_kg": 1.0,
                    "max_energy_j": 10.0,
                    "require_healthy": True,
                },
                "propellant_kg": 2.0,
                "energy_used_j": 5.0,
                "healthy": True,
            }
        ],
    }


def test_normalization_and_no_undiscounted_cycle_farming():
    weights = {"a": 1, "b": 9}
    scores = [
        evaluate_utility(s, weights)["score"]
        for s in (
            status(),
            status(True),
            status(True, True),
            status(False, True),
            status(),
        )
    ]
    assert scores == [0.0, 0.1, 1.0, 0.9, 0.0]
    assert sum(right - left for left, right in pairwise(scores)) == pytest.approx(0)
    assert evaluate_utility(status(True, True), {"a": 1000, "b": 1000})["score"] == 1.0


@pytest.mark.parametrize("value", [0, -1, 1001, True, 1.0, float("inf")])
def test_priority_bounds_are_strict(value):
    with pytest.raises(ValueError):
        GoalUtilitySpec(priorities={"a": value})


def test_unknown_condition_and_malformed_status_rejected():
    with pytest.raises(ValueError, match="known"):
        resolve_weights(
            GoalUtilitySpec(priorities={"missing": 9}), status()["conditions"]
        )
    malformed = status()
    malformed["conditions"][0]["met"] = "yes"
    with pytest.raises(ValueError):
        evaluate_utility(malformed, {"a": 1, "b": 1})


@pytest.mark.parametrize(
    "change",
    [
        {"propellant_kg": 0.9},
        {"energy_used_j": 11.0},
        {"healthy": False},
        {"propellant_kg": None},
        {"energy_used_j": float("nan")},
    ],
)
def test_safety_failure_dominates_any_completion_and_is_latched(change):
    unsafe = status(True, True)
    unsafe["resources"][0].update(change)
    first = evaluate_utility(unsafe, {"a": 1000, "b": 1})
    assert first["score"] == -2 and first["violations"]
    repaired = evaluate_utility(
        status(True, True), {"a": 1000, "b": 1}, previous_violations=first["violations"]
    )
    assert repaired["score"] == -2 and repaired["violations"] == first["violations"]


def test_soft_goal_failures_do_not_erase_safe_partial_credit_unknown_failures_do():
    partial = status(True)
    partial["failures"] = ["deadline", "orbit_window_missed:b"]
    assert evaluate_utility(partial, {"a": 9, "b": 1})["score"] == 0.9
    partial["failures"].append("new_unrecognised_safety_failure:sat")
    assert evaluate_utility(partial, {"a": 9, "b": 1})["score"] == -2


def config(utility=True):
    value = {
        "actors": 2,
        "generation": {
            "split": "train",
            "seed": 0,
            "windows": True,
            "competition": True,
        },
    }
    if utility:
        value["utility"] = {"priorities": {"slot-sat-00": 9, "slot-sat-01": 1}}
    return value


def lower_priority(observation):
    view = copy.deepcopy(observation)
    weights = view["reward_protocol"]["condition_weights"]
    view["reward_protocol"]["condition_weights"] = {
        key: 1001 - value for key, value in weights.items()
    }
    return orbit_policy(view)


def run_low():
    episode = IndustrialEpisode(
        OrbitSession(config()), split="train", policy_origin="scripted_public"
    )
    index = 0
    while not episode.done:
        obs = episode.observe()
        episode.call(
            str(index),
            canonical_json(
                {
                    "action": lower_priority(obs),
                    "expected_revision": obs["revision"],
                    "idempotency_key": str(index),
                }
            ),
        )
        index += 1
    return episode.record()


def test_competing_goals_choose_priority_and_keep_full_success_honest(tmp_path):
    high = run_case({"configuration": config()}, "public")
    low = run_low()
    for record, expected in ((high, 0.9), (low, 0.1)):
        assert verify_episode(record)["valid"] and not record["metrics"]["success"]
        assert record["metrics"]["goal"]["completed"] == 1
        assert record["metrics"]["utility"]["decision_return"] == pytest.approx(
            expected
        )
        assert sum(e["response"]["reward"] for e in record["events"]) == pytest.approx(
            expected
        )
        assert record["metrics"]["utility"]["raw_native_return"] == 1.0
        prompt = json.loads(record["messages"][1]["content"])["observation"]
        assert prompt["dispatch_bound"]["maximum_completed_conditions"] == 1
        assert prompt["dispatch_bound"]["maximum_weighted_fraction"] == 0.9
    export_training(high, tmp_path / "partial.rl.jsonl", format="rl")
    assert json.loads((tmp_path / "partial.rl.jsonl").read_text())["return"] == 0.9
    with pytest.raises(ValueError, match="SFT"):
        export_training(high, tmp_path / "partial.sft.jsonl", format="sft")


def test_utility_does_not_change_native_physics_and_legacy_reward_is_default():
    base = {"generation": {"split": "train", "seed": 0}}
    legacy = OrbitSession(base)
    weighted = OrbitSession({**base, "utility": {}})
    assert "reward_protocol" not in legacy.observe()
    while not legacy._done:
        action = orbit_policy(legacy.observe())
        for session in (legacy, weighted):
            session.step(
                action,
                expected_revision=session.revision,
                idempotency_key=str(session.revision),
            )
        assert canonical_json(
            legacy._native.checkpoint(compact=True)
        ) == canonical_json(weighted._native.checkpoint(compact=True))
    assert (
        legacy.metrics()["native_return"] == 2.0 and "utility" not in legacy.metrics()
    )
    assert weighted.metrics()["utility"]["decision_return"] == 1.0
    assert legacy.metrics()["goal"] == weighted.metrics()["goal"]


def test_real_native_fuel_violation_has_negative_utility():
    session = OrbitSession(
        {
            "generation": {"split": "train", "seed": 0},
            "utility": {"priorities": {"slot-sat-00": 1000, "slot-sat-01": 1}},
        }
    )
    obs = session.observe()
    first = orbit_policy(obs)["action_ids"][0]
    row = next(r for r in obs["catalogue"] if r["id"] == first)
    actor = row["commands"][0]["actor_id"]
    duration = row["commands"][0]["duration_sec"]
    opposite = next(
        r["id"]
        for r in obs["catalogue"]
        if r["id"] not in ("wait", first) and r["commands"][0]["actor_id"] == actor
    )

    def advance(action):
        return session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )

    advance({"action_ids": [first]})
    while session.observe()["time_sec"] < duration:
        advance({})
    advance({"action_ids": [opposite]})
    while not session._done:
        advance({})
    assert any(
        f.startswith("fuel_reserve:") for f in session.metrics()["goal"]["failures"]
    )
    assert session.metrics()["utility"]["score"] == -2
    assert session.metrics()["utility"]["decision_return"] <= -2


def test_dispatch_rejection_penalty_not_double_counted_and_budget_cannot_pay_credit():
    session = OrbitSession(config())
    names = [
        row["id"] for row in session.observe()["catalogue"] if row["id"] != "wait"
    ][:2]
    result = session.step(
        {"action_ids": names}, expected_revision=0, idempotency_key="reject"
    )
    assert (
        result["reward"] == -0.1
        and result["reward_breakdown"]["rejection_penalty"] == -0.1
    )
    assert result["reward_breakdown"]["utility_delta"] == 0
    assert (
        session.step(
            {"action_ids": names}, expected_revision=0, idempotency_key="reject"
        )
        == result
    )
    assert session.metrics()["utility"]["decision_return"] == -0.1
    episode = make_episode("orbit_and_deployment/priority_test", max_calls=1)
    episode.call("bad", "{}")
    assert (
        episode.done
        and sum(e["response"]["reward"] for e in episode.record()["events"]) == -0.1
    )
    assert episode.record()["metrics"]["utility"]["decision_return"] == 0.0
