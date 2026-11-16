"""Native equivalence, controls, training and replay for the phasing adapter."""

import json

import pytest

from space_env.industrial import OrbitSession
from space_env.industrial.agent import export_training, verify_episode
from space_env.industrial.missions import make_env, run_mission
from space_env.industrial.orbits import orbit_policy
from space_env.industrial.session import restore
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSession
from space_env.tasks.deployment import deployment_recipe


@pytest.mark.parametrize("actors", [True, 1, 9, 2.0])
def test_invalid_fleet_size(actors):
    with pytest.raises(ValueError):
        OrbitSession({"actors": actors})


def test_native_and_adapter_have_identical_physics_goals_and_rewards():
    wrapped = OrbitSession({"actors": 2})
    case = deployment_recipe(2)
    native = TaskSession(case["scenario"], case["task"])
    for index in range(12):
        action = orbit_policy(wrapped.observe())
        result = wrapped.step(
            action, expected_revision=index, idempotency_key=str(index)
        )
        original = native.step(
            {"@central": action.get("action_ids", [])},
            expected_revision=index,
            idempotency_key="orbit-" + str(index),
        )
        for key in ("reward", "elapsed_sec", "terminated", "truncated", "feedback"):
            assert result[key] == original[key]
        assert canonical_json(
            wrapped._native.checkpoint(compact=True)
        ) == canonical_json(native.checkpoint(compact=True))
    assert wrapped.metrics()["success"]
    assert wrapped.metrics()["goal"]["completed"] == 2
    assert all(
        9.98 < r["propellant_kg"] < 10 for r in wrapped.metrics()["goal"]["resources"]
    )


def test_unknown_choice_cannot_mutate_physics_and_duplicates_keep_native_rejection():
    session = OrbitSession()
    initial = session.checkpoint()
    with pytest.raises(ValueError, match="unknown"):
        session.step(
            {"action_ids": ["privileged"]}, expected_revision=0, idempotency_key="bad"
        )
    assert session.checkpoint() == initial
    result = session.step(
        {"action_ids": ["burn-sat-00", "burn-sat-00"]},
        expected_revision=0,
        idempotency_key="duplicate",
    )
    assert not result["feedback"]["accepted"] and result["elapsed_sec"] == 5
    assert session.metrics()["native_rejections"] == 1
    assert not session.metrics()["success"]


def test_json_restore_idempotency_and_forged_state_detection():
    session = OrbitSession()
    action = orbit_policy(session.observe())
    first = session.step(action, expected_revision=0, idempotency_key="burn")
    restored = restore(json.loads(json.dumps(session.checkpoint())))
    assert restored.observe() == session.observe()
    assert restored.step(action, expected_revision=0, idempotency_key="burn") == first
    assert restored.revision == 1
    assert restored.metrics() == session.metrics()
    forged = json.loads(json.dumps(session.checkpoint()))
    forged["state_sha256"] = "0" * 64
    forged["sha256"] = digest({k: v for k, v in forged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="state replay"):
        restore(forged)


def test_wait_fails_and_public_works_with_verified_training_export(tmp_path):
    wait = run_mission("orbit_and_deployment/phasing_2", policy="wait")
    assert not wait["metrics"]["success"] and verify_episode(wait)["valid"]
    public = run_mission(
        "orbit_and_deployment/phasing_2", policy="public", split="train"
    )
    assert public["metrics"]["success"] and verify_episode(public)["valid"]
    assert "reference_steps" not in canonical_json(public["messages"])
    assert "configuration" not in json.loads(public["messages"][1]["content"])
    for format in ("sft", "rl"):
        path = tmp_path / (format + ".jsonl")
        export_training(public, path, format=format)
        row = json.loads(path.read_text())
        assert row["kind"] == "native_orbits" and row["success"]
        assert "initial" not in row and "final" not in row


def test_gym_call_budget_is_truncation_and_goal_is_public():
    env = make_env("orbit_and_deployment/phasing_4", max_calls=1)
    text, _ = env.reset(seed=3)
    observation = json.loads(text)
    assert (
        len(observation["objective"]["conditions"]) == 4 and "task_text" in observation
    )
    action = orbit_policy(observation)
    result = env.step(
        canonical_json(
            {"action": action, "expected_revision": 0, "idempotency_key": "one"}
        )
    )
    assert result[3] and not result[2]
    assert "trainer_only" not in json.loads(result[0])
    env.close()
