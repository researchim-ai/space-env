"""Native tool dialogue, information boundaries and cold-replayed training rows."""

import json

import pytest

from space_env.llm.episode import (
    OBSERVE,
    STEP,
    LLMEpisode,
    export_training,
    seal,
    verify_record,
)
from space_env.runtime import reference_scenario
from space_env.simulation.snapshot import canonical_json
from space_env.tasks import TaskSession
from tests.test_custody_network import spec as custody_fixture
from tests.test_runtime_tasks import task


def session():
    return TaskSession(
        reference_scenario(
            custody_fixture.__wrapped__(), id="llm-test", lineage="llm-family"
        ),
        task(),
    )


def episode(**kwargs):
    return LLMEpisode(
        session(),
        task_text="Deliver the products to Ground.",
        group="test-geometry",
        **kwargs,
    )


def test_actor_only_messages_and_schema_no_control_plane_or_operator_actions():
    agent = episode()
    text = canonical_json(agent.messages)
    for forbidden in (
        "recover-ground",
        "initial_checkpoint",
        "model_sources",
        "station_recovered",
        "operator_event",
        "transitions",
    ):
        assert forbidden not in text
    assert set(agent.observe()["roles"]) == {"a", "b"}
    properties = agent.tool_schemas()[1]["function"]["parameters"]["properties"][
        "choices"
    ]["properties"]
    assert set(properties) == {"a", "b"}
    assert all("recover-ground" not in r["items"]["enum"] for r in properties.values())
    messages = agent.messages
    messages[0]["content"] = "mutated"
    assert agent.messages[0]["content"] != "mutated"


@pytest.mark.parametrize(
    "arguments", ["{", '{"x":1,"x":2}', '{"x":NaN}', "[]", '{"secret":1}']
)
def test_invalid_json_is_a_recorded_failed_call_not_a_native_step(arguments):
    agent = episode(max_calls=1)
    response = agent.invoke(OBSERVE, arguments, call_id="bad")
    assert not response["ok"]
    assert agent.observe()["revision"] == 0
    record = agent.record()
    assert record["metrics"]["tool_calls"] == record["metrics"]["failed_calls"] == 1
    assert record["metrics"]["stop_reason"] == "tool_budget"
    assert verify_record(record)["verified"]
    with pytest.raises(ValueError):
        agent.invoke(OBSERVE, "{}", call_id="after-budget")


def test_operator_authority_and_wrong_actor_actions_cannot_be_requested():
    agent = episode()
    for i, choices in enumerate(
        (
            {"operator": ["recover-ground"]},
            {"a": ["ground-b"]},
            {"@central": ["ground-a"]},
        )
    ):
        result = agent.invoke(
            STEP,
            canonical_json(
                {"choices": choices, "expected_revision": 0, "idempotency_key": str(i)}
            ),
            call_id=str(i),
        )
        assert not result["ok"] and agent.observe()["revision"] == 0


def test_idempotent_retries_do_not_duplicate_rl_reward_or_physical_time():
    original = session()
    agent = LLMEpisode(original, task_text="Deliver.", group="g")
    args = canonical_json(
        {
            "choices": {"a": ["wait"]},
            "expected_revision": 0,
            "idempotency_key": "one",
            "advance_sec": 0.5,
        }
    )
    assert agent.invoke(STEP, args, call_id="first")["ok"]
    assert agent.invoke(STEP, args, call_id="retry")["ok"]
    assert original.revision == 0
    record = agent.record()
    assert record["metrics"]["native_decisions"] == 1
    assert [e["transition_applied"] for e in record["events"]] == [True, False]
    assert record["events"][1]["training_reward"] == 0
    assert verify_record(record)["verified"]


def test_resealed_semantic_tamper_fails_native_dialogue_replay():
    agent = episode()
    agent.invoke(OBSERVE, "{}", call_id="o")
    record = agent.record()
    record["events"][0]["response"]["observation"]["revision"] = 999
    record = seal({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_record(record)


def test_rl_retains_failures_sft_rejects_unfinished_goal_and_holdout(tmp_path):
    agent = episode()
    agent.invoke(OBSERVE, "{", call_id="invalid")
    record = agent.record()
    target = tmp_path / "rl.jsonl"
    export_training(record, target)
    row = json.loads(target.read_text())
    assert row["quality"] == "behavior_rollout_including_failures"
    assert len(row["rewards"]) == 1 and row["transition_applied"] == [False]
    assert "initial_checkpoint" not in canonical_json(row)
    with pytest.raises(ValueError):
        export_training(record, tmp_path / "sft.jsonl", format="sft")
    holdout = episode(split="test").record()
    with pytest.raises(ValueError):
        export_training(holdout, tmp_path / "test.jsonl")
    assert not (tmp_path / "test.jsonl").exists()


def test_native_reference_dialogue_can_produce_causal_sft(tmp_path):
    from space_env.llm.benchmark import reference_policy, run_agent
    from space_env.workspace import MissionWorkspace

    mission = MissionWorkspace.network()
    try:
        native = mission.operator.session
        metadata = native._scenario["metadata"]["network_workspace"]
        agent = LLMEpisode(
            native,
            task_text="Deliver survey product with acknowledgement.",
            group="demo",
        )
        record = run_agent(
            agent,
            reference_policy(
                metadata["steps"], {a.id: a.role for a in native._task.actions}
            ),
        )
        assert record["metrics"]["success"] and record["metrics"]["sft_eligible"]
        export_training(record, tmp_path / "sft.jsonl", format="sft")
        row = json.loads((tmp_path / "sft.jsonl").read_text())
        assert all(
            mask == (message["role"] == "assistant")
            for mask, message in zip(row["message_loss_mask"], row["messages"])
        )
        assert row["quality"] == "successful_execution_not_expert"
        assert "reference/" not in canonical_json(
            [m for m in row["messages"] if m["role"] == "user"]
        )
    finally:
        mission.close()


def test_dialogue_budget_stops_before_another_provider_call_and_retains_last_reply():
    initial = session()
    probe = LLMEpisode(initial, task_text="Deliver.", group="g")
    budget = len(canonical_json(probe.messages).encode()) + 100
    agent = LLMEpisode(
        initial, task_text="Deliver.", group="g", max_dialogue_bytes=budget
    )
    assert agent.invoke(OBSERVE, "{}", call_id="observe")["ok"]
    assert (
        agent.done
        and agent.record()["metrics"]["stop_reason"] == "dialogue_byte_budget"
    )
    assert verify_record(agent.record())["verified"]


def test_same_initial_task_has_same_rl_group_across_different_rollouts(tmp_path):
    a, b = episode(), episode()
    b.invoke(OBSERVE, "{}", call_id="o")
    export_training(a.record(), tmp_path / "a.jsonl")
    export_training(b.record(), tmp_path / "b.jsonl")
    x, y = [
        json.loads((tmp_path / name).read_text()) for name in ("a.jsonl", "b.jsonl")
    ]
    assert x["rl_group_id"] == y["rl_group_id"]
    assert x["episode_sha256"] != y["episode_sha256"]


def test_public_runner_callback_only_receives_owned_messages_and_tools():
    from space_env.llm.benchmark import run_agent

    agent = episode()

    def stop(messages, tools):
        messages.clear()
        tools.clear()

    record = run_agent(agent, stop)
    assert len(record["messages"]) == 2
    assert record["metrics"]["stop_reason"] == "caller_stopped"


def test_finish_rejects_incomplete_goal_without_time_or_private_state():
    from space_env.llm.episode import FINISH

    agent = episode()
    before = agent.observe()
    response = agent.invoke(FINISH, "{}", call_id="claim")
    assert not response["ok"] and response["error"]["code"] == "mission_incomplete"
    assert agent.observe() == before and not agent.done
    assert set(response) == {"ok", "error", "observation"}
    assert verify_record(agent.record())["verified"]


def test_verified_finish_can_close_delivered_goal_before_horizon_and_rl_discount(
    tmp_path,
):
    from space_env.llm.benchmark import public_policy
    from space_env.llm.episode import FINISH
    from space_env.workspace import MissionWorkspace

    mission = MissionWorkspace.network()
    try:
        agent = LLMEpisode(mission.operator.session, task_text="Deliver.", group="g")
        policy = public_policy("public_memory", 5.0)
        for i in range(30):
            call = policy(agent.messages, agent.tool_schemas())
            agent.invoke(call["name"], call["arguments"], call_id=str(i))
            requests = agent._session._runtime._backend.report()["lifecycle"][
                "requests"
            ]
            if all(r["status"] == "completed" for r in requests.values()):
                break
        assert not agent.done
        before = agent.observe()
        assert agent.invoke(FINISH, "{}", call_id="finish")["mission_complete"]
        assert agent.done and agent.observe() == before
        record = agent.record()
        assert record["metrics"]["success"] and record["metrics"]["sft_eligible"]
        assert record["metrics"]["stop_reason"] == "verified_goal"
        assert record["events"][-1]["transition_applied"] is False
        assert record["events"][-1]["training_reward"] == 0
        export_training(record, tmp_path / "rl-finish.jsonl")
        row = json.loads((tmp_path / "rl-finish.jsonl").read_text())
        assert row["discounts"][-1] == 0 and row["rewards"][-1] == 0
        export_training(record, tmp_path / "sft-finish.jsonl", format="sft")
    finally:
        mission.close()
