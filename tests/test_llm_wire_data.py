"""Synthetic provider unit inputs backed by real native replay, not LLM scores."""

import copy
import io
import json

import pytest

from examples.llm.generate_local import wire_sft_rows
from space_env.llm.benchmark import public_policy
from space_env.llm.episode import STEP, LLMEpisode, seal
from space_env.llm.ollama import OllamaPolicy
from space_env.workspace import MissionWorkspace


@pytest.fixture(scope="module")
def successful_wire():
    class FakeProvider:
        value = None

        def open(self, request, timeout):
            return io.BytesIO(
                json.dumps(
                    {
                        "model": "qwen3.5:4b",
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 1,
                        "eval_count": 1,
                        "message": {"content": json.dumps(self.value)},
                    }
                ).encode()
            )

    mission = MissionWorkspace.network()
    try:
        agent = LLMEpisode(
            mission.operator.session,
            task_text="Deliver.",
            group="wire-unit",
            policy_origin="scripted_public",
        )
        oracle = public_policy("public_memory", 5.0)
        policy = OllamaPolicy(protocol_profile="json_fixed_v1")
        provider = FakeProvider()
        policy._opener = provider
        while not agent.done:
            scripted = oracle(agent.messages, agent.tool_schemas())
            choices = json.loads(scripted["arguments"])["choices"]
            provider.value = {
                "tool": STEP,
                **{
                    f"action_{i}": choices[r][0]
                    for i, r in enumerate(sorted(agent.roles))
                },
            }
            call = policy(agent.messages, agent.tool_schemas())
            assert agent.invoke(
                call["name"], call["arguments"], call_id=str(len(agent._events))
            )["ok"]
        record = agent.record()
        assert record["metrics"]["sft_eligible"]
        return record, policy.trace
    finally:
        mission.close()


def test_wire_targets_are_actual_semantic_json_not_sdk_metadata(successful_wire):
    record, trace = successful_wire
    rows = wire_sft_rows(record, trace, model="qwen3.5:4b")
    assert len(rows) == len(record["events"])
    for row in rows:
        target = json.loads(row["messages"][-1]["content"])
        assert set(target) == {"tool", "action_0", "action_1", "action_2"}
        assert row["message_loss_mask"] == [False, False, True]
        assert row["native_replay_verified"] and row["split"] == "train"


@pytest.mark.parametrize("fault", ["target", "prompt", "split", "delivery"])
def test_wire_export_rejects_unbound_target_prompt_and_evaluation_split(
    successful_wire, fault
):
    record, trace = copy.deepcopy(successful_wire)
    if fault == "target":
        target = json.loads(trace[0]["response"]["message"]["content"])
        target["action_0"] = "wait"
        trace[0]["response"]["message"]["content"] = json.dumps(target)
    elif fault == "prompt":
        trace[0]["request"]["messages"][0]["content"] = "invented context"
    elif fault == "delivery":
        trace[0]["delivered_calls"] = []
    else:
        record["configuration"]["split"] = "test"
        record = seal({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        wire_sft_rows(record, trace, model="qwen3.5:4b")
