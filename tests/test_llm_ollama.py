"""Local inference boundary and evidence accounting without calling a model."""

import copy
import io
import json

import pytest

from space_env.llm.ollama import InferenceStopped, OllamaPolicy, window_messages


def messages():
    return [
        {"role": "system", "content": "operate"},
        {"role": "user", "content": "task"},
    ]


class Opener:
    def __init__(self, result):
        self.result, self.requests = result, []

    def open(self, request, timeout):
        self.requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(self.result).encode())


def response(calls=None):
    return {
        "model": "qwen3.5:4b",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 20,
        "eval_count": 10,
        "message": {"role": "assistant", "content": "", "tool_calls": calls or []},
    }


def test_no_remote_endpoint_or_credentials_or_implicit_model_download():
    for endpoint in (
        "https://example.com",
        "http://10.0.0.1:11434",
        "http://user:secret@localhost",
        "http://localhost/a",
    ):
        with pytest.raises(ValueError):
            OllamaPolicy(endpoint=endpoint)


def test_tool_calls_are_preserved_in_order_without_inventing_actions():
    p = OllamaPolicy()
    p._opener = Opener(
        response(
            [
                {"function": {"name": "first", "arguments": {"x": 1}}},
                {"function": {"name": "second", "arguments": {"x": 2}}},
            ]
        )
    )
    assert p(messages(), [])["name"] == "first"
    assert p(messages(), [])["name"] == "second"
    assert len(p._opener.requests) == 1
    assert p.metrics()["input_tokens"] == 20 and p.metrics()["output_tokens"] == 10
    assert p._opener.requests[0]["think"] is False
    assert p._opener.requests[0]["stream"] is False


def test_context_window_keeps_complete_tool_pairs_and_converts_arguments():
    ms = messages()
    for i in range(4):
        ms.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": str(i),
                            "function": {"name": "step", "arguments": '{"x":1}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "observation"},
            ]
        )
    before = copy.deepcopy(ms)
    selected = window_messages(ms, 2)
    assert len(selected) == 6 and ms == before
    assert selected[2]["tool_calls"][0]["id"] == "2"
    assert selected[3]["tool_name"] == "step"
    assert selected[2]["tool_calls"][0]["function"]["arguments"] == {"x": 1}


@pytest.mark.parametrize("fault", ["truncated", "wrong_model", "usage", "arguments"])
def test_bad_generation_is_retained_and_not_repaired(fault):
    r = response()
    if fault == "truncated":
        r["done_reason"] = "length"
    elif fault == "wrong_model":
        r["model"] = "another-model"
    elif fault == "usage":
        r.pop("eval_count")
    else:
        r["message"]["tool_calls"] = [
            {"function": {"name": "step", "arguments": "broken"}}
        ]
    p = OllamaPolicy()
    p._opener = Opener(r)
    with pytest.raises(InferenceStopped):
        p(messages(), [])
    assert p.trace[0]["response"] == r


def test_no_tools_stops_instead_of_manufacturing_wait_and_budget_blocks_request():
    p = OllamaPolicy(max_requests=1)
    p._opener = Opener(response())
    assert p(messages(), []) is None
    with pytest.raises(InferenceStopped):
        p(messages(), [])
    assert len(p._opener.requests) == 1
    p = OllamaPolicy(context_tokens=8192)
    p._opener = Opener(response())
    with pytest.raises(InferenceStopped):
        p(messages(), [])
    assert not p._opener.requests


@pytest.mark.parametrize("profile", ["baseline_v1", "persistent_v1", "semantic_v1"])
def test_protocol_profiles_change_only_public_system_contract(profile):
    from space_env.llm.ollama import inference_messages

    original = messages()
    selected = inference_messages(original, 2, profile)
    assert original == messages()
    assert selected[1:] == original[1:]
    if profile == "baseline_v1":
        assert selected == original
    else:
        assert "terminated=true" in selected[0]["content"]
        assert "observe alone never advances time" in selected[0]["content"]
    assert ("status_code legend" in selected[0]["content"]) == (
        profile == "semantic_v1"
    )
    p = OllamaPolicy(protocol_profile=profile)
    p._opener = Opener(response())
    assert p(original, []) is None
    assert p._opener.requests[0]["messages"] == selected
    assert p.metrics()["protocol_profile"] == profile


def test_profile_does_not_manufacture_actions_after_prose_stop():
    p = OllamaPolicy(protocol_profile="persistent_v1")
    p._opener = Opener(response())
    assert p(messages(), []) is None
    assert not p._pending
    with pytest.raises(ValueError):
        OllamaPolicy(protocol_profile="invented")


def test_brief_context_uses_latest_public_state_and_only_enabled_ids():
    from space_env.llm.episode import STEP
    from space_env.llm.ollama import inference_context

    view = {
        "time_sec": 20.0,
        "entities": [
            {
                "kind": "command",
                "id": "sent",
                "index": 0,
                "key": "command:sent",
                "values": {"status_code": 7.0},
            }
        ],
        "relations": [],
        "catalogue": [
            {"id": "wait", "enabled": True, "reason": "wait"},
            {"id": "send", "enabled": False, "reason": "product_not_local"},
        ],
    }
    team = {"revision": 4, "roles": {"a": view}}
    ms = [
        {"role": "system", "content": "Operate"},
        {
            "role": "user",
            "content": json.dumps({"task": "Deliver", "observation": {"revision": 0}}),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "observe", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": json.dumps({"observation": team}),
        },
    ]
    tools = [
        {
            "function": {
                "name": STEP,
                "parameters": {
                    "properties": {
                        "choices": {
                            "properties": {"a": {"items": {"enum": ["wait", "send"]}}}
                        }
                    }
                },
            }
        }
    ]
    before = copy.deepcopy((ms, tools))
    selected, narrowed = inference_context(ms, tools, 2, "brief_v1")
    assert (ms, tools) == before
    current = json.loads(selected[1]["content"])
    assert current["current_revision"] == 4
    assert current["roles"]["a"]["entities"][0]["values"]["status"] == "completed"
    assert current["roles"]["a"]["entities"][0]["index"] == 0
    assert narrowed[0]["function"]["parameters"]["properties"]["choices"]["properties"][
        "a"
    ]["items"]["enum"] == ["wait"]
    assert "space_env_actor_finish" in selected[0]["content"]


def test_structured_wire_decodes_only_the_models_explicit_role_choices():
    from space_env.llm.episode import STEP
    from tests.test_llm_episode import episode

    agent = episode()
    value = {
        "tool": STEP,
        "action_0": "wait",
        "action_1": "wait",
        "expected_revision": 0,
        "idempotency_key": "unique",
        "advance_sec": 0.5,
    }
    raw = response()
    raw["message"]["content"] = json.dumps(value)
    policy = OllamaPolicy(protocol_profile="json_action_v1")
    policy._opener = Opener(raw)
    call = policy(agent.messages, agent.tool_schemas())
    assert json.loads(call["arguments"]) == {
        "choices": {"a": ["wait"], "b": ["wait"]},
        "expected_revision": 0,
        "idempotency_key": "unique",
        "advance_sec": 0.5,
    }
    assert agent.invoke(call["name"], call["arguments"], call_id="x")["ok"]
    sent = policy._opener.requests[0]
    assert sent["tools"] == []
    assert set(sent["format"]["required"]) == set(value)
    assert sent["format"]["properties"]["action_0"]["type"] == "string"
    assert policy.trace[0]["response"] == raw


@pytest.mark.parametrize(
    "fault",
    ["missing", "nonfinite", "boolean_revision", "broken_json"],
)
def test_invalid_structured_decision_is_never_repaired(fault):
    from space_env.llm.episode import STEP
    from tests.test_llm_episode import episode

    agent = episode()
    value = {
        "tool": STEP,
        "action_0": "wait",
        "action_1": "wait",
        "expected_revision": 0,
        "idempotency_key": "x",
        "advance_sec": 0.5,
    }
    if fault == "unknown_action":
        value["action_0"] = "recover-ground"
    elif fault == "missing":
        value.pop("action_1")
    elif fault == "nonfinite":
        value["advance_sec"] = float("inf")
    elif fault == "boolean_revision":
        value["expected_revision"] = True
    raw = response()
    raw["message"]["content"] = "{" if fault == "broken_json" else json.dumps(value)
    policy = OllamaPolicy(protocol_profile="json_action_v1")
    policy._opener = Opener(raw)
    with pytest.raises(InferenceStopped):
        policy(agent.messages, agent.tool_schemas())
    assert agent.observe()["revision"] == 0
    assert policy.trace[0]["response"] == raw


def test_http_failure_retains_status_body_and_marks_usage_unknown():
    import urllib.error

    class Fail:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "parser failure",
                {},
                io.BytesIO(b'{"error":"invalid tool XML"}'),
            )

    p = OllamaPolicy()
    p._opener = Fail()
    with pytest.raises(InferenceStopped):
        p(messages(), [])
    assert p.trace[0]["http_status"] == 500
    assert "invalid tool XML" in p.trace[0]["provider_error_body"]
    assert p.metrics()["usage_incomplete"] is True


def test_zero_duration_is_valid_for_observe_but_never_a_physical_step():
    from space_env.llm.episode import OBSERVE, STEP
    from tests.test_llm_episode import episode

    agent = episode()
    policy = OllamaPolicy(protocol_profile="json_action_v1")
    value = {
        "tool": OBSERVE,
        "expected_revision": 0,
        "idempotency_key": "o",
        "advance_sec": 0,
        "action_0": "wait",
        "action_1": "wait",
    }
    raw = response()
    raw["message"]["content"] = json.dumps(value)
    policy._opener = Opener(raw)
    call = policy(agent.messages, agent.tool_schemas())
    assert call == {"name": OBSERVE, "arguments": "{}"}
    assert policy.trace[0]["delivered_calls"] == [call]
    assert policy.trace[0]["decoded_calls"] == [call]
    value["tool"] = STEP
    raw["message"]["content"] = json.dumps(value)
    policy._opener = Opener(raw)
    call = policy(agent.messages, agent.tool_schemas())
    result = agent.invoke(call["name"], call["arguments"], call_id="bad-zero")
    assert not result["ok"]
    assert agent.observe()["revision"] == 0


def test_well_formed_bad_duration_is_native_feedback_not_clipped_or_abort():
    from space_env.llm.episode import STEP
    from tests.test_llm_episode import episode

    agent = episode()
    p = OllamaPolicy(protocol_profile="json_action_v1")
    value = {
        "tool": STEP,
        "action_0": "wait",
        "action_1": "wait",
        "expected_revision": 0,
        "idempotency_key": "invalid",
        "advance_sec": 95.0,
    }
    raw = response()
    raw["message"]["content"] = json.dumps(value)
    p._opener = Opener(raw)
    call = p(agent.messages, agent.tool_schemas())
    assert json.loads(call["arguments"])["advance_sec"] == 95.0
    result = agent.invoke(call["name"], call["arguments"], call_id="bad-duration")
    assert result["error"]["code"] == "advance_duration_out_of_range"
    assert result["error"]["details"]["maximum_sec"] == 5.0
    assert agent.observe()["revision"] == 0
    value.update(advance_sec=0.5, idempotency_key="fixed")
    raw["message"]["content"] = json.dumps(value)
    p._opener = Opener(raw)
    call = p(agent.messages, agent.tool_schemas())
    assert agent.invoke(call["name"], call["arguments"], call_id="recovered")["ok"]
    assert agent.observe()["revision"] == 1


def test_fixed_cadence_model_chooses_actions_sdk_only_supplies_transaction_metadata():
    from space_env.llm.episode import STEP
    from tests.test_llm_episode import episode

    agent = episode()
    policy = OllamaPolicy(protocol_profile="json_fixed_v1", fixed_decision_sec=0.5)
    raw = response()
    raw["message"]["content"] = json.dumps(
        {"tool": STEP, "action_0": "wait", "action_1": "wait"}
    )
    policy._opener = Opener(raw)
    call = policy(agent.messages, agent.tool_schemas())
    args = json.loads(call["arguments"])
    assert args == {
        "choices": {"a": ["wait"], "b": ["wait"]},
        "expected_revision": 0,
        "idempotency_key": "llm-fixed/0/0",
        "advance_sec": 0.5,
    }
    assert agent.invoke(call["name"], call["arguments"], call_id="x")["ok"]
    sent = policy._opener.requests[0]
    assert set(sent["format"]["required"]) == {"tool", "action_0", "action_1"}
    assert "Fixed-cadence transport" in sent["messages"][0]["content"]
    second = policy(agent.messages, agent.tool_schemas())
    assert json.loads(second["arguments"])["expected_revision"] == 1
    assert json.loads(second["arguments"])["idempotency_key"] == "llm-fixed/1/1"
    raw["message"]["content"] = json.dumps(
        {"tool": STEP, "action_0": "wait", "action_1": "wait", "expected_revision": 0}
    )
    policy._opener = Opener(raw)
    with pytest.raises(InferenceStopped):
        policy(agent.messages, agent.tool_schemas())
