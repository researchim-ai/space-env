"""Local-only Ollama tool adapter with explicit context window and raw evidence.

No model download, cloud provider, command repair or invented fallback action.
"""

from __future__ import annotations

import copy
import json
import math
import time
import urllib.error
import urllib.request
from collections import deque
from urllib.parse import urlsplit

from space_env.simulation.snapshot import canonical_json


class InferenceStopped(RuntimeError):
    """A budget/provider/protocol failure; the partial native episode is retained."""


def window_messages(messages, pairs):
    if type(pairs) is not int or not 1 <= pairs <= 16:
        raise ValueError("Invalid context window")
    if len(messages) < 2 or (len(messages) - 2) % 2:
        raise ValueError("Expected two-message context and complete tool pairs")
    result = copy.deepcopy(messages[:2] + messages[max(2, len(messages) - 2 * pairs) :])
    calls = {}
    for message in result:
        if message["role"] == "assistant":
            message.setdefault("content", "")
            for call in message.get("tool_calls", []):
                calls[call["id"]] = call["function"]["name"]
                arguments = call["function"]["arguments"]
                call["function"]["arguments"] = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
        elif message["role"] == "tool":
            message["tool_name"] = calls[message.pop("tool_call_id")]
    return result


PROFILES = (
    "baseline_v1",
    "persistent_v1",
    "semantic_v1",
    "brief_v1",
    "json_action_v1",
    "json_fixed_v1",
)
CONTINUATION_CONTRACT = (
    "\nInteractive execution contract: continue calling tools until a tool response explicitly says terminated=true or truncated=true. "
    "A prose promise to monitor or continue is not an action and ends the rollout. "
    "An accepted command is not a completed operation or delivered product. "
    "When operations are running, advance physical time with a joint step containing declared wait actions; "
    "observe alone never advances time. Choose advance_sec within the tool schema limit. "
    "Use current command/product/request states to avoid resubmitting already completed work. "
    "Every further command and every wait must be chosen by you through a tool call."
)


def inference_messages(messages, pairs, profile="baseline_v1"):
    if profile not in PROFILES:
        raise ValueError("Unknown inference protocol profile")
    selected = window_messages(messages, pairs)
    if profile != "baseline_v1":
        selected[0]["content"] += CONTINUATION_CONTRACT
    if profile == "semantic_v1":
        from space_env.tasks.observation import STATES

        selected[0]["content"] += (
            "\nPublic observation status_code legend: " + canonical_json(STATES)
        )
    return selected


def inference_context(messages, tools, pairs, profile="baseline_v1"):
    if profile not in ("brief_v1", "json_action_v1", "json_fixed_v1"):
        return inference_messages(messages, pairs, profile), copy.deepcopy(tools)
    from space_env.llm.episode import STEP
    from space_env.tasks.observation import STATES

    window_messages(messages, pairs)  # validate the owned causal conversation
    task = json.loads(messages[1]["content"])["task"]
    latest = json.loads(messages[-1]["content"])
    team = latest["observation"]
    states = {v: k for k, v in STATES.items()}
    roles, enabled = {}, {}
    for role, view in team["roles"].items():
        entities = []
        for entity in view["entities"]:
            values = copy.deepcopy(entity["values"])
            if "status_code" in values:
                values["status"] = states.get(values["status_code"], "unknown")
            entities.append(
                {
                    "kind": entity["kind"],
                    "id": entity["id"],
                    "index": entity["index"],
                    "key": entity["key"],
                    "values": values,
                }
            )
        available = [copy.deepcopy(a) for a in view["catalogue"] if a["enabled"]]
        enabled[role] = {a["id"] for a in available}
        roles[role] = {
            "time_sec": view["time_sec"],
            "entities": entities,
            "relations": copy.deepcopy(view["relations"]),
            "available_actions": available,
            "blocked_actions": [
                {"id": a["id"], "reason": a["reason"]}
                for a in view["catalogue"]
                if not a["enabled"]
            ],
        }
        if "goal" in view:
            roles[role]["goal"] = copy.deepcopy(view["goal"])
    narrowed = copy.deepcopy(tools)
    for tool in narrowed:
        if tool["function"]["name"] == STEP:
            for role, schema in tool["function"]["parameters"]["properties"]["choices"][
                "properties"
            ].items():
                schema["items"]["enum"] = [
                    a for a in schema["items"]["enum"] if a in enabled[role]
                ]
    recent = [
        copy.deepcopy(call["function"])
        for m in messages[max(2, len(messages) - 2 * pairs) :]
        if m["role"] == "assistant"
        for call in m.get("tool_calls", [])
    ]
    context = {
        "task": task,
        "current_revision": team["revision"],
        "roles": roles,
        "recent_tool_calls": recent,
        "last_feedback": latest.get("feedback"),
        "last_error": latest.get("error"),
        "terminated": latest.get("terminated", False),
    }
    system = (
        messages[0]["content"]
        + CONTINUATION_CONTRACT
        + (
            "\nUse only current_revision as expected_revision, and use a unique idempotency key per new decision. "
            "Only currently enabled action IDs are offered in tools. Single-action masks do not guarantee joint feasibility: "
            "any rejected member rejects the entire joint command batch. Do not repeat a relay after its product moved to the destination. "
            "Once every request is completed by native Ground ACK, call space_env_actor_finish to verify completion, even before the horizon. "
            "A rejected completion claim means continue operating; never merely assert success in prose. "
            "This is a deterministic view of current public observations, not a plan or future-state forecast."
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": canonical_json(context)},
    ], narrowed


def action_schema(tools, *, fixed=False):
    from space_env.llm.episode import FINISH, OBSERVE, STEP

    native = next(
        t["function"]["parameters"] for t in tools if t["function"]["name"] == STEP
    )
    roles = sorted(native["properties"]["choices"]["properties"])
    properties = {
        "tool": {
            "type": "string",
            "enum": [STEP, FINISH] if fixed else [OBSERVE, STEP, FINISH],
        }
    }
    for key in ("expected_revision", "idempotency_key", "advance_sec"):
        properties[key] = copy.deepcopy(native["properties"][key])
    properties["advance_sec"].pop("exclusiveMinimum", None)
    properties["advance_sec"]["minimum"] = 0.0
    properties["advance_sec"]["description"] = (
        "Zero for observe/finish; strictly positive duration for joint_step."
    )
    if fixed:
        for key in ("expected_revision", "idempotency_key", "advance_sec"):
            properties.pop(key)
    for i, role in enumerate(roles):
        properties[f"action_{i}"] = copy.deepcopy(
            native["properties"]["choices"]["properties"][role]["items"]
        )
        properties[f"action_{i}"]["description"] = (
            f"One action for role {role}. All role actions are ignored for observe/finish."
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def decode_calls(
    response,
    tools,
    profile,
    *,
    current_revision=None,
    request_index=None,
    fixed_decision_sec=5.0,
):
    if profile not in ("json_action_v1", "json_fixed_v1"):
        decoded = []
        for call in response.get("message", {}).get("tool_calls", []):
            function = call.get("function", {})
            if not isinstance(function.get("name"), str) or not isinstance(
                function.get("arguments"), dict
            ):
                raise InferenceStopped("provider_tool_call_format")
            decoded.append(
                {
                    "name": function["name"],
                    "arguments": canonical_json(function["arguments"]),
                }
            )
        return decoded
    from space_env.llm.episode import STEP, _json

    try:
        value = _json(response["message"]["content"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InferenceStopped("invalid_structured_action_response") from exc
    schema = action_schema(tools, fixed=profile == "json_fixed_v1")
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise InferenceStopped("invalid_structured_action_fields")
    for key, spec in schema["properties"].items():
        v = value[key]
        if spec["type"] == "string" and (
            not isinstance(v, str)
            or (key == "tool" and v not in spec["enum"])
            or len(v) < spec.get("minLength", 0)
            or len(v) > spec.get("maxLength", 65536)
        ):
            raise InferenceStopped("invalid_structured_action_string")
        if spec["type"] == "integer" and (
            type(v) is not int or v < spec.get("minimum", 0)
        ):
            raise InferenceStopped("invalid_structured_action_revision")
        if spec["type"] == "number" and (
            type(v) not in (int, float) or not math.isfinite(v)
            # Physical duration bounds are graded by the native tool, allowing
            # the model to recover from a well-formed but invalid decision.
        ):
            raise InferenceStopped("invalid_structured_action_duration")
    if value["tool"] != STEP:
        return [{"name": value["tool"], "arguments": "{}"}]
    native = next(
        t["function"]["parameters"] for t in tools if t["function"]["name"] == STEP
    )
    roles = sorted(native["properties"]["choices"]["properties"])
    if profile == "json_fixed_v1":
        if (
            type(current_revision) is not int
            or current_revision < 0
            or type(request_index) is not int
            or request_index < 0
        ):
            raise InferenceStopped("invalid_sdk_transaction_metadata")
        arguments = {
            "expected_revision": current_revision,
            "idempotency_key": f"llm-fixed/{current_revision}/{request_index}",
            "advance_sec": fixed_decision_sec,
        }
    else:
        arguments = {
            k: value[k] for k in ("expected_revision", "idempotency_key", "advance_sec")
        }
    arguments["choices"] = {
        role: [value[f"action_{i}"]] for i, role in enumerate(roles)
    }
    return [{"name": STEP, "arguments": canonical_json(arguments)}]


def wire_context(messages, tools, pairs, profile, fixed_decision_sec=5.0):
    selected, narrowed = inference_context(messages, tools, pairs, profile)
    schema = (
        action_schema(narrowed, fixed=profile == "json_fixed_v1")
        if profile in ("json_action_v1", "json_fixed_v1")
        else None
    )
    if profile == "json_fixed_v1":
        from space_env.llm.episode import STEP

        maximum = next(
            t["function"]["parameters"]["properties"]["advance_sec"]["maximum"]
            for t in tools
            if t["function"]["name"] == STEP
        )
        if (
            type(fixed_decision_sec) not in (int, float)
            or not math.isfinite(fixed_decision_sec)
            or not 1e-9 < fixed_decision_sec <= maximum
        ):
            raise ValueError("Fixed cadence outside declared step limits")
        selected[0]["content"] += (
            f"\nFixed-cadence transport: each joint step advances {fixed_decision_sec} seconds. The SDK supplies current revision and idempotency metadata; do not generate those fields. You choose every role action or finish. A fresh observation is supplied after every action, so no separate observe action is exposed in this profile."
        )

    if schema is not None:
        selected[0]["content"] += (
            "\nWire format: return exactly ONE JSON object matching the following schema; do not emit XML or native tool_calls. "
            "Each action_i is a scalar action ID for its described role. Supply all fields even for observe/finish; role actions "
            "are ignored for those two tools. This transport is restricted to one action per role per decision. Schema: "
            + canonical_json(schema)
        )
    return selected, narrowed, schema


class OllamaPolicy:
    def __init__(
        self,
        *,
        endpoint="http://127.0.0.1:11519",
        model="qwen3.5:4b",
        context_tokens=65536,
        output_tokens=256,
        max_requests=32,
        window_pairs=2,
        timeout_sec=120.0,
        max_wall_sec=600.0,
        protocol_profile="baseline_v1",
        fixed_decision_sec=5.0,
    ):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("Only explicit loopback HTTP endpoints are supported")
        if not isinstance(model, str) or not 1 <= len(model) <= 256:
            raise ValueError("Invalid model")
        if (
            any(
                type(v) is not int or v < 1
                for v in (context_tokens, output_tokens, max_requests)
            )
            or not output_tokens < context_tokens <= 262144
            or max_requests > 4096
        ):
            raise ValueError("Invalid inference limits")
        if not 0 < timeout_sec <= 600 or not 0 < max_wall_sec <= 86400:
            raise ValueError("Invalid inference time budgets")
        window_messages(
            [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            window_pairs,
        )
        if protocol_profile not in PROFILES:
            raise ValueError("Unknown inference protocol profile")
        self.protocol_profile = protocol_profile
        self.fixed_decision_sec = fixed_decision_sec
        self.endpoint, self.model = endpoint.rstrip("/"), model
        self.context_tokens, self.output_tokens = context_tokens, output_tokens
        self.max_requests, self.window_pairs = max_requests, window_pairs
        self.timeout_sec, self.max_wall_sec = timeout_sec, max_wall_sec
        self.trace = []
        self._pending = deque()
        self._started = None
        # Ignore ambient proxies for local inference. No credentials are read.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _emit(self):
        call, index = self._pending.popleft()
        self.trace[index].setdefault("delivered_calls", []).append(copy.deepcopy(call))
        return copy.deepcopy(call)

    def __call__(self, messages, tools):
        if self._started is None:
            self._started = time.monotonic()
        remaining = self.max_wall_sec - (time.monotonic() - self._started)
        if remaining <= 0:
            raise InferenceStopped("inference_wall_budget")
        if self._pending:
            return self._emit()
        if len(self.trace) >= self.max_requests:
            raise InferenceStopped("inference_request_budget")
        selected, selected_tools, structured_format = wire_context(
            messages,
            tools,
            self.window_pairs,
            self.protocol_profile,
            self.fixed_decision_sec,
        )
        # UTF-8 bytes conservatively bound text tokens; reserve additional template
        # overhead. This is an admission bound, not reported tokenizer usage.
        if (
            len(
                canonical_json({"messages": selected, "tools": selected_tools}).encode()
            )
            + self.output_tokens
            + 8192
            > self.context_tokens
        ):
            raise InferenceStopped("conservative_context_budget")
        payload = {
            "model": self.model,
            "messages": selected,
            "tools": selected_tools,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "seed": 0,
                "num_ctx": self.context_tokens,
                "num_predict": self.output_tokens,
            },
        }
        if structured_format is not None:
            payload["format"] = structured_format
            payload["tools"] = []
        request = urllib.request.Request(
            self.endpoint + "/api/chat",
            canonical_json(payload).encode(),
            {"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            with self._opener.open(
                request, timeout=min(self.timeout_sec, remaining)
            ) as response:
                body = response.read(4_000_001)
            if len(body) > 4_000_000:
                raise InferenceStopped("provider_response_byte_budget")
            result = json.loads(body)
            canonical_json(result)
        except urllib.error.HTTPError as exc:
            self.trace.append(
                {
                    "request": payload,
                    "error_type": type(exc).__name__,
                    "http_status": exc.code,
                    "provider_error_body": exc.read(16384).decode(errors="replace"),
                    "wall_sec": time.monotonic() - start,
                }
            )
            raise InferenceStopped("provider_http_error") from exc
        except Exception as exc:
            self.trace.append(
                {
                    "request": payload,
                    "error_type": type(exc).__name__,
                    "wall_sec": time.monotonic() - start,
                }
            )
            raise InferenceStopped("provider_transport_or_response_error") from exc
        self.trace.append(
            {
                "request": payload,
                "response": result,
                "wall_sec": time.monotonic() - start,
            }
        )
        if result.get("model") != self.model or result.get("done") is not True:
            raise InferenceStopped("provider_model_or_completion_mismatch")
        if result.get("done_reason") == "length":
            raise InferenceStopped("generation_token_budget")
        for key in ("prompt_eval_count", "eval_count"):
            if type(result.get(key)) is not int or result[key] < 0:
                raise InferenceStopped("provider_usage_missing")
        try:
            revision = (
                json.loads(selected[1]["content"])["current_revision"]
                if self.protocol_profile == "json_fixed_v1"
                else None
            )
            calls = decode_calls(
                result,
                selected_tools,
                self.protocol_profile,
                current_revision=revision,
                request_index=len(self.trace) - 1,
                fixed_decision_sec=self.fixed_decision_sec,
            )
        except InferenceStopped as exc:
            self.trace[-1]["decode_error"] = str(exc)
            raise
        self.trace[-1]["decoded_calls"] = copy.deepcopy(calls)
        self.trace[-1]["delivered_calls"] = []
        self._pending.extend((call, len(self.trace) - 1) for call in calls)
        return self._emit() if self._pending else None

    def metrics(self):
        responses = [t["response"] for t in self.trace if "response" in t]
        return {
            "requests": len(self.trace),
            "usage_incomplete": any(
                "response" not in t
                or any(
                    type(t["response"].get(k)) is not int or t["response"][k] < 0
                    for k in ("prompt_eval_count", "eval_count")
                )
                for t in self.trace
            ),
            "input_tokens": sum(
                r.get("prompt_eval_count", 0)
                for r in responses
                if type(r.get("prompt_eval_count")) is int
                and r["prompt_eval_count"] >= 0
            ),
            "output_tokens": sum(
                r.get("eval_count", 0)
                for r in responses
                if type(r.get("eval_count")) is int and r["eval_count"] >= 0
            ),
            "inference_wall_sec": sum(t["wall_sec"] for t in self.trace),
            "model": self.model,
            "temperature": 0.0,
            "seed": 0,
            "think": False,
            "context_tokens": self.context_tokens,
            "max_output_tokens_per_request": self.output_tokens,
            "context_window_pairs": self.window_pairs,
            "protocol_profile": self.protocol_profile,
            "fixed_decision_sec": self.fixed_decision_sec
            if self.protocol_profile == "json_fixed_v1"
            else None,
            "local_only": True,
            "token_counts_source": "Ollama response, summed across requests, including repeated context",
        }
