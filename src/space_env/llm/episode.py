"""Causal actor-only chat tools, cold replay and conditional SFT/episodic RL.

Privileged checkpoints belong to the audit record, never to messages. Action
catalogues are declared command templates, not free-form flight software control.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from space_env.planning.actor_projection import (
    actor_catalogues,
    actor_roles,
    project_actor_observation,
)
from space_env.runtime import RuntimeContractError
from space_env.simulation.snapshot import canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl
from space_env.tasks.schema import TaskDecision
from space_env.tasks.session import TaskSession, score

STEP = "space_env_actor_joint_step"
OBSERVE = "space_env_actor_observe"
FINISH = "space_env_actor_finish"
SYSTEM = "Operate the declared spacecraft mission using the provided tools. Select only declared actor action IDs. Observations are current local telemetry; masks are conservative, not guarantees. Operator events and hidden state are unavailable. Delivered results require native ground acknowledgement."


def seal(body):
    return body | {"sha256": digest(body)}


def sources():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def _json(text):
    def pairs(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError("duplicate JSON key")
            value[k] = v
        return value

    def constant(value):
        raise ValueError("nonfinite JSON")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class _PublicDecisionError(ValueError):
    def __init__(self, code, details):
        super().__init__(code)
        self.code, self.public_details = code, details


class LLMEpisode:
    """One owner. The caller supplies a provider-independent sequence of tool calls.

    The agent sees only actor projections. Benchmark labels/checkpoints/source
    recipes are trainer-side. All attempts, including invalid JSON, spend calls.
    """

    def __init__(
        self,
        session,
        *,
        task_text,
        group,
        split="train",
        max_calls=256,
        max_dialogue_bytes=16000000,
        policy_origin="unspecified",
    ):
        if not isinstance(session, TaskSession):
            raise TypeError("native TaskSession required")
        if (
            not isinstance(task_text, str)
            or not 1 <= len(task_text) <= 16384
            or not isinstance(group, str)
            or not 1 <= len(group) <= 256
            or split not in ("train", "development", "test")
        ):
            raise ValueError("Invalid episode task/group/split")
        if type(max_calls) is not int or not 1 <= max_calls <= 4096:
            raise ValueError("Invalid tool-call budget")
        if (
            type(max_dialogue_bytes) is not int
            or not 1024 <= max_dialogue_bytes <= 16000000
        ):
            raise ValueError("Invalid dialogue byte budget")
        if policy_origin not in (
            "unspecified",
            "scripted_public",
            "scripted_privileged",
            "external_llm",
            "human",
        ):
            raise ValueError("Invalid policy origin")
        self.max_dialogue_bytes, self.policy_origin = max_dialogue_bytes, policy_origin
        # Own a cold-replayed snapshot; subsequent caller mutation cannot change us.
        self._initial = session.checkpoint(compact=True)
        self._session = TaskSession.from_checkpoint(self._initial)
        self._spec = self._session._task
        self.roles = actor_roles(self._spec)
        if not self.roles:
            raise ValueError("At least one spacecraft actor role required")
        self._catalogues = actor_catalogues(self._spec, self.roles)
        self._task_text, self.group, self.split = task_text, group, split
        self.max_calls = max_calls
        self._sources = sources()
        self._events, self._ids = [], set()
        self._stop = None
        self._messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": canonical_json(
                    {"task": task_text, "observation": self.observe()}
                ),
            },
        ]
        if self._spec.navigation_sensors:
            self._messages[0]["content"] += (
                " Orbit coordinates are EKF estimates from received synthetic measurements, not truth. Covariance and delayed packets are provided. Action masks are declarative: native admission can reject. Other resource counters remain ideal."
            )
        if self._spec.goal is not None:
            self._messages[0]["content"] += (
                " Completion is defined by the declared native mission goals in local observations, not necessarily product delivery. Finish verifies those goals and physical budgets."
            )

        if len(canonical_json(self._messages).encode()) > self.max_dialogue_bytes:
            raise ValueError("Initial dialogue exceeds byte budget")

    def observe(self):
        return {
            "schema": "space-env.llm-actor-team/1",
            "revision": self._session.revision,
            "roles": {
                r: project_actor_observation(self._session.observe(r), self._spec, r)
                for r in self.roles
            },
        }

    @property
    def messages(self):
        return copy.deepcopy(self._messages)

    @property
    def done(self):
        return self._stop is not None or len(self._events) >= self.max_calls

    def tool_schemas(self):
        choices = {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": False,
            "properties": {
                r: {
                    "type": "array",
                    "maxItems": self._spec.roles[r].action_slots,
                    "items": {"type": "string", "enum": list(self._catalogues[r])},
                }
                for r in self.roles
            },
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": OBSERVE,
                    "description": "Read current actor-only telemetry without advancing time.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": STEP,
                    "description": "Execute one joint actor decision and advance physical time. Omitted roles wait. No operator event actions.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["choices", "expected_revision", "idempotency_key"],
                        "properties": {
                            "choices": choices,
                            "expected_revision": {"type": "integer", "minimum": 0},
                            "idempotency_key": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 256,
                            },
                            "advance_sec": {
                                "type": "number",
                                "exclusiveMinimum": 1e-9,
                                "maximum": self._spec.max_decision_sec,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": FINISH,
                    "description": (
                        "Claim completion of the declared physical mission goal. Native state and resource limits are checked without advancing time; incomplete claims are rejected."
                        if self._spec.goal is not None
                        else "Claim completion after native delivery/ACK of every mission request. Validates the goal without advancing time. Incomplete claims are rejected; no future state is disclosed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def invoke(self, name, arguments, *, call_id):
        if self.done:
            raise ValueError("LLM episode finished or call budget exhausted")
        if (
            not isinstance(call_id, str)
            or not 1 <= len(call_id) <= 128
            or call_id in self._ids
        ):
            raise ValueError("Unique nonempty tool call ID required")
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 128
            or not isinstance(arguments, str)
            or len(arguments.encode()) > 65536
        ):
            raise ValueError("Invalid tool call envelope")
        before_revision = self._session.revision
        try:
            args = _json(arguments)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object")  # noqa: TRY004
            if name == OBSERVE:
                if args:
                    raise ValueError("Observe accepts no arguments")
                response = {"ok": True, "observation": self.observe()}
            elif name == FINISH:
                if args:
                    raise ValueError("Finish accepts no arguments")
                life = self._session._runtime._backend.report().get("lifecycle") or {}
                requests = life.get("requests", {})
                complete = bool(requests) and all(
                    r["status"] == "completed" for r in requests.values()
                )
                if self._spec.goal is not None:
                    complete = self._session.goal_status()["success"]
                if complete:
                    self._stop = "verified_goal"
                    response = {
                        "ok": True,
                        "mission_complete": True,
                        "terminated": True,
                        "truncated": False,
                        "reward": 0.0,
                        "discount": 0.0,
                        "elapsed_sec": 0.0,
                        "observation": self.observe(),
                    }
                else:
                    response = {
                        "ok": False,
                        "error": {"code": "mission_incomplete"},
                        "observation": self.observe(),
                    }
            elif name == STEP:
                decision = TaskDecision.model_validate(args)
                if set(decision.choices) - set(self.roles) or any(
                    a not in self._catalogues[r]
                    for r, aa in decision.choices.items()
                    for a in aa
                ):
                    raise ValueError("Only declared actor actions are allowed")
                if len(decision.idempotency_key) > 256:
                    raise ValueError("Idempotency key too long")
                if (
                    decision.advance_sec is not None
                    and decision.advance_sec > self._spec.max_decision_sec
                ):
                    raise _PublicDecisionError(
                        "advance_duration_out_of_range",
                        {
                            "minimum_exclusive_sec": 1e-9,
                            "maximum_sec": self._spec.max_decision_sec,
                        },
                    )
                transition = self._session.step(**decision.model_dump())
                response = {
                    "ok": True,
                    **{
                        k: transition[k]
                        for k in (
                            "revision",
                            "elapsed_sec",
                            "reward",
                            "discount",
                            "terminated",
                            "truncated",
                            "feedback",
                        )
                    },
                    "observation": self.observe(),
                }
                if transition["terminated"] or transition["truncated"]:
                    self._stop = (
                        "terminated"
                        if transition["terminated"]
                        else "native_decision_budget"
                    )
            else:
                raise ValueError("Unknown tool")
        except (
            ValueError,
            ValidationError,
            RuntimeContractError,
            json.JSONDecodeError,
        ) as exc:
            response = {
                "ok": False,
                "error": {"code": getattr(exc, "code", "invalid_tool_call")},
                "observation": self.observe(),
            }
            if isinstance(exc, _PublicDecisionError):
                response["error"]["details"] = copy.deepcopy(exc.public_details)
            elif isinstance(exc, ValidationError):
                response["error"]["details"] = [
                    {"location": list(e["loc"]), "type": e["type"]}
                    for e in exc.errors()
                ]
        applied = self._session.revision != before_revision
        # Repeated native idempotency keys have no new physical transition/reward.
        event = {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "response": response,
            "transition_applied": applied,
            "training_reward": response.get("reward", 0.0) if applied else 0.0,
        }
        self._events.append(copy.deepcopy(event))
        self._ids.add(call_id)
        self._messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": canonical_json(response),
                },
            ]
        )
        if (
            not self._stop
            and len(canonical_json(self._messages).encode()) > self.max_dialogue_bytes
        ):
            self._stop = "dialogue_byte_budget"
        return copy.deepcopy(response)

    def record(self):
        if sources() != self._sources:
            raise ValueError("Source files changed during LLM episode")
        outcome = score(self._session._runtime)
        life = self._session._runtime._backend.report().get("lifecycle") or {}
        total = len(life.get("requests", {}))
        failed = sum(
            not e["response"]["ok"]
            or not e["response"].get("feedback", {}).get("accepted", True)
            for e in self._events
        )
        success = bool(
            total
            and outcome["completion"] == total
            and self._stop in ("terminated", "verified_goal")
        )
        goal = self._session.goal_status()
        if goal is not None:
            success = goal["success"] and self._stop in ("terminated", "verified_goal")
        record = {
            "schema": "space-env.llm-episode/1",
            "initial_checkpoint": self._initial,
            "configuration": {
                "task_text": self._task_text,
                "group": self.group,
                "split": self.split,
                "max_calls": self.max_calls,
                "max_dialogue_bytes": self.max_dialogue_bytes,
                "policy_origin": self.policy_origin,
            },
            "model_sources": self._sources,
            "tools": self.tool_schemas(),
            "events": copy.deepcopy(self._events),
            "messages": self.messages,
            "final_checkpoint_sha256": self._session.checkpoint(compact=True)["sha256"],
            "metrics": {
                "success": success,
                "fulfilled": outcome["completion"],
                "total": total,
                "weighted_fulfilled": outcome["priority"],
                "tool_calls": len(self._events),
                "failed_calls": failed,
                "native_decisions": sum(e["transition_applied"] for e in self._events),
                "return": sum(e["training_reward"] for e in self._events),
                "dialogue_utf8_bytes": len(canonical_json(self._messages).encode()),
                "stop_reason": self._stop
                or ("tool_budget" if self.done else "caller_stopped"),
                "sft_eligible": success
                and failed == 0
                and len(canonical_json(self._messages).encode())
                <= self.max_dialogue_bytes,
                "dialogue_budget_exceeded": len(canonical_json(self._messages).encode())
                > self.max_dialogue_bytes,
            },
            "scope": (
                "catalogue-constrained sensor-aware actor agent; estimated orbit and ideal operational counters; native replay is not flight qualification; no external LLM automatically called"
                if self._spec.navigation_sensors
                else "catalogue-constrained actor-only tool agent; native ideal local telemetry; successful replay is not expert qualification; no external LLM automatically called"
            ),
        }
        if goal is not None:
            record["metrics"]["goal"] = goal
        return seal(record)


def verify_record(record):
    if record.get("schema") != "space-env.llm-episode/1" or record.get(
        "sha256"
    ) != digest({k: v for k, v in record.items() if k != "sha256"}):
        raise ValueError("Invalid LLM record seal")
    replay = LLMEpisode(
        TaskSession.from_checkpoint(record["initial_checkpoint"]),
        **record["configuration"],
    )
    if replay._sources != record["model_sources"]:
        raise ValueError("Cold replay requires recorded source revision")
    for event in record["events"]:
        replay.invoke(event["name"], event["arguments"], call_id=event["call_id"])
    if replay.record() != record:
        raise ValueError("LLM dialogue/native replay mismatch")
    return {
        "verified": True,
        "events": len(record["events"]),
        "episode_sha256": record["sha256"],
    }


def _training_row(record, *, format="rl"):
    if format not in ("rl", "sft"):
        raise ValueError("format must be rl or sft")
    if record["configuration"]["split"] != "train":
        raise ValueError("Training export rejects development/test episodes")
    if format == "sft" and not record["metrics"]["sft_eligible"]:
        raise ValueError(
            "SFT requires complete successful native goal and no failed/rejected calls"
        )
    row = {
        "schema": "space-env.llm-training/1",
        "episode_sha256": record["sha256"],
        "group": record["configuration"]["group"],
        "split": "train",
        "format": format,
        "messages": copy.deepcopy(record["messages"]),
        "tools": copy.deepcopy(record["tools"]),
        "replay_verified": True,
        "quality": "successful_execution_not_expert"
        if format == "sft"
        else "behavior_rollout_including_failures",
        "target": "assistant_tool_calls_only; tool/user/system text are context, not SFT targets",
        "policy_origin": record["configuration"]["policy_origin"],
        "action_space": "operator_compiled_command_catalogue",
        "physics_profile": record["initial_checkpoint"]["scenario"]["physics_profile"],
    }
    row["message_loss_mask"] = (
        [m["role"] == "assistant" for m in row["messages"]] if format == "sft" else None
    )
    row["rl_group_id"] = digest(
        {
            "initial": record["initial_checkpoint"]["sha256"],
            "task": record["configuration"]["task_text"],
            "tools": record["tools"],
            "max_calls": record["configuration"]["max_calls"],
        }
    )
    row["assistant_message_indices"] = [2 + 2 * i for i in range(len(record["events"]))]
    if format == "rl":
        row.update(
            rewards=[e["training_reward"] for e in record["events"]],
            discounts=[
                0.0
                if e["response"].get("terminated")
                else (
                    e["response"].get("discount", 1.0)
                    if e["transition_applied"]
                    else 1.0
                )
                for e in record["events"]
            ],
            transition_applied=[e["transition_applied"] for e in record["events"]],
            metrics=record["metrics"],
        )
    return row


def export_training(record, output, *, format="rl"):
    verify_record(record)
    row = _training_row(record, format=format)
    return atomic_jsonl(
        output, [row], max_bytes=64_000_000, max_row_bytes=64_000_000, max_rows=1
    )
