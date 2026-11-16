"""Provider-independent JSON tools and verified episodic SFT/RL export.

No model inference or training is performed here. Privileged replay records
and training messages have deliberately separate schemas and contents.
"""

from __future__ import annotations

import copy
from pathlib import Path

from space_env.industrial.session import restore
from space_env.llm.episode import _json, sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

TOOL = "space_env_industrial_step"


class IndustrialEpisode:
    def __init__(
        self,
        session,
        *,
        split="train",
        policy_origin="unspecified",
        max_calls=256,
        task_text=None,
    ):
        if (
            split not in {"train", "development", "test"}
            or policy_origin
            not in {
                "unspecified",
                "scripted_public",
                "scripted_privileged",
                "external_llm",
                "human",
            }
            or type(max_calls) is not int
            or not 1 <= max_calls <= 512
        ):
            raise ValueError("invalid episode metadata or call budget")
        declared_split = getattr(session, "dataset_split", None)
        if declared_split is not None and split != declared_split:
            raise ValueError("episode split differs from generated scenario provenance")
        if task_text is not None and (
            not isinstance(task_text, str)
            or not task_text.strip()
            or len(task_text.encode("utf-8")) > 8192
        ):
            raise ValueError(
                "task_text must be a nonempty public brief up to 8192 bytes"
            )
        if session.revision != 0:
            raise ValueError("start from a fresh session")
        self._initial = session.checkpoint()
        self._session = restore(self._initial)
        self.split, self.policy_origin, self.max_calls = split, policy_origin, max_calls
        self.task_text = task_text
        self._events = []
        self._budget_penalty = getattr(self._session, "episode_budget_penalty", 0.0)
        self._ids = set()
        self._decision_keys = set()
        self._messages = [
            {
                "role": "system",
                "content": "Operate the declared research mission using the JSON step tool. Use current observations only. An emergency retreat does not complete an inspection; in an explicitly declared retreat task, a safe retreat is the objective. Profiles declare their fidelity explicitly; coupled native RF service and nonlinear central-orbit coasts do not imply flight certification.",
            },
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "observation": self.observe(),
                        "tool": self.tool_schema(),
                        **({"task_text": task_text} if task_text is not None else {}),
                        **(
                            {
                                "episode_budget": {
                                    "max_calls": self.max_calls,
                                    "incomplete_penalty": self._budget_penalty,
                                }
                            }
                            if self._budget_penalty
                            else {}
                        ),
                    }
                ),
            },
        ]

    def observe(self):
        return self._session.observe()

    def tool_schema(self):
        from space_env.industrial.campaign import CampaignAction
        from space_env.industrial.native_service import NativeServiceAction
        from space_env.industrial.operations import OperationsAction
        from space_env.industrial.orbits import OrbitAction
        from space_env.industrial.proximity import ProximityAction
        from space_env.industrial.service import ServiceAction

        action = {
            "native_campaign": CampaignAction,
            "operations": OperationsAction,
            "proximity": ProximityAction,
            "service": ServiceAction,
            "native_service": NativeServiceAction,
            "native_proximity": ProximityAction,
            "native_orbits": OrbitAction,
        }[self._session.kind]
        action_schema = action.model_json_schema()
        definitions = action_schema.pop("$defs", {})
        return {
            "name": TOOL,
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "minLength": 1},
                    "action": action_schema,
                },
                "required": ["expected_revision", "idempotency_key", "action"],
                "additionalProperties": False,
                "$defs": definitions,
            },
        }

    @property
    def done(self):
        return self._session._done or len(self._events) >= self.max_calls

    def call(self, call_id, arguments, *, name=TOOL):
        if self.done:
            raise ValueError("episode finished or call budget exhausted")
        if (
            not isinstance(call_id, str)
            or not 1 <= len(call_id) <= 128
            or call_id in self._ids
            or not isinstance(arguments, str)
            or len(arguments.encode()) > 131072
            or not isinstance(name, str)
            or len(name) > 128
        ):
            raise ValueError("invalid or duplicate tool call envelope")
        self._ids.add(call_id)
        try:
            if name != TOOL:
                raise ValueError("unknown tool")
            payload = _json(arguments)
            if not isinstance(payload, dict) or set(payload) != {
                "action",
                "expected_revision",
                "idempotency_key",
            }:
                raise ValueError("invalid tool payload")
            key = payload["idempotency_key"]
            if not isinstance(key, str) or key in self._decision_keys:
                raise ValueError("a model call must use a fresh decision key")
            response = self._session.step(**payload)
            self._decision_keys.add(key)
        except (ValueError, TypeError, KeyError):
            # Do not return backend exception strings, which may mention hidden
            # configuration. Invalid calls spend the model budget, not flight time.
            response = {
                "error": "invalid_decision",
                "revision": self._session.revision,
                "reward": -0.1,
                "elapsed_sec": 0.0,
            }
        if (
            self._budget_penalty
            and len(self._events) + 1 >= self.max_calls
            and not self._session._done
        ):
            response["reward"] -= self._budget_penalty
            response["episode_penalty"] = {
                "code": "incomplete_declared_task",
                "amount": -self._budget_penalty,
            }
        self._events.append(
            {
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "response": copy.deepcopy(response),
            }
        )
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
        return copy.deepcopy(response)

    def record(self):
        if self._initial["sources"] != sources():
            raise ValueError("source changed during episode")
        body = {
            "schema": "space-env.industrial-episode/1",
            "initial": copy.deepcopy(self._initial),
            "final": self._session.checkpoint(),
            "split": self.split,
            "policy_origin": self.policy_origin,
            "max_calls": self.max_calls,
            "events": copy.deepcopy(self._events),
            "messages": copy.deepcopy(self._messages),
            "metrics": self._session.metrics(),
            "call_budget_exhausted": len(self._events) >= self.max_calls
            and not self._session._done,
            "inference_attested": False,
            "training_attested": False,
        }
        if self.task_text is not None:
            body["task_text"] = self.task_text
        return {**body, "sha256": digest(body)}


def verify_episode(record):
    if (
        not isinstance(record, dict)
        or record.get("schema") != "space-env.industrial-episode/1"
        or record.get("sha256")
        != digest({k: v for k, v in record.items() if k != "sha256"})
    ):
        raise ValueError("invalid industrial episode seal")
    session = restore(record["initial"])
    episode = IndustrialEpisode(
        session,
        split=record["split"],
        policy_origin=record["policy_origin"],
        max_calls=record["max_calls"],
        task_text=record.get("task_text"),
    )
    if (
        not isinstance(record["events"], list)
        or len(record["events"]) > episode.max_calls
    ):
        raise ValueError("invalid tool-call history")
    for event in record["events"]:
        result = episode.call(event["call_id"], event["arguments"], name=event["name"])
        if canonical_json(result) != canonical_json(event["response"]):
            raise ValueError("tool response replay mismatch")
    if canonical_json(episode.record()) != canonical_json(record):
        raise ValueError("episode replay mismatch")
    return {
        "valid": True,
        "calls": len(record["events"]),
        "kind": session.kind,
        "success": record["metrics"]["success"],
        **(
            {"task_accepted": record["metrics"]["task_acceptance"]["accepted"]}
            if "task_acceptance" in record["metrics"]
            else {}
        ),
    }


def training_success(record, *, objective="native"):
    """Select the label on a verified record; export_training always replays first."""
    if objective == "native":
        return record["metrics"]["success"]
    if objective != "declared":
        raise ValueError("objective must be native or declared")
    if (
        not record["initial"]["configuration"].get("acceptance")
        or "task_acceptance" not in record["metrics"]
    ):
        raise ValueError("declared objective was not configured at episode start")
    return (
        record["metrics"]["task_acceptance"]["accepted"] is True
        and not record["call_budget_exhausted"]
    )


def export_training(record, output, *, format="rl", objective="native"):
    verify_episode(record)
    if format not in {"sft", "rl"} or record["split"] != "train":
        raise ValueError("training export requires train split and sft/rl format")
    selected_success = training_success(record, objective=objective)
    if (
        format == "sft"
        and (
            not selected_success
            or any(
                "error" in e["response"]
                or e["response"].get("feedback", {}).get("accepted") is False
                for e in record["events"]
            )
        )
    ) or record["policy_origin"] == "unspecified":
        raise ValueError(
            "SFT needs successful error-free attempts; policy provenance must be explicit"
        )
    path = Path(output)
    if path.exists():
        raise ValueError("choose a new training output")
    # Only causal messages and scalar rewards cross into training rows.
    rewards = [e["response"]["reward"] for e in record["events"]]
    row = {
        "schema": "space-env.industrial-training/1",
        "format": format,
        "kind": record["initial"]["kind"],
        "source_record_sha256": record["sha256"],
        "messages": record["messages"],
        "policy_origin": record["policy_origin"],
        "rewards": rewards,
        "return": sum(rewards),
        "terminal": record["metrics"]["terminal"],
        "call_budget_exhausted": record["call_budget_exhausted"],
        "success": record["metrics"]["success"],
    }
    if objective == "declared":
        row.update(
            schema="space-env.industrial-objective-training/1",
            success=bool(selected_success),
            native_success=record["metrics"]["success"],
            success_basis="declared_task_acceptance",
            task_objective=record["initial"]["configuration"]["acceptance"],
            task_status=record["metrics"]["task_acceptance"],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, canonical_json(row) + "\n")
    return {"path": str(path), "rows": 1, "format": format}
