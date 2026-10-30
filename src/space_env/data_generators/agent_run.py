"""Replayable mission-tool dialogues and verified causal training context."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

from space_env.data_generators.portable_run import (
    _differences,
    _outcome,
    runtime_manifest,
)
from space_env.eval.statistics import assign_split
from space_env.mission.agent_protocol import invoke_tool, tool_schemas
from space_env.mission.tools import MissionTools
from space_env.simulation.snapshot import atomic_write, canonical_json, digest, encode


def _state(tools: MissionTools) -> dict:
    return encode({"revision": tools.revision, "calls": tools.calls, "call_budget": tools.call_budget,
                   "pending": [asdict(command) for command in tools._pending],
                   "stream": tools.stream.capture(),
                   "channels": {name: stream.capture() for name, stream in tools._channels.items()},
                   "world": _outcome(tools.session.world)})


def _content(value):
    body = copy.deepcopy(value)
    # Strip only known envelope checksums. User event/product fields named sha256
    # remain semantic data and must still be compared.
    body.pop("sha256", None)
    body["session"].pop("sha256", None)
    body["session"]["world"].pop("sha256", None)
    return body


class AgentRunRecorder:
    """Single-owner tool recorder. All actions must use invoke while recording.

    Task context is explicitly supplied, not extracted from hidden fault causes.
    Known requests are declared in context; this is not an unknown-demand profile.
    """

    def __init__(self, tools: MissionTools, *, task: str, lineage: str, provenance: dict | None = None):
        if not isinstance(task, str) or not task.strip() or not isinstance(lineage, str) or not lineage.strip():
            raise ValueError("task and scenario lineage must be nonempty strings")
        self.tools = tools
        self.lineage = lineage
        self.provenance = copy.deepcopy(provenance or {})
        canonical_json(self.provenance)
        self.context = encode({"instruction": task, "requests": [asdict(item) for item in tools.session.requests],
                               "actors": list(tools.session.world.engines),
                               "horizon_sec": tools.session.world.horizon_sec,
                               "observation_contract": "mission-tools-sensor/1; current-validity-masks; nominal-geometry-forecast; declared-known-demand",
                               "protocol": "mission-tools/1", "tools": tool_schemas()})
        self.runtime = runtime_manifest()
        self.initial = tools.capture()
        self.events: list[dict] = []
        self._ids: set[str] = set()

    def invoke(self, name: str, arguments: dict, *, call_id: str | None = None) -> dict:
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError("tool name must be a string and arguments an object")
        call_id = f"call-{len(self.events)}" if call_id is None else call_id
        if not isinstance(call_id, str) or not call_id or call_id in self._ids:
            raise ValueError("tool call ID must be nonempty and unique")
        arguments = copy.deepcopy(arguments)
        canonical_json(arguments)
        response = encode(invoke_tool(self.tools, name, arguments))
        event = {"index": len(self.events), "call_id": call_id, "name": name,
                 "arguments": arguments, "response": response, "after": _state(self.tools)}
        canonical_json(event)
        self.events.append(event)
        self._ids.add(call_id)
        return copy.deepcopy(response)

    def save(self, path: str | Path) -> dict:
        body = {"schema": "space-env.agent-run/1", "lineage": self.lineage, "context": self.context,
                "provenance": self.provenance, "runtime": self.runtime, "initial": self.initial,
                "events": self.events, "final": self.tools.capture()}
        record = {**body, "sha256": digest(body)}
        atomic_write(path, canonical_json(record))
        return copy.deepcopy(record)


def read_agent_run(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "sha256"}
    if record.get("schema") != "space-env.agent-run/1" or record.get("sha256") != digest(body):
        raise ValueError("unsupported or corrupt agent run")
    if not isinstance(record.get("lineage"), str) or not record["lineage"].strip():
        raise ValueError("agent run needs scenario lineage")
    if record["context"].get("protocol") != "mission-tools/1" or record["context"].get("tools") != tool_schemas():
        raise ValueError("unsupported mission tool protocol")
    return record


def _verify(record: dict) -> dict:
    if record["runtime"] != runtime_manifest():
        return {"ok": False, "reason": "runtime_mismatch", "events_verified": 0}
    tools = MissionTools.from_snapshot(record["initial"])
    context = record["context"]
    if not isinstance(context.get("instruction"), str) or not context["instruction"].strip():
        raise ValueError("agent context needs a task instruction")
    declared = {"requests": encode([asdict(item) for item in tools.session.requests]),
                "actors": list(tools.session.world.engines), "horizon_sec": tools.session.world.horizon_sec,
                "observation_contract": "mission-tools-sensor/1; current-validity-masks; nominal-geometry-forecast; declared-known-demand"}
    if any(context.get(key) != value for key, value in declared.items()):
        raise ValueError("agent context differs from initial declared mission")
    # Verify nested hashes and reconstructibility, not just the outer run checksum.
    MissionTools.from_snapshot(record["final"])
    ids = set()
    for index, event in enumerate(record["events"]):
        ident = event["call_id"]
        if event["index"] != index or not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("invalid agent event sequence or call identity")
        ids.add(ident)
        actual = encode(invoke_tool(tools, event["name"], event["arguments"]))
        differences = _differences(event["response"], actual, path=f"events[{index}].response")
        differences.extend(_differences(event["after"], _state(tools), path=f"events[{index}].after"))
        if differences:
            return {"ok": False, "reason": "event_mismatch", "events_verified": index, "mismatches": differences[:20]}
    differences = _differences(_content(record["final"]), _content(tools.capture()), path="final")
    return {"ok": not differences, "reason": "final_mismatch" if differences else "verified",
            "events_verified": len(record["events"]), "mismatches": differences,
            "run_sha256": record["sha256"], "lineage": record["lineage"]}


def replay_agent_run(path: str | Path) -> dict:
    return _verify(read_agent_run(path))


def export_tool_dialogue(run_path: str | Path, output: str | Path) -> dict:
    """One complete causal conversation, without verifier state in model inputs.

    Successful replay does not qualify the teacher. Provenance and failed calls
    remain visible. No reward, expert label or safety certification is inferred.
    """
    record = read_agent_run(run_path)
    verified = _verify(record)
    if not verified["ok"]:
        raise ValueError(f"agent run replay failed: {verified['reason']}")
    context = {key: value for key, value in record["context"].items() if key != "tools"}
    messages = [{"role": "user", "content": canonical_json(context)}]
    for event in record["events"]:
        messages.append({"role": "assistant", "tool_calls": [{"id": event["call_id"], "type": "function",
                         "function": {"name": event["name"], "arguments": canonical_json(event["arguments"])}}]})
        messages.append({"role": "tool", "tool_call_id": event["call_id"],
                         "content": canonical_json(event["response"])})
    row = {"schema": "space-env.tool-dialogue/1", "lineage": record["lineage"],
           "split": assign_split(record["lineage"], ""), "run_sha256": record["sha256"],
           "tools": record["context"]["tools"], "messages": messages, "provenance": record["provenance"],
           "replay_verified": True}
    content = canonical_json(row) + "\n"
    atomic_write(output, content)
    import hashlib

    return {"rows": 1, "messages": len(messages), "events": len(record["events"]),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "split": row["split"], "run_sha256": record["sha256"]}


def record_baseline(
    tools: MissionTools, output: str | Path, *, lineage: str,
    policy: str = "priority", step_sec: float = 30., max_steps: int = 100,
) -> dict:
    """A bounded, recorded heuristic teacher using exactly the public tool protocol.

    Prefer downlink when telemetry reports buffered data and a current contact;
    otherwise submit one chosen released request per eligible actor. This is not a
    resource-constrained solver or a claim of optimality. Budget failures stay in
    the event log; no hidden bypass is used to finish the episode.
    """
    import math

    from space_env.mission.tools import choose_request

    if policy not in {"fifo", "deadline", "priority", "rolling"}:
        raise ValueError("unsupported baseline policy")
    if not math.isfinite(step_sec) or step_sec <= 0 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("positive finite step and positive integer max_steps are required")
    recorder = AgentRunRecorder(
        tools, task="Collect declared mission products and deliver them before their deadlines.", lineage=lineage,
        provenance={"teacher": "tool-heuristic/1", "policy": policy, "expert": False},
    )
    for index in range(max_steps):
        remaining = tools.session.world.horizon_sec - tools.session.world.now_sec()
        if remaining <= 1e-9:
            break
        observed = recorder.invoke("observe", {})
        if not observed["ok"]:
            break
        view = observed["result"]
        chosen = choose_request(view["requests"], policy, now_sec=view["now_sec"], horizon_sec=600)
        batch = []
        for actor, reading in view["actors"].items():
            if reading.get("missing"):
                continue
            masks = view["masks"][actor]
            if masks["comms"] and reading.get("stored_bits", 0) > 0:
                batch.append({"command_id": f"baseline-{index}-{actor}", "actor_id": actor, "kind": "comms",
                              "duration_sec": min(step_sec, remaining)})
            elif chosen is not None and chosen["id"] in masks["imaging"]:
                batch.append({"command_id": f"baseline-{index}-{actor}", "actor_id": actor, "kind": "imaging",
                              "request_id": chosen["id"]})
        if batch and not recorder.invoke("submit", {"commands": batch})["ok"]:
            break
        if not recorder.invoke("advance", {"dt_sec": min(step_sec, remaining), "expected_revision": view["revision"]})["ok"]:
            break
    record = recorder.save(output)
    return {"run_sha256": record["sha256"], "events": len(record["events"]),
            "failed_calls": sum(not event["response"]["ok"] for event in record["events"]),
            "now_sec": tools.session.world.now_sec(), "revision": tools.revision}
