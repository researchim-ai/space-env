"""Bounded flat incident ancestry; no recursive records or source PID coupling."""

from __future__ import annotations

import copy

from space_env.planning.chain import verify_chain_proposal
from space_env.planning.incident import _recover_epoch
from space_env.planning.mission_plan import (
    PlanExecution,
    _sealed,
    make_mission_plan,
    validate_plan,
)
from space_env.simulation.snapshot import canonical_json, digest

MAX_EPOCHS = 8
MAX_JOURNAL_BYTES = 64_000_000


def _seal(body):
    value = body | {"sha256": digest(body)}
    if len(canonical_json(value).encode()) > MAX_JOURNAL_BYTES:
        raise ValueError("incident journal byte budget exceeded")
    return copy.deepcopy(value)


def create_journal(proposal):
    verify_chain_proposal(proposal, native=True)
    return _seal(
        {
            "schema": "space-env.incident-journal/1",
            "origin": copy.deepcopy(proposal),
            "epochs": [],
        }
    )


def current_plan(journal):
    return copy.deepcopy(
        journal["epochs"][-1]["result"]["plan"]
        if journal["epochs"]
        else journal["origin"]["plan"]
    )


def _fork(base, edits, start):
    if type(start) is not int or not 0 <= start <= len(base["steps"]):
        raise ValueError("invalid journal prefix boundary")
    steps = [
        {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")}
        for s in base["steps"]
    ]
    if not isinstance(edits, list) or start + len(edits) > len(steps):
        raise ValueError("journal edits exceed full horizon")
    actions = {a["id"]: a for a in base["task"]["actions"]}
    for i, row in enumerate(edits, start):
        if (
            set(row) != {"at_sec", "advance_sec", "action_ids"}
            or row["at_sec"] != steps[i]["at_sec"]
            or row["advance_sec"] != steps[i]["advance_sec"]
            or any(
                a != "wait" and (a not in actions or actions[a]["kind"] == "inject")
                for a in row["action_ids"]
            )
        ):
            raise ValueError(
                "journal actor edits cannot change clock or inject operator events"
            )
        steps[i] = copy.deepcopy(row)
    return make_mission_plan(
        base["scenario"],
        base["task"],
        steps,
        goals=base["goals"],
        episode_seed=base["episode_seed"],
    )


def _core(result):
    body = {
        k: v
        for k, v in result.items()
        if k not in {"source_proposal", "source_record", "sha256"}
    }
    body["schema"] = "space-env.incident-epoch/1"
    body["scope"] = "flat journal epoch; native committed prefix, fixed original demand"
    return _seal(body)


def verify_journal(value, *, native=False):
    if type(native) is not bool:
        raise ValueError("native journal audit flag must be boolean")
    if len(canonical_json(value).encode()) > MAX_JOURNAL_BYTES:
        raise ValueError("incident journal byte budget exceeded")
    if (
        not _sealed(value)
        or set(value) != {"schema", "origin", "epochs", "sha256"}
        or value["schema"] != "space-env.incident-journal/1"
        or not isinstance(value["epochs"], list)
        or len(value["epochs"]) > MAX_EPOCHS
    ):
        raise ValueError("invalid incident journal envelope/epoch budget")
    verify_chain_proposal(value["origin"], native=native)
    base = value["origin"]["plan"]
    cutoff = 0
    previous = value["origin"]["sha256"]
    for number, node in enumerate(value["epochs"]):
        if (
            not _sealed(node)
            or set(node)
            != {
                "index",
                "parent_sha256",
                "parent_plan_sha256",
                "cursor",
                "actor_edits",
                "source_episode_sha256",
                "source_physical_sha256",
                "incident",
                "settings",
                "result",
                "sha256",
            }
            or type(node["index"]) is not int
            or node["index"] != number + 1
            or node["parent_sha256"] != previous
            or node["parent_plan_sha256"] != base["sha256"]
            or type(node["cursor"]) is not int
            or not cutoff <= node["cursor"] < len(base["steps"])
            or len(node["actor_edits"]) != node["cursor"] - cutoff
            or not _sealed(node["result"])
        ):
            raise ValueError("incident journal ancestry/cursor mismatch")
        fork = _fork(base, node["actor_edits"], cutoff)
        result = node["result"]
        validate_plan(result["plan"])
        if (
            result["schema"] != "space-env.incident-epoch/1"
            or result["committed_prefix_steps"] <= node["cursor"]
            or result["source_prefix_sha256"] != node["source_physical_sha256"]
            or canonical_json(result["plan"]["steps"][: node["cursor"]])
            != canonical_json(fork["steps"][: node["cursor"]])
        ):
            raise ValueError("incident journal immutable prefix mismatch")
        if native:
            execution = PlanExecution(fork)
            execution.run(decisions=node["cursor"])
            record = execution.checkpoint()
            if (
                record["episode"]["sha256"] != node["source_episode_sha256"]
                or digest(record["episode"]["runtime"]["backend_report"])
                != node["source_physical_sha256"]
            ):
                raise ValueError("incident journal source prefix native mismatch")
            context = {"plan": fork, "source_task": value["origin"]["source_task"]}
            fresh = _core(
                _recover_epoch(
                    context,
                    record,
                    node["incident"],
                    settings=node["settings"],
                    history=number > 0,
                )
            )
            if canonical_json(fresh) != canonical_json(result):
                raise ValueError("incident journal epoch native mismatch")
        base = result["plan"]
        cutoff = result["committed_prefix_steps"]
        previous = node["sha256"]
    return {
        "integrity_passed": True,
        "native_epochs_recomputed": native,
        "epochs": len(value["epochs"]),
        "current_plan_sha256": base["sha256"],
        "committed_prefix_steps": cutoff,
        "expert_teacher_qualified": False,
    }


def append_incident(journal, record, incident=None, *, settings=None):
    verify_journal(journal, native=True)
    if len(journal["epochs"]) >= MAX_EPOCHS:
        raise ValueError("incident journal epoch budget exceeded")
    base = current_plan(journal)
    cutoff = (
        journal["epochs"][-1]["result"]["committed_prefix_steps"]
        if journal["epochs"]
        else 0
    )
    rows = record["episode"]["transitions"]
    cursor = len(rows)
    if not cutoff <= cursor < len(base["steps"]):
        raise ValueError("incident journal cannot rewind committed history")
    edits = []
    for i, row in enumerate(rows[cutoff:], cutoff):
        req = row["request"]
        if set(req["choices"]) != {"@central"}:
            raise ValueError("journal execution requires compiled joint requests")
        edits.append(
            {
                "at_sec": base["steps"][i]["at_sec"],
                "advance_sec": req["advance_sec"],
                "action_ids": req["choices"]["@central"],
            }
        )
    fork = _fork(base, edits, cutoff)
    verified = PlanExecution.from_checkpoint(fork, record)
    if verified.cursor != cursor:
        raise ValueError("journal source cursor mismatch")
    context = {"plan": fork, "source_task": journal["origin"]["source_task"]}
    result = _core(
        _recover_epoch(
            context,
            record,
            incident,
            settings=settings,
            history=bool(journal["epochs"]),
        )
    )
    previous = (
        journal["epochs"][-1]["sha256"]
        if journal["epochs"]
        else journal["origin"]["sha256"]
    )
    node = _seal(
        {
            "index": len(journal["epochs"]) + 1,
            "parent_sha256": previous,
            "parent_plan_sha256": base["sha256"],
            "cursor": cursor,
            "actor_edits": edits,
            "source_episode_sha256": record["episode"]["sha256"],
            "source_physical_sha256": digest(
                record["episode"]["runtime"]["backend_report"]
            ),
            "incident": copy.deepcopy(result["incident"]),
            "settings": result["schedule"]["settings"],
            "result": result,
        }
    )
    return _seal(
        {
            "schema": "space-env.incident-journal/1",
            "origin": journal["origin"],
            "epochs": [*journal["epochs"], node],
        }
    )
