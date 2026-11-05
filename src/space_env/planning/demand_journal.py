"""Real native demand injection and versioned, immutable executable catalogues."""

from __future__ import annotations

import copy

from pydantic import Field

from space_env.planning.chain import profile, verify_chain_proposal
from space_env.planning.demand_contract import (
    DemandEvent,
    DemandGoals,
    DemandPayload,
    effective_scenario,
)
from space_env.planning.incident import _recover_epoch, verify_incident
from space_env.planning.incident_journal import _fork, current_plan, verify_journal
from space_env.planning.mission_plan import PlanExecution, _sealed, validate_plan
from space_env.reference.lifecycle import LifecycleSpec, graph
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec

MAX_DEMAND_EPOCHS = 4
MAX_DEMAND_BYTES = 64_000_000


class DemandBatch(StrictModel):
    additions: tuple[DemandPayload, ...] = Field(min_length=1, max_length=4)


def _seal(body):
    result = body | {"sha256": digest(body)}
    if len(canonical_json(result).encode()) > MAX_DEMAND_BYTES:
        raise ValueError("demand journal byte budget exceeded")
    return copy.deepcopy(result)


def _base(reference, native):
    schema = reference.get("schema")
    if schema == "space-env.chain-proposal/1":
        verify_chain_proposal(reference, native=native)
        return reference, reference["plan"], 0
    if schema == "space-env.incident-journal/1":
        audit = verify_journal(reference, native=native)
        return (
            reference["origin"],
            current_plan(reference),
            audit["committed_prefix_steps"],
        )
    if schema == "space-env.incident-recovery/1":
        if not native:
            if not _sealed(reference):
                raise ValueError("invalid incident demand base seal")
            validate_plan(reference["plan"])
        else:
            verify_incident(reference)
        return (
            reference["source_proposal"],
            reference["plan"],
            reference["committed_prefix_steps"],
        )
    raise ValueError("unsupported demand journal base")


def _version(plan, generation):
    return digest(
        {
            "schema": "space-env.catalogue-version/1",
            "generation": generation,
            "task_sha256": digest(plan["task"]),
        }
    )


def _bound_source(origin):
    scenario = copy.deepcopy(origin["plan"]["scenario"])
    task = copy.deepcopy(origin["source_task"])
    chains = profile(scenario, task)[-1]
    owners = {aid: c["id"] for c in chains for aid in c["actions"].values()}
    for a in task["actions"]:
        a["request_ref"] = owners[a["id"]]
    return scenario, TaskSpec.model_validate(task).model_dump(
        mode="json", by_alias=True
    )


def create_demand_journal(reference):
    origin, plan, _ = _base(reference, True)
    _bound_source(origin)
    return _seal(
        {
            "schema": "space-env.demand-journal/1",
            "origin": origin,
            "base_reference": copy.deepcopy(reference),
            "epochs": [],
            "catalogue_version": _version(plan, 0),
        }
    )


def current_demand_plan(journal):
    if journal["epochs"]:
        return copy.deepcopy(journal["epochs"][-1]["result"]["plan"])
    return copy.deepcopy(_base(journal["base_reference"], False)[1])


def _augment(base, source_task, events, payloads, at_sec):
    new_events = [
        DemandEvent(at_sec=at_sec, payload=p).model_dump(mode="json") for p in payloads
    ]
    cumulative = [*events, *new_events]
    intent = effective_scenario(base, cumulative)
    task = copy.deepcopy(source_task)
    added = []
    targets = {t["id"] for t in base["definition"]["acquisition_targets"]}
    for index, payload in enumerate(payloads):
        p = DemandPayload.model_validate(payload)
        r = p.request
        if r.target_id not in targets or len(r.eligible_actors) != 1:
            raise ValueError("demand needs one existing actor/target")
        if len(p.jobs) != 1:
            raise ValueError("demand planner requires one metadata job")
        a = r.eligible_actors[0]
        job = p.jobs[0]
        if a not in base["definition"]["orbits"]:
            raise ValueError("unknown demand actor")
        role = next((k for k, v in task["roles"].items() if v["actors"] == [a]), None)
        if role is None:
            raise ValueError("demand actor has no policy role")
        tag = digest({"request": r.model_dump(mode="json"), "at_sec": at_sec})[:20]
        for kind, fields, duration in [
            ("slew", {"slew": {"frame": "GCRS", "target_id": r.target_id}}, 2.0),
            ("imaging", {"request_id": r.id}, r.acquisition_sec),
            ("transform", {"job_id": job.id}, job.work_sec),
            (
                "downlink",
                {
                    "product_id": r.delivery_product_id,
                    "destination_id": r.destination_id,
                },
                2.0,
            ),
        ]:
            aid = f"@new/{tag}/{index}/{kind}"
            action = {
                "id": aid,
                "role": role,
                "kind": "submit",
                "request_ref": r.id,
                "commands": [
                    {
                        "command_id": aid,
                        "actor_id": a,
                        "kind": kind,
                        "duration_sec": duration,
                        **fields,
                    }
                ],
            }
            task["actions"].append(action)
            added.append(action)
    task["max_actions"] = len(task["actions"]) + 1
    task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    life = LifecycleSpec.model_validate(intent["definition"]["lifecycle"])
    graph({r.id: r for r in life.requests}, {j.id: j for j in life.jobs})
    profile(intent, task)
    return intent, task, cumulative, added


def _compile(origin, base_plan, record, source_task, events, batch, settings, history):
    verified = PlanExecution.from_checkpoint(base_plan, record)
    runtime = verified._session._runtime
    active = any(
        c["status"] in {"accepted", "running"} for c in runtime.command_status()
    )
    at = runtime.time_sec + (source_task["decision_sec"] if active else 0.0)
    payloads = [
        p.model_dump(mode="json") for p in DemandBatch.model_validate(batch).additions
    ]
    intent, task, cumulative, added = _augment(
        origin["plan"]["scenario"], source_task, events, payloads, at
    )
    goals = copy.deepcopy(base_plan["goals"])
    goals["demand"] = {"events": cumulative, "min_fulfilled": len(cumulative)}
    goals["min_fulfilled"] = max(
        goals["min_fulfilled"],
        origin["plan"]["goals"]["min_fulfilled"] + len(cumulative),
    )
    DemandGoals.model_validate(goals["demand"])
    context = {"plan": base_plan, "source_task": task}
    result = _recover_epoch(
        context,
        record,
        settings=settings,
        history=history,
        intent_scenario=intent,
        demand_updates=payloads,
        catalogue_additions=added,
        demand_goals=goals,
    )
    core = {
        k: v
        for k, v in result.items()
        if k not in {"source_proposal", "source_record", "sha256"}
    }
    core.update(
        schema="space-env.demand-epoch/1",
        scope="native request additions; immutable prefix, expanded goal denominator",
        source_task=task,
        intent_sha256=digest(intent),
    )
    return _seal(core), cumulative


def verify_demand_journal(value, *, native=False):
    if type(native) is not bool:
        raise ValueError("native demand audit flag must be boolean")
    if (
        len(canonical_json(value).encode()) > MAX_DEMAND_BYTES
        or not _sealed(value)
        or set(value)
        != {
            "schema",
            "origin",
            "base_reference",
            "epochs",
            "catalogue_version",
            "sha256",
        }
        or value["schema"] != "space-env.demand-journal/1"
        or not isinstance(value["epochs"], list)
        or len(value["epochs"]) > MAX_DEMAND_EPOCHS
    ):
        raise ValueError("invalid demand journal/budget")
    origin, base, cutoff = _base(value["base_reference"], native)
    if canonical_json(origin) != canonical_json(value["origin"]):
        raise ValueError("demand origin mismatch")
    _, task = _bound_source(origin)
    events = []
    previous = value["base_reference"]["sha256"]
    version = _version(base, 0)
    for number, node in enumerate(value["epochs"]):
        if (
            not _sealed(node)
            or set(node)
            != {
                "index",
                "parent_sha256",
                "parent_plan_sha256",
                "parent_catalogue_version",
                "cursor",
                "actor_edits",
                "source_episode_sha256",
                "source_physical_sha256",
                "batch",
                "settings",
                "result",
                "catalogue_version",
                "sha256",
            }
            or type(node["index"]) is not int
            or node["index"] != number + 1
            or node["parent_sha256"] != previous
            or node["parent_plan_sha256"] != base["sha256"]
            or node["parent_catalogue_version"] != version
            or type(node["cursor"]) is not int
            or not cutoff <= node["cursor"] < len(base["steps"])
            or len(node["actor_edits"]) != node["cursor"] - cutoff
            or not _sealed(node["result"])
        ):
            raise ValueError("demand journal ancestry/cursor mismatch")
        fork = _fork(base, node["actor_edits"], cutoff)
        core = node["result"]
        validate_plan(core["plan"])
        if (
            core["schema"] != "space-env.demand-epoch/1"
            or core["committed_prefix_steps"] <= node["cursor"]
            or canonical_json(core["plan"]["steps"][: node["cursor"]])
            != canonical_json(fork["steps"][: node["cursor"]])
            or node["catalogue_version"] != _version(core["plan"], number + 1)
        ):
            raise ValueError("demand immutable prefix/catalogue mismatch")
        declared = core["plan"]["goals"]["demand"]["events"]
        additions = DemandBatch.model_validate(node["batch"]).additions
        if len(declared) != len(events) + len(additions):
            raise ValueError("demand event count mismatch")
        intent, expected_task, expected_events, _ = _augment(
            origin["plan"]["scenario"],
            task,
            events,
            [p.model_dump(mode="json") for p in additions],
            declared[len(events)]["at_sec"],
        )
        old_actions = {a["id"]: a for a in base["task"]["actions"]}
        new_actions = {a["id"]: a for a in core["plan"]["task"]["actions"]}
        if (
            canonical_json(expected_events) != canonical_json(declared)
            or canonical_json(expected_task) != canonical_json(core["source_task"])
            or core["intent_sha256"] != digest(intent)
            or core["forecast"]["scenario_sha256"] != digest(intent)
            or core["forecast"]["task_sha256"] != digest(expected_task)
            or canonical_json(core["plan"]["scenario"])
            != canonical_json(base["scenario"])
            or any(new_actions.get(k) != v for k, v in old_actions.items())
        ):
            raise ValueError("demand intent/catalogue lineage mismatch")
        if native:
            x = PlanExecution(fork)
            x.run(decisions=node["cursor"])
            record = x.checkpoint()
            if (
                record["episode"]["sha256"] != node["source_episode_sha256"]
                or digest(record["episode"]["runtime"]["backend_report"])
                != node["source_physical_sha256"]
            ):
                raise ValueError("demand native source prefix mismatch")
            fresh, new_events = _compile(
                origin,
                fork,
                record,
                task,
                events,
                node["batch"],
                node["settings"],
                history="@incident" in fork["task"]["roles"],
            )
            if canonical_json(fresh) != canonical_json(core):
                raise ValueError("demand epoch native mismatch")
            events = new_events
        else:
            events = core["plan"]["goals"]["demand"]["events"]
        task = core["source_task"]
        base = core["plan"]
        cutoff = core["committed_prefix_steps"]
        version = node["catalogue_version"]
        previous = node["sha256"]
    if version != value["catalogue_version"]:
        raise ValueError("demand current catalogue version mismatch")
    return {
        "integrity_passed": True,
        "native_epochs_recomputed": native,
        "epochs": len(value["epochs"]),
        "committed_prefix_steps": cutoff,
        "current_plan_sha256": base["sha256"],
        "catalogue_version": version,
        "total_added_requests": len(events),
        "expert_teacher_qualified": False,
    }


def append_demand(journal, record, batch, *, settings=None):
    verify_demand_journal(journal, native=True)
    if len(journal["epochs"]) >= MAX_DEMAND_EPOCHS:
        raise ValueError("demand epoch budget exceeded")
    origin, base, cutoff = _base(journal["base_reference"], False)
    _, task = _bound_source(origin)
    events = []
    if journal["epochs"]:
        latest = journal["epochs"][-1]["result"]
        base = latest["plan"]
        task = latest["source_task"]
        cutoff = latest["committed_prefix_steps"]
        events = base["goals"]["demand"]["events"]
    cursor = len(record["episode"]["transitions"])
    if not cutoff <= cursor < len(base["steps"]):
        raise ValueError("demand cannot rewind committed history")
    edits = []
    for i, row in enumerate(record["episode"]["transitions"][cutoff:], cutoff):
        req = row["request"]
        if set(req["choices"]) != {"@central"}:
            raise ValueError("demand needs compiled joint actor prefix")
        edits.append(
            {
                "at_sec": base["steps"][i]["at_sec"],
                "advance_sec": req["advance_sec"],
                "action_ids": req["choices"]["@central"],
            }
        )
    fork = _fork(base, edits, cutoff)
    core, _cumulative = _compile(
        origin,
        fork,
        record,
        task,
        events,
        batch,
        settings,
        history="@incident" in fork["task"]["roles"],
    )
    version = _version(core["plan"], len(journal["epochs"]) + 1)
    previous = (
        journal["epochs"][-1]["sha256"]
        if journal["epochs"]
        else journal["base_reference"]["sha256"]
    )
    node = _seal(
        {
            "index": len(journal["epochs"]) + 1,
            "parent_sha256": previous,
            "parent_plan_sha256": base["sha256"],
            "parent_catalogue_version": journal["catalogue_version"],
            "cursor": cursor,
            "actor_edits": edits,
            "source_episode_sha256": record["episode"]["sha256"],
            "source_physical_sha256": digest(
                record["episode"]["runtime"]["backend_report"]
            ),
            "batch": DemandBatch.model_validate(batch).model_dump(mode="json"),
            "settings": core["schedule"]["settings"],
            "result": core,
            "catalogue_version": version,
        }
    )
    return _seal(
        {
            "schema": "space-env.demand-journal/1",
            "origin": journal["origin"],
            "base_reference": journal["base_reference"],
            "epochs": [*journal["epochs"], node],
            "catalogue_version": version,
        }
    )
