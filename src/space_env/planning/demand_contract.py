"""Declared request additions and goal-only effective graph; never preload data."""

import copy

from pydantic import Field, model_validator

from space_env.reference.lifecycle import CustomerRequest, TransformationJob
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest


class DemandPayload(StrictModel):
    request: CustomerRequest
    jobs: tuple[TransformationJob, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def unique_jobs(self):
        if len({j.id for j in self.jobs}) != len(self.jobs):
            raise ValueError("duplicate demand job")
        return self


class DemandEvent(StrictModel):
    at_sec: float = Field(ge=0, le=21600, strict=True)
    payload: DemandPayload


class DemandGoals(StrictModel):
    events: tuple[DemandEvent, ...] = Field(min_length=1, max_length=64)
    min_fulfilled: int = Field(ge=0, le=64, strict=True)

    @model_validator(mode="after")
    def unique_requests(self):
        if len({e.payload.request.id for e in self.events}) != len(
            self.events
        ) or self.min_fulfilled > len(self.events):
            raise ValueError(
                "duplicate demand or new-service floor exceeds denominator"
            )
        return self


def effective_scenario(base, events):
    result = copy.deepcopy(base)
    life = result["definition"]["lifecycle"]
    request_ids = {r["id"] for r in life["requests"]}
    job_ids = {j["id"] for j in life["jobs"]}
    for raw in events:
        event = DemandEvent.model_validate(raw)
        r = event.payload.request
        if r.id in request_ids or r.release_sec < event.at_sec:
            raise ValueError("duplicate/backdated declared demand")
        request_ids.add(r.id)
        life["requests"].append(r.model_dump(mode="json"))
        for job in event.payload.jobs:
            if job.id in job_ids:
                raise ValueError("duplicate declared demand job")
            job_ids.add(job.id)
            life["jobs"].append(job.model_dump(mode="json"))
    return result


def validate_demand_schedule(scenario, task, steps, demand):
    demand = DemandGoals.model_validate(demand)
    if scenario["backend"] != "reference-spacecraft":
        raise ValueError("demand goals require spacecraft lifecycle")
    effective_scenario(scenario, demand.events)
    events = []
    actions = {a.id: a for a in task.actions}
    for step in steps:
        for aid in step.action_ids:
            action = actions.get(aid)
            if (
                action is not None
                and action.kind == "inject"
                and action.event["type"] == "request_added"
            ):
                events.append(
                    DemandEvent(
                        at_sec=step.at_sec, payload=action.event["payload"]
                    ).model_dump(mode="json")
                )
    if canonical_json(events) != canonical_json(
        [e.model_dump(mode="json") for e in demand.events]
    ):
        raise ValueError(
            "demand goal manifest differs from scheduled operator additions"
        )


def demand_evidence(scenario, report, demand):
    demand = DemandGoals.model_validate(demand)
    effective = effective_scenario(scenario, demand.events)
    expected = [e.model_dump(mode="json") for e in demand.events]
    actual = []
    try:
        for event in report["lifecycle"]["events"]:
            if event["type"] == "request_added":
                actual.append(
                    DemandEvent(
                        at_sec=event["time_sec"], payload=event["payload"]
                    ).model_dump(mode="json")
                )
    except (ValueError, KeyError, TypeError):
        actual = None
    request_ids = {e.payload.request.id for e in demand.events}
    fulfilled = sum(
        report["lifecycle"]["requests"].get(r, {}).get("status") == "completed"
        for r in request_ids
    )
    checks = {
        "committed_event_trace": actual is not None
        and canonical_json(actual) == canonical_json(expected),
        "minimum_added_service": fulfilled >= demand.min_fulfilled,
        "complete_demand_denominator": set(report["lifecycle"]["requests"])
        == {r["id"] for r in effective["definition"]["lifecycle"]["requests"]},
    }
    return effective, {
        "passed": all(checks.values()),
        "checks": checks,
        "added_requests": len(request_ids),
        "fulfilled_added": fulfilled,
        "effective_intent_sha256": digest(effective),
        "scope": "goal-only graph reconstructed from declared and native committed additions; not simulator preload",
    }
