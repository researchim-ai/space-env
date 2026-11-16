"""Author-declared causal stimuli, independent of an agent's command catalogue.

This is a simulation driver, not a hardware fault distribution or autonomous
recovery policy. Same-time events execute in declaration order, individually.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from space_env.reference.lifecycle import CustomerRequest, TransformationJob
from space_env.runtime import RuntimeContractError
from space_env.runtime.schema import FiniteScalar, StrictModel


class ActorPayload(StrictModel):
    actor: str = Field(min_length=1, max_length=128)


class FaultPayload(ActorPayload):
    subsystem: Literal["bus", "adcs", "payload", "cpu", "radio", "propulsion"]


class StationPayload(StrictModel):
    station: str = Field(min_length=1, max_length=128)


class OutagePayload(StationPayload):
    start_sec: FiniteScalar = Field(ge=0)
    end_sec: FiniteScalar = Field(gt=0)

    @model_validator(mode="after")
    def interval(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("outage end must follow start")
        return self


class RequestPayload(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)


class PriorityPayload(RequestPayload):
    priority: int = Field(ge=0, strict=True)


class DeadlinePayload(RequestPayload):
    deadline_sec: FiniteScalar = Field(gt=0)


class AddedDemandPayload(StrictModel):
    # A raw observation may be directly deliverable, with no transformation job.
    request: CustomerRequest
    jobs: tuple[TransformationJob, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def unique_jobs(self):
        if len({job.id for job in self.jobs}) != len(self.jobs):
            raise ValueError("duplicate demand job")
        return self


class TaskEnvironmentEvent(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    at_sec: FiniteScalar = Field(ge=0, le=259200)
    type: Literal[
        "request_added",
        "request_cancelled",
        "request_priority_changed",
        "request_deadline_changed",
        "spacecraft_fault",
        "spacecraft_fault_cleared",
        "spacecraft_safe_mode",
        "spacecraft_recover",
        "station_outage_added",
        "station_recovered",
    ]
    payload: dict

    @model_validator(mode="after")
    def payload_contract(self):
        model = {
            "request_added": AddedDemandPayload,
            "request_cancelled": RequestPayload,
            "request_priority_changed": PriorityPayload,
            "request_deadline_changed": DeadlinePayload,
            "spacecraft_fault": FaultPayload,
            "spacecraft_fault_cleared": FaultPayload,
            "spacecraft_safe_mode": ActorPayload,
            "spacecraft_recover": ActorPayload,
            "station_outage_added": OutagePayload,
            "station_recovered": StationPayload,
        }[self.type]
        value = model.model_validate(self.payload)
        if self.type == "request_added" and value.request.release_sec < self.at_sec:
            raise ValueError("scheduled demand cannot be backdated")
        if self.type == "station_outage_added" and value.start_sec < self.at_sec:
            raise ValueError("scheduled outage cannot be backdated")
        if (
            self.type == "request_deadline_changed"
            and value.deadline_sec <= self.at_sec
        ):
            raise ValueError("new deadline must follow event time")
        object.__setattr__(self, "payload", value.model_dump(mode="json"))
        return self


def validate_environment(events, runtime):
    """Validate static scope before execution; state-dependent rejection is logged."""
    if not events:
        return set()
    capabilities = runtime.describe_capabilities()
    if "events.inject" not in capabilities["supported"]:
        raise ValueError("environment events require native injection capability")
    actors = set(capabilities["actors"])
    backend = runtime._backend
    mission = getattr(backend, "mission", None)
    stations = set(mission.stations) if mission is not None else set()
    declared_requests = set()
    validation = None
    for event in events:
        if event.at_sec >= runtime.horizon_sec:
            raise ValueError("environment event must precede native horizon")
        if (
            event.type == "station_outage_added"
            and event.payload["end_sec"] > runtime.horizon_sec
        ):
            raise ValueError("scheduled outage exceeds native horizon")
        if "actor" in event.payload and event.payload["actor"] not in actors:
            raise ValueError("environment event references unknown actor")
        if "station" in event.payload and event.payload["station"] not in stations:
            raise ValueError("environment event references unknown station")
        if (
            event.type.startswith("spacecraft_")
            and getattr(backend, "bus", None) is None
        ):
            raise ValueError("spacecraft stimulus requires coupled bus")
        if (
            event.type.startswith("request_")
            and getattr(backend, "lifecycle", None) is None
        ):
            raise ValueError("demand stimulus requires native lifecycle")
        if event.type == "request_added":
            request = event.payload["request"]
            if (
                set(request["eligible_actors"]) - actors
                or request["destination_id"] not in stations
            ):
                raise ValueError("new demand references unknown actor/station")
            if validation is None:
                validation = runtime.fork()
            # Validate the complete future demand graph on a disposable runtime.
            # No future inventory or request is installed in the live mission.
            validation.inject(
                [
                    {
                        "id": f"@task-validation/{event.id}",
                        "type": event.type,
                        "payload": event.payload,
                        "time_sec": validation.time_sec,
                        "source": "task-environment-validation/1",
                        "reason": "validate declared demand graph",
                    }
                ],
                expected_revision=validation.revision,
                idempotency_key=f"@task-validation/{event.id}",
            )
            declared_requests.add(request["id"])
    return declared_requests


def apply_environment(runtime, events, start_index):
    """Apply due stimuli at now, returning a new cursor and public outcomes.

    Native inject is atomic per event. Invalid *state-dependent* events are attempted
    once and reported as rejected without leaking backend exception text.
    """
    index, outcomes = start_index, []
    while index < len(events) and events[index].at_sec <= runtime.time_sec + 1e-9:
        event = events[index]
        native_id = f"@task-environment/{index}/{event.id}"
        accepted = True
        try:
            runtime.inject(
                [
                    {
                        "id": native_id,
                        "type": event.type,
                        "payload": event.payload,
                        "time_sec": runtime.time_sec,
                        "source": "task-environment/1",
                        "reason": "author-declared external stimulus",
                    }
                ],
                expected_revision=runtime.revision,
                idempotency_key=native_id,
            )
        except RuntimeContractError:
            accepted = False
        outcomes.append(
            {
                "id": event.id,
                "type": event.type,
                "time_sec": runtime.time_sec,
                "accepted": accepted,
                "code": "applied" if accepted else "native_rejected",
            }
        )
        index += 1
    return index, outcomes
