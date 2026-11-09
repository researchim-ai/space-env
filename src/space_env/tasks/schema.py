"""Immutable task intent and bounded action catalogues, independent of physics."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from space_env.navigation.models import SensorSpec
from space_env.runtime.schema import FiniteScalar, RuntimeCommand, StrictModel
from space_env.tasks.goals import MissionGoal
from space_env.tasks.modeling import ModelRequirement


class TaskRole(StrictModel):
    action_slots: int = Field(default=1, ge=1, le=8, strict=True)
    actors: tuple[str, ...] = Field(default=(), max_length=64)
    stations: tuple[str, ...] = Field(default=(), max_length=128)
    demand_actors: tuple[str, ...] = Field(default=(), max_length=64)
    demand_stations: tuple[str, ...] = Field(default=(), max_length=128)

    @model_serializer(mode="wrap")
    def legacy_demand_authority(self, handler):
        value = handler(self)
        if not self.demand_actors:
            value.pop("demand_actors", None)
        if not self.demand_stations:
            value.pop("demand_stations", None)
        return value


class TaskAction(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1)
    kind: Literal["submit", "cancel_actor", "inject"]
    commands: tuple[RuntimeCommand, ...] = Field(default=(), max_length=64)
    actor: str = ""
    event: dict | None = None
    request_ref: str | None = Field(default=None, min_length=1, max_length=128)

    @model_serializer(mode="wrap")
    def legacy_request_ref(self, handler):
        value = handler(self)
        if self.request_ref is None:
            value.pop("request_ref", None)
        return value

    @model_validator(mode="after")
    def payload(self):
        if self.request_ref is not None and self.kind != "submit":
            raise ValueError("request_ref is only for submission intent")
        if any(c.forecast is not None for c in self.commands):
            raise ValueError(
                "task templates cannot contain privileged forecast bindings"
            )
        if self.id == "wait":
            raise ValueError("wait is the reserved no-op action")
        if (self.kind == "submit") != bool(self.commands):
            raise ValueError(
                "submit requires commands; other actions cannot contain them"
            )
        if (self.kind == "cancel_actor") != bool(self.actor):
            raise ValueError("cancel_actor requires actor")
        if (self.kind == "inject") != (self.event is not None):
            raise ValueError("inject requires a declared event")
        return self


class TaskSpec(StrictModel):
    schema_id: Literal["space-env.task/1"] = Field(
        default="space-env.task/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    roles: dict[str, TaskRole] = Field(min_length=1, max_length=65)
    actions: tuple[TaskAction, ...] = Field(default=(), max_length=1023)
    objective: Literal["delivery", "throughput", "sla", "efficient", "robust"] = (
        "delivery"
    )
    decision_sec: float = Field(default=0.5, gt=1e-9, le=3600, strict=True)
    durations_sec: tuple[FiniteScalar, ...] = Field(default=(), max_length=32)
    max_decision_sec: float = Field(default=30.0, gt=1e-9, le=3600, strict=True)
    discount_per_sec: float = Field(default=0.99, gt=0, le=1, strict=True)
    max_decisions: int = Field(default=256, ge=1, le=10000, strict=True)
    max_entities: int = Field(default=256, ge=1, le=10000, strict=True)
    max_relations: int = Field(default=1024, ge=1, le=100000, strict=True)
    max_actions: int = Field(default=128, ge=1, le=1024, strict=True)
    max_joint_actions: int = Field(default=8, ge=1, le=64, strict=True)
    seed_policy: Literal["preserve", "derive-network"] = "preserve"
    policy_origin: Literal["scripted", "random", "external"] = "external"
    runtime_audit: Literal["full", "compact"] = "full"
    pointing_masks: bool = Field(default=False, strict=True)
    goal: MissionGoal | None = None
    navigation_sensors: dict[str, SensorSpec] = Field(
        default_factory=dict, max_length=8
    )
    model_requirements: tuple[ModelRequirement, ...] = Field(default=(), max_length=64)

    @model_serializer(mode="wrap")
    def serialize_optional_pointing(self, handler):
        value = handler(self)
        if not self.pointing_masks:
            value.pop("pointing_masks", None)
        if self.goal is None:
            value.pop("goal", None)
        if not self.navigation_sensors:
            value.pop("navigation_sensors", None)
        if not self.model_requirements:
            value.pop("model_requirements", None)
        return value

    @model_validator(mode="after")
    def contract(self):
        pairs = [(r.feature, a) for r in self.model_requirements for a in r.actors]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate actor modeling requirement")
        controlled = {a for role in self.roles.values() for a in role.actors}
        if any(actor not in controlled for _, actor in pairs):
            raise ValueError("model requirements must name controlled task actors")
        if any(not key.strip() or len(key) > 128 for key in self.roles):
            raise ValueError("invalid role ID")
        if "@central" in self.roles or len({a.id for a in self.actions}) != len(
            self.actions
        ):
            raise ValueError("reserved role or duplicate action ID")
        if len(self.actions) + 1 > self.max_actions:
            raise ValueError("action catalogue capacity exceeded")
        if sum(r.action_slots for r in self.roles.values()) > self.max_joint_actions:
            raise ValueError("joint action capacity must cover declared role slots")
        if len(set(self.durations_sec)) != len(self.durations_sec) or any(
            not 1e-9 < t <= self.max_decision_sec for t in self.durations_sec
        ):
            raise ValueError("invalid duration catalogue")
        if self.decision_sec > self.max_decision_sec:
            raise ValueError("decision_sec exceeds maximum")
        for key in ["actors", "stations"]:
            names = [
                name for role in self.roles.values() for name in getattr(role, key)
            ]
            if len(names) != len(set(names)):
                raise ValueError("roles must have disjoint " + key)
        if self.navigation_sensors:
            if set(self.navigation_sensors) != controlled:
                raise ValueError(
                    "navigation sensors must cover exactly all controlled actors"
                )
            if self.pointing_masks or self.max_decisions > 512:
                raise ValueError(
                    "sensor task forbids pointing masks and exceeds 512-decision bound"
                )
            if self.goal is not None and any(
                c.kind != "product_delivery" for c in self.goal.conditions
            ):
                raise ValueError(
                    "sensor task does not yet support native goal-error projection"
                )
            if any(
                a.kind == "inject"
                or any(
                    c.kind
                    not in {"wait", "downlink", "processing", "impulse", "finite_burn"}
                    for c in a.commands
                )
                for a in self.actions
            ):
                raise ValueError(
                    "sensor task supports inertial impulse/opt-in finite burn, coast, downlink, processing and cancellation only"
                )
        if self.navigation_sensors and any(
            c.kind == "finite_burn"
            and self.navigation_sensors[c.actor_id].control_profile
            != "inertial_finite_burn"
            for a in self.actions
            for c in a.commands
            if c.actor_id in self.navigation_sensors
        ):
            raise ValueError(
                "finite burn requires explicit inertial_finite_burn estimator profile"
            )
        for action in self.actions:
            if action.role not in self.roles:
                raise ValueError("unknown action role")
            role = self.roles[action.role]
            if any(c.actor_id not in role.actors for c in action.commands) or (
                action.actor and action.actor not in role.actors
            ):
                raise ValueError("action cannot control another role actor")
            if action.event is not None:
                event = action.event
                if set(event) != {"type", "payload"} or not isinstance(
                    event["payload"], dict
                ):
                    raise ValueError("task event requires type/payload only")
                payload = event["payload"]
                kind = event["type"]
                if kind == "station_outage_added" and (
                    set(payload) != {"station", "start_sec", "end_sec"}
                    or any(
                        type(payload.get(k)) not in (int, float)
                        or not math.isfinite(payload[k])
                        for k in ("start_sec", "end_sec")
                    )
                    or not 0 <= payload["start_sec"] < payload["end_sec"]
                ):
                    raise ValueError(
                        "unsupported or invalid station outage declaration"
                    )
                if kind == "request_added":
                    from space_env.planning.demand_contract import DemandPayload

                    try:
                        declaration = DemandPayload.model_validate(payload)
                    except ValueError as exc:
                        raise ValueError(
                            "unsupported or invalid demand declaration"
                        ) from exc
                    request = declaration.request
                    allowed = (
                        set(request.eligible_actors) <= set(role.demand_actors)
                        and request.destination_id in role.demand_stations
                    )
                elif kind in {
                    "spacecraft_recover",
                    "spacecraft_fault_cleared",
                    "spacecraft_fault",
                    "spacecraft_safe_mode",
                }:
                    allowed = payload.get("actor") in role.actors
                elif kind in {
                    "request_cancelled",
                    "request_priority_changed",
                    "request_deadline_changed",
                }:
                    # Mission demand authority is explicit and distinct from
                    # ordinary spacecraft command authority. Native lifecycle
                    # still validates ownership, terminal state and payload.
                    allowed = bool(role.demand_actors and role.demand_stations)
                elif kind in {
                    "station_recovered",
                    "station_outage_added",
                    "station_reservation_policy_changed",
                    "terminal_booking_added",
                }:
                    allowed = payload.get("station") in role.stations
                else:
                    allowed = False
                if not allowed:
                    raise ValueError(
                        "task operator event is unsupported or outside role authority"
                    )
        return self


class TaskDecision(StrictModel):
    choices: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=65)
    expected_revision: int = Field(ge=0, strict=True)
    idempotency_key: str = Field(min_length=1, max_length=256)
    advance_sec: float | None = Field(default=None, gt=1e-9, le=3600, strict=True)
