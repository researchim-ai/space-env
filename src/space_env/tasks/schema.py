"""Immutable task intent and bounded action catalogues, independent of physics."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar, RuntimeCommand, StrictModel


class TaskRole(StrictModel):
    action_slots: int = Field(default=1, ge=1, le=8, strict=True)
    actors: tuple[str, ...] = Field(default=(), max_length=64)
    stations: tuple[str, ...] = Field(default=(), max_length=128)


class TaskAction(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1)
    kind: Literal["submit", "cancel_actor", "inject"]
    commands: tuple[RuntimeCommand, ...] = Field(default=(), max_length=64)
    actor: str = ""
    event: dict | None = None

    @model_validator(mode="after")
    def payload(self):
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

    @model_validator(mode="after")
    def contract(self):
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
                if kind in {"spacecraft_recover", "spacecraft_fault_cleared"}:
                    allowed = payload.get("actor") in role.actors
                elif kind in {
                    "station_recovered",
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
