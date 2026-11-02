"""Bounded policy configuration and immutable public planning input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest


class PolicyConfig(StrictModel):
    name: Literal[
        "wait", "random_masked", "priority", "earliest_deadline", "route_aware_rolling"
    ] = "route_aware_rolling"
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)
    max_nodes: int = Field(default=1024, ge=1, le=100000, strict=True)
    max_candidates: int = Field(default=32, ge=1, le=64, strict=True)
    horizon_sec: float = Field(default=60.0, gt=0, le=86400, strict=True)
    decision_sec: float | None = Field(default=None, gt=1e-9, le=3600, strict=True)
    cooldown_steps: int = Field(default=2, ge=1, le=100, strict=True)
    stall_steps: int = Field(default=2, ge=1, le=100, strict=True)


@dataclass(frozen=True)
class PlanningInput:
    """Serialized by value: a policy cannot retain a mutable trainer reference."""

    payload: str

    def __post_init__(self):
        value = json.loads(self.payload)
        if (
            set(value)
            != {
                "schema",
                "observation",
                "roles",
                "actions",
                "active",
                "stations",
                "limits",
                "feedback",
            }
            or value["schema"] != "space-env.planning-input/1"
        ):
            raise ValueError("invalid public planning input")
        object.__setattr__(self, "payload", canonical_json(value))

    def view(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return digest(self.view())


def make_input(session, feedback=None):
    """Trusted bridge; only current public views/static claims cross the boundary.

    No backend validation, forecast, scenario, seed, truth report or future rollout.
    The policy receives the returned value, never the session itself.
    """
    observation = session.observe()
    catalogue = {a["id"]: a for a in observation["catalogue"]}
    actions = {}
    for a in session._actions:
        claims = []
        for c in a.commands if catalogue[a.id]["enabled"] else ():
            claims.extend(
                [c.actor_id, r] for r in session._runtime.command_claims(c)["resources"]
            )
        if a.actor:
            claims.append([a.actor, "actor"])
        actions[a.id] = {"role": a.role, "claims": sorted(claims)}
        actions[a.id]["command_claims"] = [
            [[c.actor_id, r] for r in session._runtime.command_claims(c)["resources"]]
            for c in a.commands
            if catalogue[a.id]["enabled"]
        ]
        # Parameters are taken only from the already causal B06 catalogue.
        assert "commands" not in actions[a.id]
    active = []
    for row in session._runtime.command_status():
        if row["status"] not in {"accepted", "running"}:
            continue
        c = row["command"]
        active.append(
            {
                "id": c["command_id"],
                "actor": c["actor_id"],
                "kind": c["kind"],
                "destination": c["destination_id"],
                "until_sec": row["until_sec"],
                "claims": [
                    [c["actor_id"], r]
                    for r in session._runtime.command_claims(c)["resources"]
                ],
            }
        )
    stations = {
        e["id"]: session._runtime.station_observation(e["id"])
        for e in observation["entities"]
        if e["kind"] == "station"
    }
    spec = session._task
    value = {
        "schema": "space-env.planning-input/1",
        "observation": observation,
        "roles": {k: v.model_dump(mode="json") for k, v in spec.roles.items()},
        "actions": {k: v for k, v in actions.items() if k in catalogue},
        "active": sorted(active, key=lambda r: r["id"]),
        "stations": stations,
        "limits": {
            "joint": spec.max_joint_actions,
            "decision_sec": spec.decision_sec,
            "max_decision_sec": spec.max_decision_sec,
        },
        "feedback": feedback,
    }
    return PlanningInput(canonical_json(value))
