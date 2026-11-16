"""A deterministic, observation-only baseline for catalogue delivery campaigns.

This is a bounded reactive heuristic, not an optimal planner, learned policy or
autonomous repair controller. Its completion contract must be supplied publicly.
It never imports a scenario compiler, reads a session or predicts future events.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.tasks.observation import STATES


class CampaignPolicySpec(StrictModel):
    minimum_completed_requests: int = Field(ge=0, le=10000, strict=True)
    completion_action_ids: tuple[str, ...] = Field(default=(), max_length=32)
    poll_sec: FiniteScalar = Field(default=0.25, ge=0.01, le=30)
    adaptive_wait: bool = Field(default=False, strict=True)
    max_wait_sec: FiniteScalar = Field(default=1.0, ge=0.01, le=3600)
    request_pointing: bool = Field(default=False, strict=True)
    adaptive_routing: bool = Field(default=False, strict=True)
    require_route_progress: bool = Field(default=False, strict=True)
    max_route_age_sec: FiniteScalar = Field(default=3600, ge=0, le=3600)

    @model_validator(mode="after")
    def completion_ids(self):
        if self.require_route_progress and not self.adaptive_routing:
            raise ValueError("route progress requires adaptive routing")
        if self.adaptive_wait and self.max_wait_sec < self.poll_sec:
            raise ValueError("adaptive maximum wait must cover the polling interval")
        if len(set(self.completion_action_ids)) != len(
            self.completion_action_ids
        ) or any(
            not name.strip() or name == "wait" for name in self.completion_action_ids
        ):
            raise ValueError("completion actions must be unique nonblank catalogue IDs")
        return self


def campaign_policy(observation, specification):
    """Choose at most ONE currently enabled action, without private phase status.

    Ready delivery precedes processing and acquisition; within a stage choose
    higher request priority, earlier deadline, then a stable action ID. Single
    submissions avoid assuming that independently enabled masks imply a jointly
    feasible resource assignment. Polling lets other resource lanes be used on
    subsequent calls while native commands continue to run.
    """
    spec = CampaignPolicySpec.model_validate(specification)
    if observation.get("schema") != "space-env.agent-campaign-observation/1":
        raise ValueError("agent campaign observation required")
    native = observation["native"]
    requests = {
        row["id"]: row["values"]
        for row in native["entities"]
        if row["kind"] == "request"
    }
    completed = sum(
        r.get("status_code") == STATES["completed"] for r in requests.values()
    )
    entities = {row["index"]: row for row in native["entities"]}
    fulfills = {}
    for edge in native["relations"]:
        if edge["kind"] != "fulfills":
            continue
        source, target = entities[edge["source"]], entities[edge["destination"]]
        if source["kind"] == "product" and target["kind"] == "request":
            fulfills.setdefault(source["id"], set()).add(target["id"])
    completion_ids = set(spec.completion_action_ids)
    routes = {r["action_id"]: r for r in observation.get("relay_routes", [])}
    stages = {"downlink": 0, "relay": 1, "transform": 2, "processing": 2, "imaging": 3}
    if spec.request_pointing:
        stages["slew"] = 4
    terminal = {STATES[k] for k in ("completed", "cancelled", "expired", "failed")}

    def associated_requests(row):
        request_ids = {row["request_ref"]} if row.get("request_ref") else set()
        for command in row.get("commands", []):
            if command.get("request_id"):
                request_ids.add(command["request_id"])
            request_ids.update(fulfills.get(command.get("product_id"), ()))
        return [requests[r] for r in request_ids if r in requests]

    def rank(row):
        associated = [
            r for r in associated_requests(row) if r.get("status_code") not in terminal
        ]
        priority = max((r.get("priority", 0) for r in associated), default=0)
        deadline = min(
            (r.get("deadline_sec", float("inf")) for r in associated),
            default=float("inf"),
        )
        return (
            min(stages[k] for k in row["command_kinds"]),
            -priority,
            deadline,
            -routes.get(row["id"], {}).get("rate_bps", 0.0)
            if spec.adaptive_routing
            else 0.0,
            -routes.get(row["id"], {}).get("receiver_free_bits", 0.0)
            if spec.adaptive_routing
            else 0.0,
            row["id"],
        )

    ready = []
    for row in native["catalogue"]:
        kinds = row.get("command_kinds", [])
        if row["id"] in completion_ids or not row.get("enabled") or not kinds:
            continue
        if row.get("kind") != "submit" or any(k not in stages for k in kinds):
            continue
        if len(row.get("commands", [])) != 1:
            continue
        if (
            kinds == ["relay"]
            and spec.adaptive_routing
            and (
                not routes.get(row["id"], {}).get("available")
                or (
                    spec.require_route_progress
                    and not routes.get(row["id"], {}).get("makes_progress")
                )
                or routes.get(row["id"], {}).get("age_sec", 0)
                > spec.max_route_age_sec + 1e-9
            )
        ):
            continue
        if kinds == ["slew"] and (
            not row.get("request_ref")
            or not row["commands"][0].get("slew", {}).get("target_id")
            or not any(
                p["boresight_error_deg"] > p["payload_tolerance_deg"]
                for p in row.get("pointing", [])
            )
        ):
            continue
        associated = associated_requests(row)
        if associated and all(r.get("status_code") in terminal for r in associated):
            continue
        ready.append(row)
    selected = min(ready, key=rank)["id"] if ready else "wait"
    if selected == "wait" and completed >= spec.minimum_completed_requests:
        enabled = {
            r["id"]
            for r in native["catalogue"]
            if r.get("enabled")
            and r.get("kind") == "submit"
            and len(r.get("commands", [])) == 1
        }
        selected = next(
            (name for name in spec.completion_action_ids if name in enabled), "wait"
        )
    duration = min(spec.poll_sec, observation["control"]["max_decision_sec"])
    if selected == "wait" and spec.adaptive_wait:
        now = native["time_sec"]
        duration = min(spec.max_wait_sec, observation["control"]["max_decision_sec"])
        # Only already-visible command occupancy, demand deadlines and local
        # product readiness. These are bounds on waiting, not completion oracles.
        boundaries = [
            row["values"][key] - now
            for row in native["entities"]
            for key in (
                ("deadline_sec",)
                if row["kind"] in {"command", "request"}
                else ("ready_sec",)
                if row["kind"] == "product"
                else ()
            )
            if row["values"].get(key, now) - now > 1e-9
        ]
        if boundaries:
            duration = min(duration, min(boundaries))
        delivery = observation.get("route_telemetry", {}).get("next_delivery_sec")
        pending_routes = any(
            row.get("command_kinds") == ["relay"]
            and any(
                r.get("status_code") not in terminal for r in associated_requests(row)
            )
            for row in native["catalogue"]
        )
        if (
            spec.adaptive_routing
            and pending_routes
            and delivery is not None
            and delivery - now > 1e-9
        ):
            duration = min(duration, delivery - now)
        if spec.request_pointing and any(
            row.get("reason") in {"current_settling", "current_body_rate"}
            for row in native["catalogue"]
        ):
            duration = min(duration, spec.poll_sec)
    return {
        "choices": {"@central": [selected]},
        "advance_sec": duration,
    }
