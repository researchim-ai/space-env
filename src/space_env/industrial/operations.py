"""Causal event driver over native EO/CPU/ISL execution, with bounded maintenance.

The agent cannot inject events. Diagnostic/reset are explicitly ideal ground
maintenance services, not a calibrated onboard FDIR or redundant hardware model.
Future stimuli remain in privileged setup/replay only.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import Field, model_validator

from space_env.api.workspace_dynamic import MissionEvent
from space_env.industrial.session import DecisionSession
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession
from space_env.tasks.heterogeneous_missions import (
    HeterogeneousMissionSpec,
    heterogeneous_mission,
)
from space_env.tasks.observation import STATES
from space_env.tasks.session import score


class ScheduledEvent(StrictModel):
    at_sec: float = Field(ge=0, lt=60, strict=True)
    event: MissionEvent


class OperationsSpec(StrictModel):
    mission: HeterogeneousMissionSpec = Field(default_factory=HeterogeneousMissionSpec)
    events: tuple[ScheduledEvent, ...] = Field(default=(), max_length=32)
    deferred_request_at_sec: float | None = Field(
        default=None, ge=2, le=50, strict=True
    )
    permanent_faults: tuple[tuple[str, Literal["radio", "cpu"]], ...] = ()
    diagnostic_sec: float = Field(default=2.0, ge=2, le=10, strict=True)
    reset_sec: float = Field(default=6.0, ge=2, le=20, strict=True)
    max_reset_attempts: int = Field(default=1, ge=1, le=3, strict=True)

    @model_validator(mode="after")
    def grid(self):
        values = [self.diagnostic_sec, self.reset_sec, *(e.at_sec for e in self.events)]
        if self.deferred_request_at_sec is not None:
            values.append(self.deferred_request_at_sec)
        if any(v / 2 != round(v / 2) for v in values):
            raise ValueError("operations stimuli and maintenance use a two-second grid")
        if any(not 0 <= e.at_sec < 60 for e in self.events):
            raise ValueError("stimulus outside mission")
        return self


class OperationsAction(StrictModel):
    kind: Literal["execute", "diagnose", "reset", "recover", "isolate"] = "execute"
    choices: dict[str, tuple[str, ...]] = Field(default_factory=dict, max_length=8)
    actor: str | None = None
    subsystem: Literal["radio", "cpu"] | None = None

    @model_validator(mode="after")
    def payload(self):
        if self.kind == "execute":
            if self.actor is not None or self.subsystem is not None:
                raise ValueError("execute only accepts actor choices")
        elif (
            self.choices
            or not self.actor
            or (self.kind == "reset") != (self.subsystem is not None)
        ):
            raise ValueError("maintenance needs actor; only reset accepts subsystem")
        return self


class OperationsSession(DecisionSession):
    kind = "operations"

    def __init__(self, configuration=None):
        self.spec = OperationsSpec.model_validate(
            {} if configuration is None else copy.deepcopy(configuration)
        )
        self._start(self.spec.model_dump(mode="json"))
        case = heterogeneous_mission(self.spec.mission)
        scenario, task = case["scenario"], case["task"]
        self._actors = list(scenario["actor_order"])
        self._public_actions = {a["id"]: copy.deepcopy(a) for a in task["actions"]}
        self._actor_roles = {
            a: r for r, role in task["roles"].items() for a in role["actors"]
        }
        if any(a not in self._actors for a, _ in self.spec.permanent_faults):
            raise ValueError("unknown permanent-fault actor")
        stations = [s["name"] for s in scenario["definition"]["operations"]["stations"]]
        task["roles"]["environment"] = {
            "action_slots": 8,
            "actors": [],
            "stations": stations,
            "demand_actors": self._actors,
            "demand_stations": stations,
        }
        self._stimuli = [
            (e.at_sec, e.event.model_dump(mode="json")) for e in self.spec.events
        ]
        if self.spec.deferred_request_at_sec is not None:
            life = scenario["definition"]["lifecycle"]
            request = life["requests"].pop()
            request["release_sec"] = self.spec.deferred_request_at_sec
            jobs = [j for j in life["jobs"] if request["raw_product_id"] in j["inputs"]]
            life["jobs"] = [j for j in life["jobs"] if j not in jobs]
            self._stimuli.append(
                (
                    self.spec.deferred_request_at_sec,
                    {
                        "type": "request_added",
                        "payload": {"request": request, "jobs": jobs},
                    },
                )
            )
        self._stimuli.sort(key=lambda e: (e[0], digest(e[1])))
        self._event_actions = []
        self._maintenance = {}

        def add(event):
            name = "@driver/" + digest(event)[:20]
            if any(a["id"] == name for a in task["actions"]):
                return name
            actor = event["payload"].get("actor")
            role = self._actor_roles.get(actor, "environment")
            task["actions"].append(
                {"id": name, "role": role, "kind": "inject", "event": event}
            )
            task["roles"][role]["action_slots"] = 8
            return name

        for _, event in self._stimuli:
            self._event_actions.append(add(event))
        for actor in self._actors:
            for subsystem in ("radio", "cpu"):
                event = {
                    "type": "spacecraft_fault_cleared",
                    "payload": {"actor": actor, "subsystem": subsystem},
                }
                self._maintenance[(actor, subsystem)] = add(event)
            self._maintenance[(actor, "recover")] = add(
                {"type": "spacecraft_recover", "payload": {"actor": actor}}
            )
            self._maintenance[(actor, "isolate")] = add(
                {"type": "spacecraft_safe_mode", "payload": {"actor": actor}}
            )
        task.update(
            max_actions=len(task["actions"]) + 1,
            max_joint_actions=sum(r["action_slots"] for r in task["roles"].values()),
            max_decisions=256,
        )
        self._native = TaskSession(scenario, task)
        self._processed = set()
        self._environment_log, self._native_log = [], []
        self._diagnoses = {}
        self._maintenance_attempts = 0
        self._reset_history = {}
        self._drain()

    def _native_step(self, ids, duration):
        if self._native._done:
            return None
        result = self._native.step(
            {"@central": ids or ["wait"]},
            expected_revision=self._native.revision,
            idempotency_key="driver/" + str(self._native.revision),
            advance_sec=float(duration),
        )
        self._native_log.append(
            {
                "reward": result["reward"],
                "feedback": result["feedback"],
                "elapsed_sec": result["elapsed_sec"],
            }
        )
        return result

    def _drain(self):
        # Each stimulus is a separate native transaction. A rejected policy
        # command can never roll back or suppress an environmental event.
        while not self._native._done:
            now = self._native._runtime.time_sec
            due = [
                i
                for i, (at, _) in enumerate(self._stimuli)
                if at <= now + 1e-9 and i not in self._processed
            ]
            if not due:
                break
            index = due[0]
            result = self._native_step([self._event_actions[index]], 2.0)
            self._processed.add(index)
            self._environment_log.append(
                {
                    "index": index,
                    "at_sec": now,
                    "accepted": result["feedback"]["accepted"],
                }
            )
        self._done = self._native._done

    def _advance(self, seconds, ids=()):
        end = min(
            self._native._runtime.horizon_sec, self._native._runtime.time_sec + seconds
        )
        first = True
        while not self._native._done and self._native._runtime.time_sec < end - 1e-9:
            now = self._native._runtime.time_sec
            future = [
                at
                for i, (at, _) in enumerate(self._stimuli)
                if i not in self._processed and at > now + 1e-9
            ]
            stop = min(end, min(future, default=end), now + 2)
            self._native_step(list(ids) if first else [], stop - now)
            first = False
            self._drain()
        self._done = self._native._done

    def observe(self):
        roles = {}
        views = {
            role: self._native.observe(role)
            for role in sorted(set(self._actor_roles.values()))
        }
        # One central operations agent shares the union of current actor views.
        # Gateways need declared request IDs to expose their delivery templates;
        # they need not themselves be eligible imaging actors.
        visible_requests = {
            e["id"]
            for value in views.values()
            for e in value["entities"]
            if e["kind"] == "request"
        }
        for role, value in views.items():
            keep = [
                i
                for i, row in enumerate(value["catalogue"])
                if row["id"] == "wait"
                or (
                    row["id"] in self._public_actions
                    and self._public_actions[row["id"]]["role"] == role
                    and (
                        not self._public_actions[row["id"]].get("request_ref")
                        or self._public_actions[row["id"]]["request_ref"]
                        in visible_requests
                    )
                )
            ]
            value["catalogue"] = [value["catalogue"][i] for i in keep]
            value["action_mask"] = [value["action_mask"][i] for i in keep]
            # Do not leak raw hashes of the full hidden catalogue.
            value.pop("binding", None)
            value["native_revision"] = value["revision"]
            value["revision"] = self.revision
            value["schema"] = "space-env.operations-actor-observation/1"
            value["binding"] = digest(value)
            roles[role] = value
        return {
            "schema": "space-env.operations-observation/1",
            "revision": self.revision,
            "time_sec": self._native._runtime.time_sec,
            "roles": roles,
            "diagnostics": copy.deepcopy(self._diagnoses),
            "maintenance_results": copy.deepcopy(self._reset_history),
            "maintenance": {
                "diagnostic_sec": self.spec.diagnostic_sec,
                "reset_wait_sec": self.spec.reset_sec,
                "commit_sec": 2.0,
                "max_reset_attempts": self.spec.max_reset_attempts,
                "resettable_subsystems": ["radio", "cpu"],
                "scope": "ideal ground maintenance; native resource time charged, no reset-specific hardware power model",
            },
        }

    def _apply(self, value):
        action = OperationsAction.model_validate(value)
        public = self.observe()
        if action.kind == "execute":
            if set(action.choices) - set(public["roles"]) or any(
                len(ids) > 1 or len(set(ids)) != len(ids)
                for ids in action.choices.values()
            ):
                raise ValueError("one action per known actor role required")
            for role, ids in action.choices.items():
                allowed = {r["id"] for r in public["roles"][role]["catalogue"]}
                if set(ids) - allowed:
                    raise ValueError("unknown, hidden or privileged action")
        elif action.actor not in self._actors:
            raise ValueError("unknown maintenance actor")
        if (
            action.kind == "reset"
            and self._reset_history.get(action.actor, {})
            .get(action.subsystem, {})
            .get("attempts", 0)
            >= self.spec.max_reset_attempts
        ):
            raise ValueError("declared reset attempt budget exhausted")
        before, start = len(self._native_log), self._native._runtime.time_sec
        feedback = {"accepted": True, "code": "executed"}
        if action.kind == "execute":
            self._advance(
                2.0, [i for ids in action.choices.values() for i in ids if i != "wait"]
            )
        elif action.kind == "diagnose":
            self._maintenance_attempts += 1
            self._advance(self.spec.diagnostic_sec)
            bus = self._native._runtime.observe(action.actor)["actors"][action.actor][
                "spacecraft_bus"
            ]
            self._diagnoses[action.actor] = {
                "at_sec": self._native._runtime.time_sec,
                "reported_faults": list(bus["faults"]),
                "safe": bool(bus["safe"]),
                "source": "ideal synthetic built-in test",
            }
        elif action.kind == "reset":
            self._maintenance_attempts += 1
            history = self._reset_history.setdefault(action.actor, {}).setdefault(
                action.subsystem, {"attempts": 0}
            )
            history["attempts"] += 1
            self._advance(self.spec.reset_sec)
            bus = self._native._runtime.observe(action.actor)["actors"][action.actor][
                "spacecraft_bus"
            ]
            if (
                (action.actor, action.subsystem) in self.spec.permanent_faults
                or action.subsystem not in bus["faults"]
                or self._native._done
            ):
                feedback = {"accepted": False, "code": "reset_ineffective"}
            else:
                result = self._native_step(
                    [self._maintenance[(action.actor, action.subsystem)]], 2.0
                )
                feedback = result["feedback"]
                self._drain()
            history.update(
                at_sec=self._native._runtime.time_sec, accepted=feedback["accepted"]
            )
        else:
            self._maintenance_attempts += 1
            result = self._native_step(
                [self._maintenance[(action.actor, action.kind)]], 2.0
            )
            feedback = result["feedback"]
            self._drain()
        elapsed = self._native._runtime.time_sec - start
        reward = math.fsum(row["reward"] for row in self._native_log[before:])
        if action.kind == "execute" and self._native_log[before:]:
            feedback = self._native_log[before]["feedback"]
        return {
            "reward": reward,
            "terminated": self._done,
            "elapsed_sec": elapsed,
            "feedback": feedback,
        }

    def metrics(self):
        counts = score(self._native._runtime)
        requests = self._native._runtime.observe(view="truth")["backend"]["report"][
            "lifecycle"
        ]["requests"]
        return {
            "success": self._done
            and bool(requests)
            and all(r["status"] == "completed" for r in requests.values()),
            "terminal": self._done,
            "fulfilled_requests": counts["completion"],
            "fulfilled_priority": counts["priority"],
            "total_requests": len(requests),
            "cancelled_requests": sum(
                r["status"] == "cancelled" for r in requests.values()
            ),
            "maintenance_attempts": self._maintenance_attempts,
            "environment_rejections": sum(
                not r["accepted"] for r in self._environment_log
            ),
            "native_return": math.fsum(r["reward"] for r in self._native_log),
            "flight_fdir_calibrated": False,
        }

    def _witness(self):
        return {
            "native": self._native.checkpoint(compact=True),
            "processed": sorted(self._processed),
            "environment": copy.deepcopy(self._environment_log),
            "diagnoses": copy.deepcopy(self._diagnoses),
            "reset_history": copy.deepcopy(self._reset_history),
            "metrics": self.metrics(),
        }


def operations_policy(observation):
    """Telemetry-only greedy baseline; diagnostic reports are not privileged setup."""
    for actor, report in sorted(observation["diagnostics"].items()):
        faults = [f for f in report["reported_faults"] if f in {"radio", "cpu"}]
        allowed = [
            f
            for f in faults
            if observation.get("maintenance_results", {})
            .get(actor, {})
            .get(f, {})
            .get("attempts", 0)
            < observation["maintenance"].get("max_reset_attempts", 1)
        ]
        if allowed and report["at_sec"] >= observation["time_sec"] - 2:
            return {"kind": "reset", "actor": actor, "subsystem": allowed[0]}
        if (
            not report["reported_faults"]
            and report["safe"]
            and report["at_sec"] >= observation["time_sec"] - 2
        ):
            return {"kind": "recover", "actor": actor}
    choices = {}
    claimed = set()
    for role, view in observation["roles"].items():
        exhausted = any(
            r["attempts"] >= observation["maintenance"].get("max_reset_attempts", 1)
            for r in observation.get("maintenance_results", {}).get(role, {}).values()
        )
        safe = any(
            e["kind"] == "actor" and e["values"].get("safe") for e in view["entities"]
        )
        symptom = any(r.get("reason") == "subsystem_fault" for r in view["catalogue"])
        if exhausted and (symptom or safe):
            if not safe:
                return {"kind": "isolate", "actor": role}
            last = observation["diagnostics"].get(role, {}).get("at_sec", -100.0)
            if observation["time_sec"] - last >= 12.0:
                return {"kind": "diagnose", "actor": role}
            continue
        # Command admission symptoms can suggest a test; no fault identity in
        # setup or future stimuli is passed to this baseline.
        if any(r.get("reason") == "subsystem_fault" for r in view["catalogue"]):
            return {"kind": "diagnose", "actor": role}
        requests = {
            e["id"]: e["values"] for e in view["entities"] if e["kind"] == "request"
        }
        inactive = {
            rid
            for rid, r in requests.items()
            if r.get("status_code")
            in {
                float(STATES[k])
                for k in ("completed", "cancelled", "expired", "failed")
            }
        }
        candidates = [
            r
            for r in view["catalogue"]
            if r["id"] != "wait"
            and r.get("enabled")
            and r.get("kind") != "cancel_actor"
            and r.get("request_ref") not in claimed | inactive
        ]
        order = {"downlink": 0, "relay": 1, "transform": 2, "imaging": 3, "slew": 4}
        if candidates:
            chosen = min(
                candidates,
                key=lambda r: (
                    min(
                        (order.get(k, 9) for k in r.get("command_kinds", [])), default=9
                    ),
                    -requests.get(r.get("request_ref"), {}).get("priority", 0),
                    r["id"],
                ),
            )
            choices[role] = [chosen["id"]]
            if chosen.get("request_ref"):
                claimed.add(chosen["request_ref"])
    return {"kind": "execute", "choices": choices}
