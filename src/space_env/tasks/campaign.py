"""Sequential native goals over a single task runtime, with cold replay.

Central-controller API. Phase goals are trainer evaluation, not actor observations.
No reset, recipe concatenation, hidden forecasts or automatic command execution.
"""

from __future__ import annotations

import copy
from itertools import pairwise
from threading import RLock
from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks.goals import (
    MissionGoal,
    OrbitGoal,
    RelativeOrbitGoal,
    evaluate_goal,
)
from space_env.tasks.schema import TaskDecision, TaskSpec
from space_env.tasks.session import TaskSession


class MissionPhase(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    goal: MissionGoal


class CampaignSpec(StrictModel):
    schema_id: Literal["space-env.campaign/1"] = Field(
        default="space-env.campaign/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    phases: tuple[MissionPhase, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def ordered(self):
        if len({p.id for p in self.phases}) != len(self.phases):
            raise ValueError("phase IDs must be unique")
        deadlines = [p.goal.deadline_sec for p in self.phases]
        if any(b <= a for a, b in pairwise(deadlines)):
            raise ValueError(
                "phase deadlines must strictly increase in absolute mission time"
            )
        return self


class CampaignSession:
    """Bounded phase controller. Public status contains no physical goal residuals.

    Native task reward is retained separately; phase completions are counted once.
    A completed phase is historical achievement, not a continuously held invariant.
    Resource limits are absolute from scenario origin, active in their own phase.
    """

    def __init__(self, scenario, task, campaign, *, seed=None):
        self._lock = RLock()
        self._spec = CampaignSpec.model_validate(copy.deepcopy(campaign))
        base = TaskSpec.model_validate(copy.deepcopy(task))
        if base.goal is not None:
            raise ValueError("campaign task must omit goal; declare goals in phases")
        # Validate EVERY phase through the original task contract, including sensor
        # observability and reference-role visibility guards, before any execution.
        self._session = None
        for phase in self._spec.phases:
            candidate = base.model_dump(mode="json", by_alias=True)
            candidate["goal"] = phase.goal.model_dump(mode="json", by_alias=True)
            TaskSession(scenario, candidate, seed=seed)
        self._session = TaskSession(
            scenario, base.model_dump(mode="json", by_alias=True), seed=seed
        )
        self._initial = {
            "scenario": copy.deepcopy(scenario),
            "task": base.model_dump(mode="json", by_alias=True),
            "campaign": self._spec.model_dump(mode="json", by_alias=True),
            "seed": seed,
        }
        self._index = 0
        self._completed = []
        self._failures = ()
        self._achievements = ()
        self._stopped = False
        self._history = []
        self._cache = {}

    @property
    def revision(self):
        return self._session.revision

    def status(self):
        with self._lock:
            return {
                "schema": "space-env.campaign-status/1",
                "id": self._spec.id,
                "active_phase": None
                if self._index == len(self._spec.phases)
                else self._spec.phases[self._index].id,
                "completed_phases": list(self._completed),
                "success": self._index == len(self._spec.phases),
                "terminal": self._stopped,
                "failed": bool(self._failures),
                "revision": self.revision,
                "time_sec": self._session._runtime.time_sec,
            }

    def observe(self, role="@central"):
        # Deliberately no global success/goal residual projection into local roles.
        with self._lock:
            return self._session.observe(role)

    def step(self, choices, *, expected_revision, idempotency_key, advance_sec=None):
        with self._lock:
            request = TaskDecision(
                choices=choices,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                advance_sec=advance_sec,
            )
            body = request.model_dump(mode="json")
            key_body = {
                k: v
                for k, v in body.items()
                if k not in ("expected_revision", "idempotency_key")
            }
            sha = digest(key_body)
            if idempotency_key in self._cache:
                old_sha, response = self._cache[idempotency_key]
                if old_sha != sha:
                    raise ValueError("campaign idempotency conflict")
                return copy.deepcopy(response)
            if expected_revision != self.revision:
                raise ValueError("stale campaign revision")
            if self._stopped:
                raise ValueError("campaign is terminal")
            goal = self._spec.phases[self._index].goal
            now = self._session._runtime.time_sec
            dt = (
                self._session._task.decision_sec if advance_sec is None else advance_sec
            )
            if dt > self._session._task.max_decision_sec:
                raise ValueError("decision duration exceeds declared maximum")
            dt = min(dt, goal.deadline_sec - now)
            for condition in goal.conditions:
                if (
                    isinstance(condition, (OrbitGoal, RelativeOrbitGoal))
                    and condition.at_sec > now + 1e-9
                ):
                    dt = min(dt, condition.at_sec - now)
            if dt <= 1e-9:
                raise ValueError("phase deadline reached without transition")
            # One native atomic transition; phase evaluation never submits controls.
            response = self._session.step(
                choices,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                advance_sec=dt,
            )
            result = evaluate_goal(
                goal,
                self._session._runtime,
                previous_failures=self._failures,
                previous_achievements=self._achievements,
            )
            self._failures = tuple(result["failures"])
            if result["terminal"] and not result["success"] and not self._failures:
                self._failures = ("phase_deadline",)
            self._achievements = tuple(result["achieved"])
            completed = None
            if result["success"]:
                completed = self._spec.phases[self._index].id
                self._completed.append(completed)
                self._index += 1
                self._achievements = ()
            if self._index < len(self._spec.phases) and (
                response["terminated"] or response["truncated"]
            ):
                self._failures = tuple(
                    sorted(set(self._failures) | {"native_episode_ended"})
                )
            self._stopped = bool(
                self._failures
                or self._index == len(self._spec.phases)
                or response["terminated"]
                or response["truncated"]
            )
            output = {
                "transition": response,
                "campaign": self.status(),
                "completed_phase": completed,
            }
            self._history.append({"request": body, "response_sha256": digest(output)})
            self._cache[idempotency_key] = (sha, copy.deepcopy(output))
            return output

    def checkpoint(self):
        with self._lock:
            body = {
                "schema": "space-env.campaign-checkpoint/1",
                "initial": copy.deepcopy(self._initial),
                "history": copy.deepcopy(self._history),
                "status": self.status(),
                "task_checkpoint": self._session.checkpoint(),
            }
            return {**body, "sha256": digest(body)}

    @classmethod
    def from_checkpoint(cls, value):
        required = {
            "schema",
            "initial",
            "history",
            "status",
            "task_checkpoint",
            "sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value["schema"] != "space-env.campaign-checkpoint/1"
        ):
            raise ValueError("invalid campaign checkpoint")
        body = {k: v for k, v in value.items() if k != "sha256"}
        if digest(body) != value["sha256"]:
            raise ValueError("campaign checkpoint digest mismatch")
        initial = value["initial"]
        if set(initial) != {"scenario", "task", "campaign", "seed"}:
            raise ValueError("invalid campaign initial recipe")
        result = cls(**initial)
        if len(value["history"]) > result._session._task.max_decisions:
            raise ValueError("campaign replay budget exceeded")
        for row in value["history"]:
            if set(row) != {"request", "response_sha256"}:
                raise ValueError("invalid campaign replay row")
            if digest(result.step(**row["request"])) != row["response_sha256"]:
                raise ValueError("campaign transition replay mismatch")
        if result.checkpoint() != value:
            raise ValueError("campaign state replay mismatch")
        return result
