"""Role-only live controller sharing the exact offline observation projection."""

from __future__ import annotations

import copy
import os
from threading import RLock

from space_env.planning.actor_projection import (
    actor_catalogues,
    actor_roles,
    project_actor_observation,
)
from space_env.planning.chain import verify_chain_proposal
from space_env.planning.incident import verify_incident
from space_env.planning.incident_journal import current_plan, verify_journal
from space_env.planning.mission_plan import (
    PlanExecution,
    _sealed,
    make_mission_plan,
    validate_execution,
    validate_plan,
)
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec, VerifiedEpisode

MAX_CHECKPOINT_BYTES = 128_000_000


class ActorCoordinator:
    """Single-owner atomic joint decisions; operator actions are never selectable."""

    def __init__(self, reference):
        reference = copy.deepcopy(reference)
        schema = reference.get("schema")
        if schema == "space-env.incident-recovery/1":
            verify_incident(reference)
            plan = reference["plan"]
            origin = reference["source_proposal"]
            cutoff = reference["committed_prefix_steps"]
        elif schema == "space-env.incident-journal/1":
            audit = verify_journal(reference, native=True)
            plan = current_plan(reference)
            origin = reference["origin"]
            cutoff = audit["committed_prefix_steps"]
        elif schema == "space-env.demand-journal/1":
            from space_env.planning.demand_journal import (
                current_demand_plan,
                verify_demand_journal,
            )

            audit = verify_demand_journal(reference, native=True)
            plan = current_demand_plan(reference)
            origin = reference["origin"]
            cutoff = audit["committed_prefix_steps"]
        elif schema == "space-env.chain-proposal/1":
            verify_chain_proposal(reference, native=True)
            plan = reference["plan"]
            origin = reference
            cutoff = 0
        else:
            raise ValueError("unsupported actor controller reference")
        self._catalogue_version = (
            reference.get("catalogue_version")
            if schema == "space-env.demand-journal/1"
            else None
        )
        self._superseded = False
        self._reference = reference
        self._plan = copy.deepcopy(plan)
        self._cutoff = cutoff
        self._spec = TaskSpec.model_validate(plan["task"])
        self._roles = actor_roles(origin["plan"]["task"])
        self._catalogues = actor_catalogues(self._spec, self.roles)
        execution = PlanExecution(plan)
        execution.run(decisions=cutoff)
        if execution.cursor != cutoff or execution.checkpoint()["error"] is not None:
            raise ValueError("actor committed prefix replay failed")
        self._session = execution._session
        self._trace = []
        self._keys = {}
        self._lock = RLock()
        self._cursor = cutoff

    @property
    def roles(self):
        return self._roles

    @property
    def catalogues(self):
        return copy.deepcopy(self._catalogues)

    @property
    def catalogue_version(self):
        return self._catalogue_version

    def rebind(self, reference):
        """Move this live owner to a native-verified successor demand epoch."""
        with self._lock:
            self._assert_cursor()
            if reference.get(
                "schema"
            ) != "space-env.demand-journal/1" or not reference.get("epochs"):
                raise ValueError("rebind requires successor demand epoch")
            node = reference["epochs"][-1]
            episode = self._session.checkpoint(compact=True)
            if (
                node["parent_plan_sha256"] != self._plan["sha256"]
                or node["source_episode_sha256"] != episode["sha256"]
            ):
                raise ValueError("rebind source does not match live actor prefix")
            result = ActorCoordinator(reference)
            self._superseded = True
            return result

    @property
    def cursor(self):
        return self._cursor

    @property
    def done(self):
        return self._session._done

    def _assert_cursor(self):
        if self._superseded:
            raise ValueError("actor epoch superseded; use rebound controller")
        if self._session.revision != self._cursor:
            raise ValueError("actor coordinator session externally mutated")

    def observe(self):
        with self._lock:
            self._assert_cursor()
            return {
                r: project_actor_observation(
                    self._session.observe(r),
                    self._spec,
                    r,
                    catalogue_version=self._catalogue_version,
                )
                for r in self.roles
            }

    def _response(self, r):
        return {
            "before_observations": {
                a: project_actor_observation(
                    r["role_observations"][a],
                    self._spec,
                    a,
                    catalogue_version=self._catalogue_version,
                )
                for a in self.roles
            },
            "observations": {
                a: project_actor_observation(
                    r["next_role_observations"][a],
                    self._spec,
                    a,
                    catalogue_version=self._catalogue_version,
                )
                for a in self.roles
            },
            "reward": r["reward"],
            "reward_scope": "joint_team",
            "discount": r["discount"],
            "terminated": r["terminated"],
            "truncated": r["truncated"],
            "feedback": copy.deepcopy(r["feedback"]),
            "index": r["index"],
            "elapsed_sec": r["elapsed_sec"],
        }

    def step(self, choices, *, expected_bindings, idempotency_key):
        with self._lock:
            if (
                not isinstance(idempotency_key, str)
                or not 1 <= len(idempotency_key) <= 128
            ):
                raise ValueError("bounded actor idempotency key required")
            if (
                not isinstance(choices, dict)
                or set(choices) != set(self.roles)
                or not isinstance(expected_bindings, dict)
                or set(expected_bindings) != set(self.roles)
            ):
                raise ValueError("one decision/binding for every actor role required")
            normalized = {}
            for role, values in choices.items():
                if (
                    not isinstance(values, (list, tuple))
                    or not 1 <= len(values) <= self._spec.roles[role].action_slots
                    or any(
                        not isinstance(a, str) or a not in self._catalogues[role]
                        for a in values
                    )
                    or len(set(values)) != len(values)
                    or "wait" in values
                    and len(values) > 1
                ):
                    raise ValueError("actor action outside role catalogue/slots")
                normalized[role] = list(values)
            request = {
                "choices": normalized,
                "expected_bindings": copy.deepcopy(expected_bindings),
                "idempotency_key": idempotency_key,
            }
            sha = digest(request)
            if idempotency_key in self._keys:
                previous, index = self._keys[idempotency_key]
                if previous != sha:
                    raise ValueError("actor idempotency conflict")
                return self._response(self._session._transitions[index]["transition"])
            self._assert_cursor()
            if self.done:
                raise ValueError("actor episode is over")
            observations = self.observe()
            if any(
                expected_bindings[a] != observations[a]["binding"] for a in self.roles
            ):
                raise ValueError("stale actor observation binding")
            ids = sorted(
                a for values in normalized.values() for a in values if a != "wait"
            ) or ["wait"]
            r = self._session.step(
                {"@central": ids},
                expected_revision=self._cursor,
                idempotency_key=f"mission-plan/{self._cursor}",
                advance_sec=self._spec.decision_sec,
            )
            self._cursor = self._session.revision
            response = self._response(r)
            self._trace.append(
                {"request": request, "response_sha256": digest(response)}
            )
            self._keys[idempotency_key] = (sha, self._cursor - 1)
            return response

    def checkpoint(self):
        with self._lock:
            self._assert_cursor()
            body = {
                "schema": "space-env.actor-coordinator/1",
                "reference": self._reference,
                "trace": self._trace,
                "episode": self._session.checkpoint(compact=True),
                "cursor": self._cursor,
            }
            value = body | {"sha256": digest(body)}
            if len(canonical_json(value).encode()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("actor checkpoint byte budget exceeded")
            return copy.deepcopy(value)

    @classmethod
    def from_checkpoint(cls, value):
        if (
            len(canonical_json(value).encode()) > MAX_CHECKPOINT_BYTES
            or not _sealed(value)
            or set(value)
            != {"schema", "reference", "trace", "episode", "cursor", "sha256"}
            or value["schema"] != "space-env.actor-coordinator/1"
            or not isinstance(value["trace"], list)
            or type(value["cursor"]) is not int
            or len(value["trace"]) > 10000
        ):
            raise ValueError("invalid actor coordinator checkpoint")
        result = cls(value["reference"])
        for row in value["trace"]:
            response = result.step(**row["request"])
            if digest(response) != row["response_sha256"]:
                raise ValueError("actor response native replay mismatch")
        if result.cursor != value["cursor"] or canonical_json(
            result._session.checkpoint(compact=True)
        ) != canonical_json(value["episode"]):
            raise ValueError("actor checkpoint native episode mismatch")
        if canonical_json(result.checkpoint()) != canonical_json(value):
            raise ValueError("actor checkpoint trace normalization mismatch")
        return result

    def as_execution(self):
        """Seal actual actor prefix as fixed proposal for journal/bundle consumers."""
        with self._lock:
            self._assert_cursor()
            episode = self._session.checkpoint(compact=True)
            steps = [
                {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")}
                for s in self._plan["steps"]
            ]
            for i, row in enumerate(episode["transitions"]):
                steps[i]["action_ids"] = copy.deepcopy(
                    row["request"]["choices"]["@central"]
                )
            plan = make_mission_plan(
                self._plan["scenario"],
                self._plan["task"],
                steps,
                goals=self._plan["goals"],
                episode_seed=self._plan["episode_seed"],
            )
            body = {
                "schema": "space-env.mission-plan-execution/1",
                "plan_sha256": plan["sha256"],
                "build": plan["build"],
                "producer_pid": os.getpid(),
                "episode": episode,
                "error": None,
                "recovery": None,
            }
            record = body | {"sha256": digest(body)}
            validate_execution(validate_plan(plan), record)
            return {"plan": plan, "execution": record}

    def verified_rows(self, *, sft=False):
        """Cold-verified actor suffix rows, matching incident offline format."""
        from space_env.planning.mission_bundle import _quality, certificate

        sealed = self.as_execution()
        verified = VerifiedEpisode(sealed["execution"]["episode"])
        cert = certificate(sealed["plan"], sealed["execution"], _quality(verified))
        if sft and not cert["passed"]:
            raise ValueError("actor SFT requires completed native mission goal")
        context = {
            "plan": sealed["plan"],
            "sha256": self._reference["sha256"],
            "committed_prefix_steps": self._cutoff,
            "catalogue_version": self._catalogue_version,
        }
        for role in self.roles:
            yield from online_rows(verified, context, role, sft=sft)


def online_rows(verified, context, role, *, sft=False):
    from space_env.planning.incident_data import role_rows

    for row in role_rows(verified, context, role, sft=sft):
        row["controller_reference_sha256"] = row.pop("incident_recovery_sha256")
        row["schema"] = (
            "space-env.actor-online-sft/1" if sft else "space-env.actor-online-rl/1"
        )
        yield row
