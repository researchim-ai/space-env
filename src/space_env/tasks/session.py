"""One decision transaction over MissionRuntime, with physical-time reward/replay."""

from __future__ import annotations

import copy
import hashlib
import math
from threading import RLock

from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.observation import Projection, _ProjectionReads, admissible
from space_env.tasks.schema import TaskDecision, TaskSpec

_OBJECTIVES = {
    "delivery": {
        "completion": 1.0,
        "mbits": 0.0,
        "priority": 0.0,
        "energy_j": 0.0,
        "time_sec": 0.0,
        "discard_mbits": 0.0,
        "safe_entries": 0.0,
        "rejected": 0.1,
    },
    "throughput": {
        "completion": 0.0,
        "mbits": 1.0,
        "priority": 0.0,
        "energy_j": 0.0,
        "time_sec": 0.0,
        "discard_mbits": 0.0,
        "safe_entries": 0.0,
        "rejected": 0.1,
    },
    "sla": {
        "completion": 1.0,
        "mbits": 0.0,
        "priority": 1.0,
        "energy_j": 0.0,
        "time_sec": 0.0,
        "discard_mbits": 0.0,
        "safe_entries": 0.0,
        "rejected": 0.1,
    },
    "efficient": {
        "completion": 1.0,
        "mbits": 0.0,
        "priority": 0.0,
        "energy_j": 0.001,
        "time_sec": 0.001,
        "discard_mbits": 0.1,
        "safe_entries": 0.0,
        "rejected": 0.1,
    },
    "robust": {
        "completion": 1.0,
        "mbits": 0.0,
        "priority": 0.0,
        "energy_j": 0.0,
        "time_sec": 0.0,
        "discard_mbits": 0.1,
        "safe_entries": 1.0,
        "rejected": 0.1,
    },
}


def objective_registry():
    return copy.deepcopy(_OBJECTIVES)


def seeded_scenario(scenario, seed, policy):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("episode seed must be an integer in [0,2**63)")
    value = copy.deepcopy(scenario)
    if policy == "derive-network":
        value["seed"] = seed
        definition = value["definition"]
        operations = definition.get("operations", definition)

        def derived(kind, old):
            return int.from_bytes(
                hashlib.sha256(f"{seed}:{kind}:{old}".encode()).digest()[:8], "big"
            ) % (2**63)

        for station, weather in (
            operations.get("radio_network", {}).get("weather", {}).items()
        ):
            weather["seed"] = derived("weather:" + station, weather.get("seed", 0))
        if operations.get("dtn_network") is not None:
            operations["dtn_network"]["seed"] = derived(
                "dtn", operations["dtn_network"].get("seed", 0)
            )
    return value


def score(runtime):
    optimized = runtime._task_score()
    if optimized is not None:
        return optimized
    # Trainer-only outcome accounting. These truth ledgers are never observations.
    report = runtime.observe(view="truth")["backend"]["report"]
    life = report.get("lifecycle")
    metrics = runtime.report()["metrics"]
    if life is not None:
        completion = metrics["fulfilled_requests"]
        priority = sum(
            r["priority"]
            for r in life["requests"].values()
            if r["status"] == "completed"
        )
    else:
        completion = report["on_time_bits"] / 1e6
        priority = 0.0
    mbits = (
        sum(
            p["delivered_bits"]
            for p in report["products"]
            if p["status"] in {"delivered_on_time", "delivered_late"}
        )
        / 1e6
    )
    return {
        "completion": completion,
        "mbits": mbits,
        "priority": priority,
        "energy_j": math.fsum(
            v["consumed_wh"] for v in report["energy_ledger"].values()
        )
        * 3600,
        "discard_mbits": sum(
            v["discarded_bits"] for v in report.get("inventory_ledger", {}).values()
        )
        / 1e6,
        "safe": {
            a: bool(v["safe"])
            for a, v in report.get("spacecraft_bus", {}).get("states", {}).items()
        },
    }


class TaskSession:
    def __init__(self, scenario, task, *, seed=None):
        self._lock = RLock()
        self._checkpoint_compact = False
        self._original = copy.deepcopy(scenario)
        self._task = TaskSpec.model_validate(copy.deepcopy(task))
        self._seed = scenario.get("seed", 0) if seed is None else seed
        self._scenario = seeded_scenario(scenario, self._seed, self._task.seed_policy)
        if not self._scenario["backend"].startswith("reference-"):
            raise ValueError("task/1 requires a reference MissionRuntime backend")
        self._runtime = MissionRuntime(
            self._scenario,
            access=AccessPolicy(truth=True),
            audit_mode=self._task.runtime_audit,
        )
        actor_ids = set(self._runtime.describe_capabilities()["actors"])
        definition = self._scenario["definition"]
        operations = definition.get("operations", definition)
        station_ids = {g["name"] for g in operations["stations"]}
        if any(
            (set(r.actors) | set(r.demand_actors)) - actor_ids or (set(r.stations) | set(r.demand_stations)) - station_ids
            for r in self._task.roles.values()
        ):
            raise ValueError("task role references unknown actor/station")
        if (
            self._task.max_decision_sec
            > self._runtime.describe_capabilities()["limits"]["max_advance_sec"]
        ):
            raise ValueError("task decision duration exceeds runtime advance budget")
        self._actions = list(self._task.actions)
        self._by_id = {a.id: a for a in self._actions}
        self._station_ids = sorted(station_ids)
        self._projections = {
            r: Projection(self._task) for r in ["@central", *sorted(self._task.roles)]
        }
        for p in self._projections.values():
            p.station_ids = self._station_ids
        self.revision = 0
        self._native_revision = self._runtime.revision
        self._cache = {}
        self._transitions = []
        self._views = self._refresh(self._runtime, self._projections, 0)
        self._done = False

    def _refresh(self, runtime, projections, revision):
        reads = _ProjectionReads(runtime)
        return {
            r: p.build(reads, r, revision, self._actions)
            for r, p in projections.items()
        }

    def _assert_runtime(self):
        if self._runtime.revision != self._native_revision:
            raise RuntimeContractError(
                "external_mutation", "runtime was changed outside the task transaction"
            )

    def observe(self, role="@central"):
        with self._lock:
            self._assert_runtime()
            if role not in self._views:
                raise ValueError("unknown task role")
            return copy.deepcopy(self._views[role])

    def describe(self):
        return {
            "schema": "space-env.task-capabilities/1",
            "task_id": self._task.id,
            "roles": {
                k: v.model_dump(mode="json") for k, v in self._task.roles.items()
            },
            "action_ids": ["wait", *[a.id for a in self._actions]],
            "capacities": {
                k: getattr(self._task, k)
                for k in [
                    "max_entities",
                    "max_relations",
                    "max_actions",
                    "max_joint_actions",
                    "max_decisions",
                ]
            },
            "objective": self._task.objective,
            "discount_per_sec": self._task.discount_per_sec,
            "observation": "ideal current local; no estimator; per-role independent entity slots",
            "mask": "observable conservative per-action checks; not a future-success oracle",
        }

    def step(self, choices, *, expected_revision, idempotency_key, advance_sec=None):
        with self._lock:
            self._assert_runtime()
            req = TaskDecision(
                choices=choices,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                advance_sec=advance_sec,
            )
            body = req.model_dump(
                mode="json", exclude={"expected_revision", "idempotency_key"}
            )
            sha = digest(body)
            if idempotency_key in self._cache:
                old = self._cache[idempotency_key]
                if old["request_sha256"] != sha:
                    raise RuntimeContractError(
                        "idempotency_conflict", "task key reused for different request"
                    )
                return copy.deepcopy(old["response"])
            if expected_revision != self.revision:
                raise RuntimeContractError("stale_revision", "task revision differs")
            if self._done:
                raise RuntimeContractError(
                    "episode_done", "reset/load a new task episode"
                )
            roles = set(req.choices)
            if roles - set(self._views) or ("@central" in roles and len(roles) > 1):
                raise ValueError("unknown or mixed centralized/local choices")
            ids = [
                id for role in sorted(roles) for id in req.choices[role] if id != "wait"
            ]
            if len(ids) > self._task.max_joint_actions:
                raise ValueError("joint action capacity exceeded")
            if any(id not in self._by_id for id in ids):
                raise ValueError("unknown task action")
            duration = self._task.decision_sec if advance_sec is None else advance_sec
            if duration > self._task.max_decision_sec:
                raise ValueError("decision duration exceeds declared maximum")
            before = score(self._runtime)
            start = self._runtime.time_sec
            dt = min(duration, self._runtime.horizon_sec - start)
            selected = [a for a in self._actions if a.id in ids]
            reason = None
            rejected_role = None
            if len(ids) != len(set(ids)):
                reason = "duplicate_choice"
            for role, chosen in req.choices.items():
                if role != "@central" and any(
                    id != "wait" and self._by_id[id].role != role for id in chosen
                ):
                    reason = "role_scope"
                    rejected_role = role
            if reason is None:
                owner_counts = {
                    role: sum(a.role == role for a in selected)
                    for role in self._task.roles
                }
                if any(
                    owner_counts[role] > spec.action_slots
                    for role, spec in self._task.roles.items()
                ):
                    reason = "role_action_capacity"
            candidate = self._runtime.fork()

            def native(method, **kw):
                return getattr(candidate, method)(
                    expected_revision=candidate.revision,
                    idempotency_key=f"task/{self.revision}/{candidate.revision}/{method}",
                    **kw,
                )

            try:
                if reason is None:
                    for a in selected:
                        if a.kind != "cancel_actor":
                            continue
                        ok, why = admissible(candidate, self._task, a, a.role)
                        if not ok:
                            reason = why
                            rejected_role = a.role
                            break
                        for row in candidate.command_status():
                            if row["command"]["actor_id"] == a.actor and row[
                                "status"
                            ] in {"accepted", "running"}:
                                native(
                                    "cancel", command_id=row["command"]["command_id"]
                                )
                    if reason is None:
                        for a in selected:
                            if a.kind == "cancel_actor":
                                continue
                            ok, why = admissible(candidate, self._task, a, a.role)
                            if not ok:
                                reason = why
                                rejected_role = a.role
                                break
                        if reason is None:
                            for a in selected:
                                if a.kind == "inject":
                                    e = {
                                        "id": f"@task/{self.revision}/{a.id}",
                                        "type": a.event["type"],
                                        "payload": copy.deepcopy(a.event["payload"]),
                                        "time_sec": candidate.time_sec,
                                        "source": "task/1",
                                        "reason": "declared operator task action",
                                    }
                                    native("inject", events=[e])
                            commands = [
                                dict(
                                    c.model_dump(mode="json"),
                                    command_id=f"@task/{self.revision}/{a.id}/{i}",
                                )
                                for a in selected
                                for i, c in enumerate(a.commands)
                            ]
                            if commands:
                                native("submit", commands=commands)
            except RuntimeContractError:
                reason = "runtime_rejected"
            if reason is not None:
                # Discard all zero-time mutations; rejection costs one real decision interval.
                candidate = self._runtime.fork()
            native("advance", dt_sec=dt)
            after = score(candidate)
            elapsed = candidate.time_sec - start
            components = {
                k: after[k] - before[k]
                for k in [
                    "completion",
                    "mbits",
                    "priority",
                    "energy_j",
                    "discard_mbits",
                ]
            }
            components.update(
                time_sec=elapsed,
                safe_entries=sum(
                    v and not before["safe"].get(a, False)
                    for a, v in after["safe"].items()
                ),
                rejected=float(reason is not None),
            )
            weights = _OBJECTIVES[self._task.objective]
            reward = sum(
                weights[k] * v * (1 if k in {"completion", "mbits", "priority"} else -1)
                for k, v in components.items()
            )
            terminal = candidate.time_sec >= candidate.horizon_sec - 1e-9
            truncated = self.revision + 1 >= self._task.max_decisions and not terminal
            projections = copy.deepcopy(self._projections)
            views = self._refresh(candidate, projections, self.revision + 1)
            response = {
                "schema": "space-env.task-transition/1",
                "index": self.revision,
                "revision": self.revision + 1,
                "elapsed_sec": elapsed,
                "reward": reward,
                "reward_components": components,
                "discount": 0.0 if terminal else self._task.discount_per_sec**elapsed,
                "terminated": terminal,
                "truncated": truncated,
                "feedback": {
                    "accepted": reason is None,
                    "code": "accepted" if reason is None else reason,
                    "role": rejected_role,
                },
                "observation": self._views["@central"],
                "next_observation": views["@central"],
                "role_observations": self._views,
                "next_role_observations": views,
            }
            if not math.isfinite(reward):
                raise ValueError("nonfinite task reward")
            record = {"request": req.model_dump(mode="json"), "transition": response}
            canonical_json(record)
            self._runtime = candidate
            self._native_revision = candidate.revision
            self._projections = projections
            self._views = views
            self.revision += 1
            self._done = terminal or truncated
            # Committed views/records are internally append-only. All public
            # readers return owned copies; cache/history share this evidence.
            self._transitions.append(record)
            self._cache[idempotency_key] = {
                "request_sha256": sha,
                "response": response,
            }
            return copy.deepcopy(response)

    def checkpoint(self, *, compact=None):
        with self._lock:
            compact = self._checkpoint_compact if compact is None else compact
            if type(compact) is not bool:
                raise TypeError("compact must be boolean")
            self._assert_runtime()
            body = {
                "schema": "space-env.task-episode/2"
                if compact
                else "space-env.task-episode/1",
                "scenario": copy.deepcopy(self._original),
                "task": self._task.model_dump(mode="json", by_alias=True),
                "episode_seed": self._seed,
                "transitions": [
                    {
                        "request": copy.deepcopy(r["request"]),
                        "transition_sha256": digest(r["transition"]),
                    }
                    for r in self._transitions
                ]
                if compact
                else copy.deepcopy(self._transitions),
                "runtime": self._runtime.checkpoint(compact=compact),
            }
            return {**body, "sha256": digest(body)}

    def save(self, path, *, compact=None):
        value = self.checkpoint(compact=compact)
        atomic_write(path, canonical_json(value))
        return value

    @classmethod
    def from_checkpoint(cls, value):
        required = {
            "schema",
            "scenario",
            "task",
            "episode_seed",
            "transitions",
            "runtime",
            "sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value["schema"]
            not in {"space-env.task-episode/1", "space-env.task-episode/2"}
        ):
            raise ValueError("invalid task episode envelope")
        body = {k: v for k, v in value.items() if k != "sha256"}
        if digest(body) != value["sha256"]:
            raise ValueError("task episode digest mismatch")
        result = cls(value["scenario"], value["task"], seed=value["episode_seed"])
        compact = value["schema"] == "space-env.task-episode/2"
        if len(value["transitions"]) > result._task.max_decisions:
            raise ValueError("task replay decision budget exceeded")
        for row in value["transitions"]:
            fields = (
                {"request", "transition_sha256"}
                if compact
                else {"request", "transition"}
            )
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("invalid task replay row")
            actual = result.step(**row["request"])
            if (
                (digest(actual) != row["transition_sha256"])
                if compact
                else (actual != row["transition"])
            ):
                raise ValueError("task transition replay mismatch")
        if result._runtime.checkpoint(compact=compact) != value["runtime"]:
            raise ValueError("task runtime replay mismatch")
        result._checkpoint_compact = compact
        return result

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path

        return cls.from_checkpoint(json.loads(Path(path).read_text()))
