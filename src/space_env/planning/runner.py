"""One audited runner for all baselines; privileged state stays in the trainer."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from threading import RLock

from space_env.planning.model import make_input
from space_env.planning.policy import BaselinePolicy
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import TaskSession, TaskSpec


def process_peak_rss_bytes():
    """Peak of this executable, excluding Linux's inherited rusage watermark.

    On Linux ru_maxrss can survive fork/exec and retain the parent's peak. The
    current process's /proc VmHWM resets on exec and measures its own residency.
    Missing Linux proc evidence is unknown, not a substitute parent measurement.
    """
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                fields = line.split()
                if fields and fields[0] == "VmHWM:":
                    if len(fields) != 3 or fields[2] != "kB":
                        return None
                    return int(fields[1]) * 1024 if int(fields[1]) > 0 else None
        except (OSError, ValueError):
            return None
        return None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


class PlanningRun:
    def __init__(
        self, scenario, task, policy=None, *, episode_seed=None, allow_operator=False
    ):
        if type(allow_operator) is not bool:
            raise TypeError("allow_operator must be boolean")
        self.allow_operator = allow_operator
        self._lock = RLock()
        self._checkpoint_compact = False
        self.policy = BaselinePolicy(policy)
        task = copy.deepcopy(task)
        task["policy_origin"] = (
            "random" if self.policy.config.name == "random_masked" else "scripted"
        )
        if allow_operator:
            task["policy_origin"] = "external"
        self.session = TaskSession(scenario, task, seed=episode_seed)
        self.trace = []
        self.measurements = []
        self._feedback = None

    def step(self, action_ids=None):
        with self._lock:
            action_ids = copy.deepcopy(action_ids)
            if action_ids is not None:
                if not self.allow_operator:
                    raise PermissionError(
                        "operator overrides were not enabled for this run"
                    )
                if (
                    not isinstance(action_ids, list)
                    or not action_ids
                    or any(not isinstance(a, str) for a in action_ids)
                ):
                    raise ValueError("operator override requires declared action IDs")
            if self.session.revision != len(self.trace):
                raise ValueError("task session changed outside planning runner")
            state = self.policy.checkpoint()
            started = time.perf_counter()
            try:
                public = make_input(
                    self.session,
                    self._feedback,
                    conditional_claims=self.policy.config.reclaim_acked_downlinks,
                )
                decision = self.policy.decide(public)
                if action_ids is not None:
                    decision = {
                        **decision,
                        "action_ids": list(action_ids),
                        "reason": "operator_override",
                        "baseline_action_ids": decision["action_ids"],
                    }
                    self.policy._last_decision = copy.deepcopy(decision)
                canonical_json(decision)
                elapsed_policy = time.perf_counter() - started
                transition = self.session.step(
                    {"@central": decision["action_ids"]},
                    expected_revision=self.session.revision,
                    idempotency_key=f"baseline/{len(self.trace)}",
                    advance_sec=decision["advance_sec"],
                )
            except Exception:
                self.policy = BaselinePolicy.from_checkpoint(state)
                raise
            row = {
                "index": len(self.trace),
                "override": copy.deepcopy(action_ids),
                "input_sha256": public.sha256,
                "decision": decision,
                "feedback": copy.deepcopy(transition["feedback"]),
                "transition_sha256": digest(transition),
                "policy_sha256": self.policy.checkpoint()["sha256"],
            }
            self.trace.append(row)
            self.measurements.append(
                {
                    "decision_wall_sec": elapsed_policy,
                    "step_wall_sec": time.perf_counter() - started,
                    "process_peak_rss_bytes": process_peak_rss_bytes(),
                }
            )
            self._feedback = copy.deepcopy(transition["feedback"])
            return transition

    @property
    def done(self):
        return self.session._done

    def run(self, decisions=None):
        if decisions is not None and (
            isinstance(decisions, bool)
            or not isinstance(decisions, int)
            or decisions < 0
        ):
            raise ValueError("decisions must be a nonnegative integer")
        count = 0
        while not self.done and (decisions is None or count < decisions):
            self.step()
            count += 1
        return self.metrics()

    def metrics(self):
        # Evaluator-only diagnostics, never fed back to a policy.
        report = self.session._runtime.observe(view="truth")["backend"]["report"]
        life = report.get("lifecycle")
        outcomes = report["products"]
        complete = [
            p
            for p in outcomes
            if p["status"] in {"delivered_on_time", "delivered_late"}
        ]
        total = life["metrics"]["total_requests"] if life is not None else len(outcomes)
        fulfilled = (
            life["metrics"]["fulfilled_requests"]
            if life is not None
            else sum(p["status"] == "delivered_on_time" for p in outcomes)
        )
        native = self.session._runtime._backend.mission
        latencies = [
            p["completed_sec"] - native.products[p["id"]].ready_sec for p in complete
        ]
        late = [
            max(0.0, p["completed_sec"] - native.products[p["id"]].deadline_sec)
            for p in complete
        ]
        return {
            "fulfilled": fulfilled,
            "total": total,
            "completion_fraction": fulfilled / total if total else None,
            "on_time_ground_mbits": sum(
                p["delivered_bits"]
                for p in complete
                if p["status"] == "delivered_on_time"
            )
            / 1e6,
            "acked_ground_mbits": sum(p["delivered_bits"] for p in complete) / 1e6,
            "completion_latency_sec": latencies,
            "lateness_sec": late,
            "energy_j": math.fsum(
                v["consumed_wh"] for v in report["energy_ledger"].values()
            )
            * 3600,
            "discarded_bits": sum(
                v["discarded_bits"] for v in report["inventory_ledger"].values()
            ),
            "wasted_partial_bits": sum(
                report.get("dtn", {}).get("wasted_partial_bits", {}).values()
            ),
            "sampled_safe_entries": sum(
                t["transition"]["reward_components"]["safe_entries"]
                for t in self.session._transitions
            ),
            "rejections": sum(not t["feedback"]["accepted"] for t in self.trace),
            "search_nodes": sum(t["decision"]["search_nodes"] for t in self.trace),
            "decisions": len(self.trace),
            "physical_sec": self.session.observe()["time_sec"],
            "terminated": self.done
            and bool(self.session._transitions[-1]["transition"]["terminated"]),
            "truncated": self.done
            and bool(self.session._transitions[-1]["transition"]["truncated"]),
            "decision_wall_sec": [m["decision_wall_sec"] for m in self.measurements],
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "rss_scope": "process-lifetime high-water mark, not episode-exclusive",
            "qualification": "synthetic baseline execution, not expert/flight qualification",
        }

    def checkpoint(self, *, compact=None):
        with self._lock:
            compact = self._checkpoint_compact if compact is None else compact
            if type(compact) is not bool:
                raise TypeError("compact must be boolean")
            if self.session.revision != len(self.trace):
                raise ValueError("task session changed outside planning runner")
            body = {
                "schema": "space-env.planning-run/2"
                if compact
                else "space-env.planning-run/1",
                "operator_enabled": self.allow_operator,
                "episode": self.session.checkpoint(compact=compact),
                "policy": self.policy.checkpoint(),
                "trace": copy.deepcopy(self.trace),
                "measurements": copy.deepcopy(self.measurements),
            }
            return {**body, "sha256": digest(body)}

    def save(self, path, *, compact=None):
        record = self.checkpoint(compact=compact)
        atomic_write(path, canonical_json(record))
        return record

    @classmethod
    def from_checkpoint(cls, record):
        if not isinstance(record, dict) or set(record) != {
            "schema",
            "operator_enabled",
            "episode",
            "policy",
            "trace",
            "measurements",
            "sha256",
        }:
            raise ValueError("invalid planning checkpoint")
        body = {k: v for k, v in record.items() if k != "sha256"}
        if (
            record["schema"]
            not in {"space-env.planning-run/1", "space-env.planning-run/2"}
            or digest(body) != record["sha256"]
        ):
            raise ValueError("planning checkpoint digest mismatch")
        episode = record["episode"]
        compact = record["schema"] == "space-env.planning-run/2"
        if (
            type(record["operator_enabled"]) is not bool
            or not isinstance(record["trace"], list)
            or not isinstance(record["measurements"], list)
        ):
            raise ValueError("invalid planning checkpoint types")
        if not isinstance(episode, dict) or set(episode) != {
            "schema",
            "scenario",
            "task",
            "episode_seed",
            "transitions",
            "runtime",
            "sha256",
        }:
            raise ValueError("invalid planning episode fields")
        if (
            episode["schema"]
            != ("space-env.task-episode/2" if compact else "space-env.task-episode/1")
            or not isinstance(episode["transitions"], list)
            or digest({k: v for k, v in episode.items() if k != "sha256"})
            != episode["sha256"]
        ):
            raise ValueError("invalid planning episode digest/types")
        task_spec = TaskSpec.model_validate(episode["task"])
        BaselinePolicy.from_checkpoint(record["policy"])
        if (
            len(record["trace"]) != len(episode["transitions"])
            or len(record["trace"]) > task_spec.max_decisions
        ):
            raise ValueError("planning trace budget/length mismatch")
        result = cls(
            episode["scenario"],
            episode["task"],
            record["policy"]["config"],
            episode_seed=episode["episode_seed"],
            allow_operator=record["operator_enabled"],
        )
        for expected in record["trace"]:
            if not isinstance(expected, dict) or set(expected) != {
                "index",
                "override",
                "input_sha256",
                "decision",
                "feedback",
                "transition_sha256",
                "policy_sha256",
            }:
                raise ValueError("invalid planning trace fields")
            result.step(expected["override"])
            if result.trace[-1] != expected:
                raise ValueError("planning decision/transition replay mismatch")
        if (
            result.session.checkpoint(compact=compact) != episode
            or result.policy.checkpoint() != record["policy"]
        ):
            raise ValueError("planning final replay mismatch")
        measurements = record["measurements"]
        if len(measurements) != len(result.trace):
            raise ValueError("planning measurement count mismatch")
        for row in measurements:
            if set(row) != {
                "decision_wall_sec",
                "step_wall_sec",
                "process_peak_rss_bytes",
            } or any(
                isinstance(row[k], bool)
                or not isinstance(row[k], (int, float))
                or not math.isfinite(row[k])
                or row[k] < 0
                for k in ["decision_wall_sec", "step_wall_sec"]
            ):
                raise ValueError("invalid planning measurements")
            rss = row["process_peak_rss_bytes"]
            if rss is not None and (
                isinstance(rss, bool) or not isinstance(rss, int) or rss < 0
            ):
                raise ValueError("invalid RSS measurement")
        result.measurements = copy.deepcopy(measurements)
        result._checkpoint_compact = compact
        return result

    @classmethod
    def load(cls, path):
        return cls.from_checkpoint(json.loads(Path(path).read_text()))
