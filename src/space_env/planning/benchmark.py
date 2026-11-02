"""Predeclared baseline evaluation, grouped splits and explicit teacher gates."""

from __future__ import annotations

import copy
import math
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from sys import version_info
from typing import Annotated, Literal

from pydantic import Field, model_validator

from space_env.eval.statistics import assign_split
from space_env.planning.model import PolicyConfig
from space_env.planning.runner import PlanningRun
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


class BenchmarkCase(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    family: str = Field(min_length=1, max_length=128)
    scenario: dict
    task: dict
    episode_seeds: tuple[Annotated[int, Field(strict=True, ge=0, lt=2**63)], ...] = (0,)
    split: Literal["train", "holdout", "ood"] | None = None

    @model_validator(mode="after")
    def validate_case(self):
        if (
            not isinstance(self.scenario.get("lineage"), str)
            or not self.scenario["lineage"].strip()
        ):
            raise ValueError("every case requires an explicit nonempty lineage")
        if (
            not 1 <= len(self.episode_seeds) <= 32
            or len(set(self.episode_seeds)) != len(self.episode_seeds)
            or any(type(s) is not int or not 0 <= s < 2**63 for s in self.episode_seeds)
        ):
            raise ValueError("invalid/duplicate episode seeds")
        return self


class TeacherGates(StrictModel):
    candidate: str = "route_aware_rolling"
    baselines: tuple[str, ...] = ("priority", "earliest_deadline")
    min_completion_fraction: float = Field(default=1.0, ge=0, le=1, strict=True)
    max_rejections: int = Field(default=0, ge=0, strict=True)
    min_cases_per_family: int = Field(default=2, ge=1, le=1000, strict=True)
    min_margin: float = Field(default=0.0, ge=0, le=1, strict=True)


class BenchmarkSpec(StrictModel):
    schema_id: Literal["space-env.planning-benchmark/1"] = Field(
        default="space-env.planning-benchmark/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1, max_length=64)
    policies: tuple[PolicyConfig, ...] = Field(min_length=1, max_length=5)
    teacher_gates: TeacherGates | None = None
    workers: int = Field(
        default=1,
        ge=1,
        le=8,
        strict=True,
        description="Parallel execution requires Python 3.11+; serial supports Python 3.10+",
    )

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.workers > 1 and version_info < (3, 11):
            raise ValueError(
                "parallel workers require Python 3.11+; use workers=1 on Python 3.10"
            )
        if len({c.id for c in self.cases}) != len(self.cases) or len(
            {p.name for p in self.policies}
        ) != len(self.policies):
            raise ValueError("duplicate case or policy")
        if sum(len(c.episode_seeds) for c in self.cases) * len(self.policies) > 2048:
            raise ValueError("benchmark episode budget exceeded")
        splits, families = {}, {}
        for c in self.cases:
            lineage = c.scenario["lineage"]
            split = c.split or assign_split(lineage, "")
            if lineage in splits and splits[lineage] != split:
                raise ValueError("one lineage cannot cross splits")
            splits[lineage] = split
            families.setdefault(c.family, set()).add(split)
        if any("ood" in s and len(s) > 1 for s in families.values()):
            raise ValueError("OOD families must be absent from train/holdout")
        if self.teacher_gates:
            names = {p.name for p in self.policies}
            g = self.teacher_gates
            if (
                g.candidate not in names
                or not g.baselines
                or len(set(g.baselines)) != len(g.baselines)
                or g.candidate in g.baselines
                or set(g.baselines) - names
            ):
                raise ValueError(
                    "teacher gates require distinct configured candidate/baselines"
                )
        return self


def qualify_teacher(spec, rows):
    gates = spec.teacher_gates
    if gates is None:
        return {"status": "not_requested", "expert_qualified": False}
    evaluation = [r for r in rows if r["split"] in {"holdout", "ood"}]
    candidate = [r for r in evaluation if r["policy"] == gates.candidate]
    families = sorted(
        {
            c.family
            for c in spec.cases
            if (c.split or assign_split(c.scenario["lineage"], ""))
            in {"holdout", "ood"}
        }
    )
    failures = []
    for family in families:
        # Independent cases, not seeds of the same case, establish family support.
        group = [r for r in candidate if r["family"] == family]
        if len({r["lineage"] for r in group}) < gates.min_cases_per_family:
            failures.append({"family": family, "reason": "insufficient_cases"})
    if not families:
        failures.append({"reason": "no_held_out_families"})
    by_key = {(r["case"], r["episode_seed"], r["policy"]): r for r in evaluation}
    for row in candidate:
        m = row.get("metrics", {})
        fraction = m.get("completion_fraction")
        if (
            row["error"]
            or fraction is None
            or fraction < gates.min_completion_fraction
            or m["rejections"] > gates.max_rejections
            or not (m["terminated"] or m["truncated"])
        ):
            failures.append(
                {
                    "case": row["case"],
                    "seed": row["episode_seed"],
                    "reason": "candidate_threshold",
                }
            )
        for baseline in gates.baselines:
            b = by_key.get((row["case"], row["episode_seed"], baseline))
            bf = None if not b else b.get("metrics", {}).get("completion_fraction")
            if (
                not b
                or b["error"]
                or bf is None
                or fraction is None
                or fraction + 1e-12 < bf + gates.min_margin
            ):
                failures.append(
                    {
                        "case": row["case"],
                        "seed": row["episode_seed"],
                        "baseline": baseline,
                        "reason": "paired_baseline_gate",
                    }
                )
    return {
        "status": "failed_declared_gates" if failures else "passed_declared_gates",
        "expert_qualified": False,
        "candidate_rows": len(candidate),
        "failures": failures,
        "scope": "only predeclared manifest thresholds; no expert/global-optimality or flight qualification",
    }


def _run_job(case, seed, policy, directory, isolated=False):
    case = BenchmarkCase.model_validate(case)
    policy = PolicyConfig.model_validate(policy)
    directory = None if directory is None else Path(directory)
    started = time.perf_counter()
    row = {
        "case": case.id,
        "family": case.family,
        "lineage": case.scenario["lineage"],
        "split": case.split or assign_split(case.scenario["lineage"], ""),
        "episode_seed": seed,
        "policy": policy.name,
        "policy_seed": policy.seed,
        "error": None,
        "metrics": {},
        "replay_verified": False,
    }
    try:
        run = PlanningRun(
            case.scenario,
            case.task,
            policy.model_dump(mode="json"),
            episode_seed=seed,
        )
        row["metrics"] = run.run()
        record = run.checkpoint(compact=run.session._task.runtime_audit == "compact")
        PlanningRun.from_checkpoint(record)
        row["replay_verified"] = True
        row["run_sha256"] = record["sha256"]
        row["episode_sha256"] = record["episode"]["sha256"]
        row["decision_trace_sha256"] = digest(record["trace"])
        if directory is not None:
            path = directory / f"{case.id}-{seed}-{policy.name}.json"
            atomic_write(path, canonical_json(record))
            row["artifact"] = path.name
    except Exception as exc:  # noqa: BLE001 — failed episodes stay in the denominator.
        row["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
    row["wall_sec"] = time.perf_counter() - started
    if isolated:
        row["worker_meta"] = {
            "pid": os.getpid(),
            "isolation": "one-episode-per-process",
        }
    return row


def _dispatch(jobs, workers):
    if workers == 1:
        for job in jobs:
            yield _run_job(*job)
        return

    def submit(pool, job):
        try:
            return pool.submit(_run_job, *job, True)
        except Exception as exc:  # noqa: BLE001 — preserve infrastructure failures in the denominator.
            future = Future()
            future.set_exception(exc)
            return future

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        pending = deque()
        for _ in range(workers):
            job = next(jobs, None)
            if job is not None:
                pending.append((job, submit(pool, job)))
        while pending:
            job, future = pending.popleft()
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 — no hidden retry or dropped case.
                case, seed, policy, _ = job
                row = {
                    "case": case["id"],
                    "family": case["family"],
                    "lineage": case["scenario"]["lineage"],
                    "split": case["split"]
                    or assign_split(case["scenario"]["lineage"], ""),
                    "episode_seed": seed,
                    "policy": policy["name"],
                    "policy_seed": policy["seed"],
                    "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                    "metrics": {},
                    "replay_verified": False,
                    "wall_sec": 0.0,
                }
            yield row
            job = next(jobs, None)
            if job is not None:
                pending.append((job, submit(pool, job)))


def run_benchmark(manifest, *, artifacts=None, _serial_replay=False):
    spec = BenchmarkSpec.model_validate(copy.deepcopy(manifest))
    frozen = spec.model_dump(mode="json", by_alias=True)
    manifest_sha = digest(frozen)
    directory = None if artifacts is None else Path(artifacts)
    if directory is not None:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("benchmark artifacts directory must be empty")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write(
            directory / "manifest.json",
            canonical_json({"manifest": frozen, "sha256": manifest_sha}),
        )
    jobs = (
        (
            case.model_dump(mode="json"),
            seed,
            policy.model_dump(mode="json"),
            None if directory is None else str(directory),
        )
        for case in spec.cases
        for seed in case.episode_seeds
        for policy in spec.policies
    )
    workers = 1 if _serial_replay else spec.workers
    rows = list(_dispatch(jobs, workers))
    summary = {}
    for p in spec.policies:
        group = [r for r in rows if r["policy"] == p.name]
        successful = [r for r in group if not r["error"]]
        summary[p.name] = {
            "episodes": len(group),
            "errors": len(group) - len(successful),
            "completion_fraction_mean_all_episodes": math.fsum(
                r.get("metrics", {}).get("completion_fraction") or 0.0
                for r in group
                if not r["error"]
            )
            / len(group),
            "zero_denominator_episodes": sum(
                r.get("metrics", {}).get("completion_fraction") is None for r in group
            ),
            "rejections": sum(r["metrics"].get("rejections", 0) for r in group),
        }
    body = {
        "schema": "space-env.planning-evaluation/1",
        "execution": {
            "requested_workers": spec.workers,
            "actual_workers": workers,
            "pending_bound": workers,
        },
        "manifest": frozen,
        "manifest_sha256": manifest_sha,
        "rows": rows,
        "summary": summary,
        "teacher": qualify_teacher(spec, rows),
        "passed_execution": all(not r["error"] and r["replay_verified"] for r in rows),
        "note": "infeasible and error episodes retained; execution success is not mission success",
    }
    return {**body, "sha256": digest(body)}


def verify_evaluation(record):
    """Recompute all semantic results; measured wall time/RSS are not replay facts."""
    body = {k: v for k, v in record.items() if k != "sha256"}
    if record.get("schema") != "space-env.planning-evaluation/1" or digest(
        body
    ) != record.get("sha256"):
        raise ValueError("invalid evaluation digest")
    fresh = run_benchmark(record["manifest"], _serial_replay=True)

    def semantic(value):
        import copy

        result = copy.deepcopy(value)
        result.pop("sha256", None)
        result.pop("execution", None)
        for row in result["rows"]:
            for key in ["wall_sec", "run_sha256", "artifact", "worker_meta"]:
                row.pop(key, None)
            for key in ["decision_wall_sec", "process_peak_rss_bytes"]:
                row["metrics"].pop(key, None)
        return result

    if semantic(fresh) != semantic(record):
        raise ValueError("evaluation semantic replay mismatch")
    return {
        "ok": True,
        "episodes": len(fresh["rows"]),
        "manifest_sha256": fresh["manifest_sha256"],
    }
