"""Equal-information chain benchmark; bounded search and native-only scoring."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.planning.chain import (
    collect_chain_forecast,
    materialize_task,
    profile,
    schedule_chains,
    validate_view,
)
from space_env.planning.mission_bundle import (
    publish_plan_bundle,
    read_json,
    verify_plan_bundle,
)
from space_env.planning.mission_plan import (
    PlanExecution,
    PlanGoals,
    _sealed,
    make_mission_plan,
)
from space_env.planning.runner import process_peak_rss_bytes
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest

POLICIES = ("priority_greedy", "deadline_rule", "bounded_order_search")


class Budget(StrictModel):
    max_search_nodes: int = Field(default=1000000, ge=1, le=2000000, strict=True)
    max_orders: int = Field(default=8, ge=2, le=32, strict=True)
    timeout_sec: int = Field(default=300, ge=1, le=3600, strict=True)
    max_result_bytes: int = Field(default=64000000, ge=1024, le=256000000, strict=True)
    max_rss_bytes: int = Field(
        default=2147483648, ge=1000000, le=8589934592, strict=True
    )


class Gates(StrictModel):
    min_test_groups: int = Field(default=2, ge=2, le=32, strict=True)
    min_completion_fraction: float = Field(default=0.1, gt=0, le=1, strict=True)
    min_paired_margin: float = Field(default=0.0, ge=0, le=1, strict=True)
    max_rejections: int = Field(default=0, ge=0, strict=True)


class Case(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    family: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    split: Literal["train", "development", "test"]
    group: str = Field(min_length=1, max_length=128)
    geometry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario: dict
    task: dict
    goals: dict
    negative: bool = Field(default=False, strict=True)


class Suite(StrictModel):
    schema_id: Literal["space-env.release-suite/1"] = Field(alias="schema")
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    cases: tuple[Case, ...] = Field(min_length=1, max_length=32)
    budget: Budget = Field(default_factory=Budget)
    gates: Gates = Field(default_factory=Gates)
    scope: str = Field(min_length=1, max_length=1000)
    max_total_artifact_bytes: int = Field(
        default=1073741824, ge=1000000, le=8589934592, strict=True
    )

    @model_validator(mode="after")
    def split_integrity(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case")
        groups, geometries, definitions = {}, {}, set()
        for c in self.cases:
            d = c.scenario["definition"]
            profile(c.scenario, c.task)
            goals = PlanGoals.model_validate(c.goals)
            if goals.min_fulfilled > len(d["lifecycle"]["requests"]):
                raise ValueError("service floor exceeds request denominator")
            if goals.coupled is None or goals.coupled.native_spec_sha256 != digest(d):
                raise ValueError("release suite requires scenario-bound coupled goals")
            if (
                d["horizon_sec"] > 1800
                or c.task["decision_sec"] < 5
                or c.task["max_decisions"] > 361
            ):
                raise ValueError("release benchmark bounded to 1800s/361 decisions")
            geometry = digest(
                {
                    "orbits": d["orbits"],
                    "targets": d["acquisition_targets"],
                    "stations": d["operations"]["stations"],
                }
            )
            if c.geometry_sha256 != geometry:
                raise ValueError("declared geometry mismatch")
            for key, table in ((c.group, groups), (geometry, geometries)):
                if key in table and table[key] != c.split:
                    raise ValueError("physical group crosses split")
                table[key] = c.split
            identity = digest(d)
            if identity in definitions:
                raise ValueError("duplicate physical definition")
            definitions.add(identity)
            if c.goals.get("min_fulfilled", 0) < 1:
                raise ValueError(
                    "positive service floor required even for negative cases"
                )
        return self


def select_schedule(view, policy, budget=None):
    """All policies see only this same sealed forecast, never native outcomes."""
    validate_view(view)
    if policy not in POLICIES:
        raise ValueError("unknown equal-information policy")
    cfg = Budget.model_validate(budget or {})
    chains = view["chains"]
    deadline = [
        c["id"]
        for c in sorted(
            chains, key=lambda c: (c["deadline_sec"], -c["priority"], c["id"])
        )
    ]
    priority = [
        c["id"]
        for c in sorted(
            chains, key=lambda c: (-c["priority"], c["deadline_sec"], c["id"])
        )
    ]
    orders = [priority] if policy == POLICIES[0] else [deadline]
    if policy == POLICIES[2]:
        orders = [deadline, priority, list(reversed(deadline))]
        orders += [
            deadline[k:] + deadline[:k]
            for k in range(1, min(len(deadline), cfg.max_orders))
        ]
    distinct = []
    for order in orders:
        if order not in distinct:
            distinct.append(order)
    distinct = distinct[: cfg.max_orders]
    used = 0
    best = None
    attempts = []
    by_id = {c["id"]: c for c in chains}
    for order in distinct:
        remaining = cfg.max_search_nodes - used
        if remaining < 1:
            break
        try:
            schedule = schedule_chains(
                view, {"max_search_nodes": remaining}, order=order
            )
        except ValueError as exc:
            if "search budget exceeded" not in str(exc):
                raise
            # The failed call consumed the remaining allowance. No reset/retry.
            attempts.append({"order": order, "budget_exhausted": True})
            used = cfg.max_search_nodes
            break
        used += schedule["search_nodes"]
        score = (
            sum(by_id[a["request"]]["priority"] for a in schedule["allocations"]),
            schedule["predicted_fulfilled"],
        )
        attempts.append(
            {
                "order": order,
                "score": list(score),
                "search_nodes": schedule["search_nodes"],
            }
        )
        if best is None or score > best[0]:
            best = (score, schedule)
    if best is None:
        raise ValueError("search budget exhausted before any complete schedule")
    return {
        "policy": policy,
        "forecast_sha256": view["sha256"],
        "schedule": best[1],
        "total_search_nodes": used,
        "attempts": attempts,
        "budget": cfg.model_dump(mode="json"),
        "scope": "bounded order search over serial greedy packer; not globally optimal MILP",
    }


def native_metrics(case, report, cert):
    requests = report["lifecycle"]["requests"]
    original = case["scenario"]["definition"]["lifecycle"]["requests"]
    if set(requests) != {r["id"] for r in original}:
        raise ValueError("native demand denominator differs")
    served = {k for k, v in requests.items() if v["status"] == "completed"}
    groups = {a: [] for a in case["scenario"]["actor_order"]}
    for r in original:
        groups[r["eligible_actors"][0]].append(r["id"])
    fractions = {
        a: sum(k in served for k in ids) / len(ids) for a, ids in groups.items()
    }
    values = list(fractions.values())
    weighted = sum(r.get("priority", 1) for r in original if r["id"] in served)
    return {
        "fulfilled": len(served),
        "total": len(original),
        "completion_fraction": len(served) / len(original),
        "served_priority": weighted,
        "total_priority": sum(r.get("priority", 1) for r in original),
        "unserved": len(original) - len(served),
        "missed_deadlines": sum(
            k not in served and r["deadline_sec"] <= report["time_sec"]
            for k, r in requests.items()
        ),
        "per_actor_completion": fractions,
        "jain_fairness": sum(values) ** 2 / (len(values) * sum(x * x for x in values))
        if any(values)
        else 0.0,
        "rejections": cert["rejections"],
        "mission_goal_passed": cert["passed"],
        "bit_balance_error": sum(
            abs(v["balance_error_bits"]) for v in report["lifecycle"]["ledger"].values()
        ),
        "unserved_base_wh": sum(report["unserved_base_wh"].values()),
    }


def teacher_report(spec, rows):
    spec = Suite.model_validate(spec)
    expected = {(c.id, p) for c in spec.cases for p in POLICIES}
    if (
        len(rows) != len(expected)
        or {(r["case"], r["policy"]) for r in rows} != expected
    ):
        raise ValueError("every paired row including failures is required")
    cases = {c.id: c for c in spec.cases}
    for row in rows:
        if row.get("error") is not None:
            n = len(cases[row["case"]].scenario["definition"]["lifecycle"]["requests"])
            if row.get("total") != n:
                raise ValueError("failed row lost demand denominator")
            continue
        m = row.get("metrics", {})
        n = len(cases[row["case"]].scenario["definition"]["lifecycle"]["requests"])
        fraction = m.get("completion_fraction")
        if (
            type(m.get("fulfilled")) is not int
            or not 0 <= m["fulfilled"] <= n
            or m.get("total") != n
            or type(fraction) not in (int, float)
            or not math.isfinite(fraction)
            or abs(fraction - m["fulfilled"] / n) > 1e-12
            or type(m.get("rejections")) is not int
            or m["rejections"] < 0
            or type(m.get("mission_goal_passed")) is not bool
        ):
            raise ValueError("invalid native metric denominator or fraction")
    for c in spec.cases:
        hashes = {
            r.get("forecast_sha256")
            for r in rows
            if r["case"] == c.id and r.get("error") is None
        }
        if len(hashes) > 1 or None in hashes:
            raise ValueError("paired policies did not share forecast")
    table = {(r["case"], r["policy"]): r for r in rows}
    failures = []
    intervals = {}
    families = {c.family for c in spec.cases if not c.negative}
    for family in sorted(families):
        held = [
            c
            for c in spec.cases
            if c.split == "test" and c.family == family and not c.negative
        ]
        grouped = {}
        if len({c.group for c in held}) < spec.gates.min_test_groups:
            failures.append({"family": family, "reason": "insufficient_test_groups"})
        for c in held:
            for p in POLICIES:
                row = table[c.id, p]
                if (
                    row.get("error")
                    or row.get("native_replay_verified") is not True
                    or row.get("resource_gate_passed") is not True
                ):
                    failures.append(
                        {
                            "case": c.id,
                            "policy": p,
                            "reason": "invalid_native_or_resource_gate",
                        }
                    )
            candidate = table[c.id, POLICIES[2]]
            m = candidate.get("metrics", {})
            if (
                not m.get("mission_goal_passed")
                or m.get("completion_fraction", 0) < spec.gates.min_completion_fraction
                or m.get("rejections", math.inf) > spec.gates.max_rejections
            ):
                failures.append({"case": c.id, "reason": "candidate_service_gate"})
            for p in POLICIES[:2]:
                delta = m.get("completion_fraction", 0) - table[c.id, p].get(
                    "metrics", {}
                ).get("completion_fraction", 0)
                grouped.setdefault(p, {}).setdefault(c.group, []).append(delta)
        for policy, groups in grouped.items():
            samples = [sum(x) / len(x) for x in groups.values()]
            mean = sum(samples) / len(samples)
            radius = math.sqrt(2 * math.log(40) / len(samples))
            lo, hi = max(-1.0, mean - radius), min(1.0, mean + radius)
            intervals[family + "/" + policy] = {
                "groups": len(samples),
                "mean_delta": mean,
                "lower": lo,
                "upper": hi,
                "method": "conditional 95% Hoeffding bound IF groups are independent draws; frozen geometries do not establish population sampling; no IID claim for variants",
            }
            if lo < spec.gates.min_paired_margin:
                failures.append(
                    {
                        "family": family,
                        "baseline": policy,
                        "reason": "uncertainty_or_margin_gate",
                    }
                )
    negatives = [c for c in spec.cases if c.negative]
    if not negatives:
        failures.append({"reason": "missing_negative_control"})
    for c in negatives:
        for p in POLICIES:
            row = table[c.id, p]
            m = row.get("metrics", {})
            if (
                row.get("error")
                or row.get("native_replay_verified") is not True
                or m.get("fulfilled") != 0
                or m.get("mission_goal_passed") is not False
            ):
                failures.append(
                    {
                        "case": c.id,
                        "policy": p,
                        "reason": "negative_control_not_verified",
                    }
                )
    return {
        "passed_declared_gates": not failures,
        "eligible_sft_candidate_selection": not failures,
        "expert_teacher_qualified": False,
        "failures": failures,
        "paired_intervals": intervals,
        "scope": "finite declared suite only; synthetic data and conservative group uncertainty; no global expertise claim",
    }


def _worker(case, policy, budget, directory, forecast=None):
    root = Path(directory)
    cfg = Budget.model_validate(budget)
    runtime = MissionRuntime(case["scenario"], access=AccessPolicy(forecast=True))
    if policy == "prepare":
        view = collect_chain_forecast(runtime, case["scenario"], case["task"])
        atomic_write(root / "forecast.json", canonical_json(view))
        return
    started = time.perf_counter()
    selection = select_schedule(forecast, policy, budget)
    planning_sec = time.perf_counter() - started
    schedule = selection["schedule"]
    pointings = {}
    by_id = {c["id"]: c for c in forecast["chains"]}
    for a in schedule["allocations"]:
        pointings[a["request"]] = runtime.forecast_pointing(
            a["actor"], by_id[a["request"]]["target"], a["pointing_at_sec"]
        )
    task = materialize_task(case["task"], forecast, schedule, pointings)
    plan = make_mission_plan(
        case["scenario"], task, schedule["steps"], goals=case["goals"]
    )
    execution = PlanExecution(plan)
    latencies = []
    started = time.perf_counter()
    while not execution.complete:
        t = time.perf_counter()
        execution.step()
        latencies.append(time.perf_counter() - t)
    execution_sec = time.perf_counter() - started
    t = time.perf_counter()
    record = execution.checkpoint()
    checkpoint_sec = time.perf_counter() - t
    checkpoint_bytes = len(canonical_json(record).encode())
    if checkpoint_bytes > cfg.max_result_bytes:
        raise ValueError("checkpoint byte budget exceeded")
    t = time.perf_counter()
    restored = PlanExecution.from_checkpoint(plan, record)
    replay_sec = time.perf_counter() - t
    assert restored.checkpoint()["episode"]["sha256"] == record["episode"]["sha256"]
    t = time.perf_counter()
    publish_plan_bundle(plan, record, root / "dataset")
    export_sec = time.perf_counter() - t
    # Bundle publication already cold-verifies native state and every exported row.
    cert = read_json(root / "dataset/certificate.json")
    audit = verify_plan_bundle(root / "dataset", native=False)
    atomic_write(root / "pointings.json", canonical_json(pointings))
    atomic_write(root / "selection.json", canonical_json(selection))
    rss = process_peak_rss_bytes()
    bytes_used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    atomic_write(root / "selection.json", canonical_json(selection))
    body = {
        "case": case["id"],
        "policy": policy,
        "forecast_sha256": forecast["sha256"],
        "native_replay_verified": True,
        "dataset_audit": audit,
        "metrics": native_metrics(
            case, record["episode"]["runtime"]["backend_report"], cert
        ),
        "measurements": {
            "planning_sec": planning_sec,
            "execution_sec": execution_sec,
            "checkpoint_sec": checkpoint_sec,
            "replay_sec": replay_sec,
            "export_sec": export_sec,
            "exported_rows": sum(audit["dataset_rows"].values()),
            "exported_rows_per_sec": sum(audit["dataset_rows"].values()) / export_sec,
            "checkpoint_bytes": checkpoint_bytes,
            "artifact_bytes": bytes_used,
            "step_p50_sec": float(np.percentile(latencies, 50)),
            "step_p95_sec": float(np.percentile(latencies, 95)),
            "peak_rss_bytes": rss,
            "search_nodes": selection["total_search_nodes"],
        },
        "resource_gate_passed": rss is not None
        and rss <= cfg.max_rss_bytes
        and bytes_used + 64000 <= cfg.max_result_bytes,
        "error": None,
        "expert_teacher_qualified": False,
    }
    atomic_write(root / "result.json", canonical_json(body))


def summarize_envelope(value, rows):
    spec = Suite.model_validate(value)
    result = {}
    for c in spec.cases:
        d = c.scenario["definition"]
        cell = f"actors={len(d['orbits'])}/requests={len(d['lifecycle']['requests'])}/horizon={d['horizon_sec']}"
        group = result.setdefault(
            cell,
            {
                "rows": 0,
                "native_verified": 0,
                "resource_passed": 0,
                "service_passed": 0,
                "max_step_p95_sec": 0.0,
                "max_peak_rss_bytes": 0,
                "max_checkpoint_bytes": 0,
            },
        )
        for row in rows:
            if row["case"] != c.id:
                continue
            group["rows"] += 1
            if row.get("error"):
                continue
            group["native_verified"] += int(row.get("native_replay_verified") is True)
            group["resource_passed"] += int(row.get("resource_gate_passed") is True)
            group["service_passed"] += int(row["metrics"]["mission_goal_passed"])
            m = row["measurements"]
            for name, key in (
                ("max_step_p95_sec", "step_p95_sec"),
                ("max_peak_rss_bytes", "peak_rss_bytes"),
                ("max_checkpoint_bytes", "checkpoint_bytes"),
            ):
                if m[key] is not None:
                    group[name] = max(group[name], m[key])
    return {
        "cells": result,
        "scope": "observed fresh-process probes on this host; variants/policies are not independent performance repeats; no interpolation or fleet-scale SLA; RSS is an observed gate, not OS memory isolation",
    }


def _launch(packet, path, budget):
    atomic_write(path / "packet.json", canonical_json(packet))
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (path / "worker.log").open("wb") as log:
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "space_env.planning.release_benchmark",
                    "worker",
                    str(path / "packet.json"),
                ],
                stdout=log,
                stderr=log,
                env=env,
                timeout=budget.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "worker_timeout"
    return None if p.returncode == 0 else "worker_exit_" + str(p.returncode)


def run_suite(value, output):
    spec = Suite.model_validate(value)
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise ValueError("benchmark output must not exist")
    root.mkdir(parents=True)
    frozen = spec.model_dump(mode="json", by_alias=True)
    atomic_write(root / "spec.json", canonical_json(frozen))
    rows = []
    for number, c in enumerate(spec.cases):
        case = c.model_dump(mode="json")
        directory = root / f"{number:03d}"
        directory.mkdir()
        t = time.perf_counter()
        error = _launch(
            {
                "case": case,
                "policy": "prepare",
                "budget": frozen["budget"],
                "directory": str(directory),
            },
            directory,
            spec.budget,
        )
        forecast_sec = time.perf_counter() - t
        view = (
            None
            if error
            else read_json(
                directory / "forecast.json", limit=spec.budget.max_result_bytes
            )
        )
        for policy in POLICIES:
            path = directory / policy
            path.mkdir()
            failure = error
            total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if (
                total_bytes + spec.budget.max_result_bytes
                > spec.max_total_artifact_bytes
            ):
                failure = "global_artifact_budget_exhausted"
            if not failure:
                failure = _launch(
                    {
                        "case": case,
                        "policy": policy,
                        "budget": frozen["budget"],
                        "directory": str(path),
                        "forecast": view,
                    },
                    path,
                    spec.budget,
                )
            if failure:
                row = {
                    "case": c.id,
                    "policy": policy,
                    "error": failure,
                    "native_replay_verified": False,
                    "total": len(
                        case["scenario"]["definition"]["lifecycle"]["requests"]
                    ),
                }
            else:
                row = read_json(
                    path / "result.json", limit=spec.budget.max_result_bytes
                )
            row.update(
                split=c.split,
                group=c.group,
                family=c.family,
                shared_forecast_sec=forecast_sec,
            )
            rows.append(row)
            atomic_write(
                root / "progress.json",
                canonical_json(
                    {
                        "finished": len(rows),
                        "expected": len(spec.cases) * len(POLICIES),
                        "rows": rows,
                    }
                ),
            )
    inventory = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            inventory[p.relative_to(root).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": file_digest(p),
            }
    body = {
        "schema": "space-env.release-benchmark/1",
        "spec_sha256": digest(frozen),
        "rows": rows,
        "teacher": teacher_report(frozen, rows),
        "measured_envelope": summarize_envelope(frozen, rows),
        "files": inventory,
        "policies": list(POLICIES),
        "information_contract": "one shared forecast per case; identical maximum search nodes, native resources, pointing capability and timeout; actual consumption reported",
        "expert_teacher_qualified": False,
    }
    body["sha256"] = digest(body)
    atomic_write(root / "report.json", canonical_json(body))
    return body


def verify_suite(directory, *, native=False):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("real benchmark directory required")
    report = read_json(root / "report.json")
    spec = Suite.model_validate(read_json(root / "spec.json"))
    if (
        not _sealed(report)
        or report.get("schema") != "space-env.release-benchmark/1"
        or report["spec_sha256"] != digest(spec.model_dump(mode="json", by_alias=True))
    ):
        raise ValueError("benchmark report/spec binding mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in root.rglob("*")) or actual != set(
        report["files"]
    ) | {"report.json"}:
        raise ValueError("benchmark inventory mismatch")
    for name, meta in report["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or (root / name).stat().st_size != meta["bytes"]
            or file_digest(root / name) != meta["sha256"]
        ):
            raise ValueError("benchmark file checksum/path mismatch")
    for row in report["rows"]:
        c = next((c for c in spec.cases if c.id == row["case"]), None)
        if c is None or any(
            row[k] != getattr(c, k) for k in ("split", "group", "family")
        ):
            raise ValueError("benchmark row case/split identity mismatch")
    for i, c in enumerate(spec.cases):
        directory = root / f"{i:03d}"
        forecast_path = directory / "forecast.json"
        if not forecast_path.exists():
            if any(r["case"] == c.id and not r.get("error") for r in report["rows"]):
                raise ValueError("successful policy missing shared forecast")
            continue
        view = read_json(forecast_path)
        if native:
            fresh = collect_chain_forecast(
                MissionRuntime(c.scenario, access=AccessPolicy(forecast=True)),
                c.scenario,
                c.task,
            )
            if fresh != view:
                raise ValueError("shared native forecast differs")
        for p in POLICIES:
            row = next(
                r for r in report["rows"] if r["case"] == c.id and r["policy"] == p
            )
            if row.get("error"):
                continue
            selection = select_schedule(view, p, spec.budget)
            if (
                selection != read_json(directory / p / "selection.json")
                or row["forecast_sha256"] != view["sha256"]
            ):
                raise ValueError("policy selection/shared information mismatch")
            saved_result = read_json(directory / p / "result.json")
            if {
                k: v
                for k, v in row.items()
                if k not in {"split", "group", "family", "shared_forecast_sec"}
            } != saved_result:
                raise ValueError("benchmark result/row mismatch")
            pointings = read_json(directory / p / "pointings.json")
            task = materialize_task(c.task, view, selection["schedule"], pointings)
            expected_plan = make_mission_plan(
                c.scenario, task, selection["schedule"]["steps"], goals=c.goals
            )
            if expected_plan != read_json(directory / p / "dataset/plan.json"):
                raise ValueError("policy selection/physical plan mismatch")
            if native:
                runtime = MissionRuntime(c.scenario, access=AccessPolicy(forecast=True))
                chains = {x["id"]: x for x in view["chains"]}
                actual_pointings = {
                    a["request"]: runtime.forecast_pointing(
                        a["actor"], chains[a["request"]]["target"], a["pointing_at_sec"]
                    )
                    for a in selection["schedule"]["allocations"]
                }
                if pointings != actual_pointings:
                    raise ValueError("native pointing mismatch")
            audit = verify_plan_bundle(directory / p / "dataset", native=native)
            cert = read_json(directory / p / "dataset/certificate.json")
            execution = read_json(directory / p / "dataset/execution.json")
            metrics = native_metrics(
                c.model_dump(mode="json"),
                execution["episode"]["runtime"]["backend_report"],
                cert,
            )
            if (
                metrics != row["metrics"]
                or audit["mission_goal_passed"] != row["metrics"]["mission_goal_passed"]
            ):
                raise ValueError("native score mismatch")
    if summarize_envelope(spec, report["rows"]) != report["measured_envelope"]:
        raise ValueError("measured envelope differs")
    if teacher_report(spec, report["rows"]) != report["teacher"]:
        raise ValueError("teacher gates differ")
    return {
        "integrity_passed": True,
        "native_replayed": native,
        "paired_rows": len(report["rows"]),
        "teacher_gates_passed": report["teacher"]["passed_declared_gates"],
        "expert_teacher_qualified": False,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker", "freeze", "run", "verify"])
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        _worker(**read_json(args.input))
        return
    if args.command == "freeze":
        from space_env.planning.release_cases import release_suite

        if Path(args.input).exists():
            raise ValueError("frozen input must not exist")
        value = Suite.model_validate(release_suite(smoke=args.smoke)).model_dump(
            mode="json", by_alias=True
        )
        atomic_write(args.input, canonical_json(value))
        return
    if args.command == "run":
        value = run_suite(read_json(args.input), args.output)
    else:
        value = verify_suite(args.input, native=args.native)
    print(canonical_json(value))


if __name__ == "__main__":
    main()
