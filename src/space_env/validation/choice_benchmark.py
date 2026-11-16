"""Bounded native choice benchmark. Scripted public planners, not LLM scores."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from space_env.llm.episode import seal, sources
from space_env.planning import PlanningRun
from space_env.planning.model import PolicyConfig
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.choice_missions import choice_mission, choice_suite
from space_env.tasks.session import score

POLICIES = ("wait", "priority", "earliest_deadline", "route_aware_rolling")


def _write(path, value):
    atomic_write(path, canonical_json(value))


def _read(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes benchmark root")
    return json.loads(path.read_text())


def policy_config(name):
    if name not in POLICIES:
        raise ValueError("unknown choice baseline")
    return PolicyConfig(
        name=name, max_candidates=64, max_nodes=1024, reclaim_acked_downlinks=True
    ).model_dump(mode="json")


def witness(run):
    report = run.session._runtime._backend.report()
    life = report["lifecycle"]
    outcome = score(run.session._runtime)
    return {
        "scope": "trainer-only outcomes; never provided as policy inputs",
        "fulfilled_priority": outcome["priority"],
        "requests": copy.deepcopy(life["requests"]),
        "energy_ledger": report["energy_ledger"],
        "native_rejection_codes": [
            t["feedback"]["code"] for t in run.trace if not t["feedback"]["accepted"]
        ],
        "assignments": {rid: row.get("actor") for rid, row in life["requests"].items()},
    }


def _worker(source, output):
    job = json.loads(Path(source).read_text())
    if job["sources"] != sources() or job["case"] != choice_mission(
        job["case"]["configuration"]
    ):
        raise ValueError("worker recipe/source mismatch")
    run = PlanningRun(
        job["case"]["scenario"], job["case"]["task"], policy_config(job["policy"])
    )
    destination = Path(output)
    while not run.done:
        run.step()
        _write(
            destination / "progress.json",
            {"decisions": len(run.trace), "time_sec": run.session._runtime.time_sec},
        )
        if len(run.trace) % 5 == 0:
            _write(destination / "partial.json", run.checkpoint(compact=True))
    record = run.checkpoint(compact=True)
    _write(destination / "episode.json", record)
    if job["sources"] != sources():
        raise ValueError("source changed during choice episode")
    _write(
        destination / "result.json",
        seal(
            {
                "record_sha256": record["sha256"],
                "metrics": run.metrics(),
                "witness": witness(run),
            }
        ),
    )


def manifest(quick=False, workers=2):
    if type(quick) is not bool:
        raise TypeError("quick must be boolean")
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("workers must be 1, 2 or 4")
    cases = choice_suite()["cases"]
    if quick:
        cases = [c for c in cases if c["configuration"]["catalogue_seed"] == 0]
    return {
        "schema": "space-env.choice-manifest/1",
        "quick": quick,
        "workers": workers,
        "cases": cases,
        "policies": list(POLICIES),
        "timeout_sec": 180,
        "scope": "public close-formation choice regression; label permutations are not independent geometries; no privileged reference, future rollout or LLM inference",
    }


def summarize(frozen, rows):
    expected = [(c["id"], p) for c in frozen["cases"] for p in POLICIES]
    if [(r["case"], r["policy"]) for r in rows] != expected:
        raise ValueError("choice comparison matrix missing, duplicate or reordered")
    if any(r["status"] not in {"completed", "error"} for r in rows):
        raise ValueError("unknown comparison outcome")
    totals = {}
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        totals[policy] = {
            "attempts": len(selected),
            "errors": sum(r["status"] == "error" for r in selected),
            "fulfilled_lower": sum(
                r.get("metrics", {}).get("fulfilled", 0) for r in selected
            ),
            "priority_lower": sum(
                r.get("witness", {}).get("fulfilled_priority", 0) for r in selected
            ),
        }
    best_observed = {}
    for case in frozen["cases"]:
        good = [
            r for r in rows if r["case"] == case["id"] and r["status"] == "completed"
        ]
        best_observed[case["id"]] = max(
            (r["witness"]["fulfilled_priority"] for r in good), default=None
        )
    return {
        "policies": totals,
        "best_observed_priority": best_observed,
        "scope": "lower-bound aggregates retain errors as zero observed delivery, not proof of zero achievable value; best observed is not an optimum; label variants are correlated",
    }


def run_suite(output, *, quick=False, workers=2):
    root = Path(output)
    if root.exists():
        raise ValueError("output must be new")
    frozen, before = manifest(quick, workers), sources()
    root.mkdir(parents=True)
    _write(root / "manifest.json", seal(frozen))

    def execute(job):
        case, policy = job
        relative = case["id"] + "--" + policy
        directory = root / relative
        directory.mkdir()
        job = {"case": case, "policy": policy, "sources": before}
        _write(directory / "input.json", job)
        row = {"case": case["id"], "policy": policy}
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.update(
            {
                k: "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            }
        )
        try:
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "space_env.validation.choice_benchmark",
                    "--worker",
                    str((directory / "input.json").resolve()),
                    "--worker-output",
                    str(directory.resolve()),
                ],
                env=environment,
                capture_output=True,
                timeout=frozen["timeout_sec"],
                check=False,
            )
            if child.returncode:
                raise RuntimeError(child.stderr[-2000:].decode(errors="replace"))
            result = _read(directory, "result.json")
            if result != seal({k: v for k, v in result.items() if k != "sha256"}):
                raise ValueError("worker result seal mismatch")
            row.update(
                status="completed",
                artifact=relative + "/episode.json",
                record_sha256=result["record_sha256"],
                metrics=result["metrics"],
                witness=result["witness"],
            )
        except Exception as exc:  # noqa: BLE001 -- retain every fixed comparison cell
            row.update(
                status="error", error_type=type(exc).__name__, error=str(exc)[:2000]
            )
        row["wall_sec"] = time.perf_counter() - started
        return row

    rows = []
    jobs = [(case, policy) for case in frozen["cases"] for policy in POLICIES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(execute, jobs):
            rows.append(row)
            _write(root / "progress.json", {"expected": len(jobs), "rows": rows})
    if before != sources():
        raise ValueError("sources changed during choice benchmark")
    report = seal(
        {
            "schema": "space-env.choice-result/1",
            "manifest_sha256": digest(frozen),
            "sources": before,
            "rows": rows,
            "summary": summarize(frozen, rows),
            "llm_measured": False,
            "optimality_proven": False,
            "scope": frozen["scope"],
        }
    )
    _write(root / "report.json", report)
    return report


def _verify_file(path):
    record = json.loads(Path(path).read_text())
    restored = PlanningRun.from_checkpoint(record)
    metrics = restored.metrics()
    for field in ("decision_wall_sec", "process_peak_rss_bytes"):
        metrics.pop(field, None)
    return {
        "record_sha256": record["sha256"],
        "scenario_sha256": digest(record["episode"]["scenario"]),
        "task_sha256": digest(record["episode"]["task"]),
        "policy": record["policy"]["config"],
        "operator_enabled": record["operator_enabled"],
        "metrics": metrics,
        "witness": witness(restored),
    }


def audit(directory):
    root = Path(directory)
    report = _read(root, "report.json")
    frozen = _read(root, "manifest.json")
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    if (
        frozen != seal(body)
        or body != manifest(body["quick"], body["workers"])
        or report["manifest_sha256"] != digest(body)
    ):
        raise ValueError("choice manifest mismatch")
    if (
        report != seal({k: v for k, v in report.items() if k != "sha256"})
        or report["sources"] != sources()
    ):
        raise ValueError("choice report source/seal mismatch")
    if (
        report.get("schema") != "space-env.choice-result/1"
        or report.get("llm_measured") is not False
        or report.get("optimality_proven") is not False
    ):
        raise ValueError("invalid choice evidence claims")
    if report["summary"] != summarize(body, report["rows"]):
        raise ValueError("choice summary mismatch")
    cases = {c["id"]: c for c in body["cases"]}

    def verify(row):
        if row["status"] == "error":
            return 0
        record = _read(root, row["artifact"])
        environment = os.environ.copy()
        environment.update(
            {
                key: "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            }
        )
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "space_env.validation.choice_benchmark",
                "--verify",
                str((root / row["artifact"]).resolve()),
            ],
            env=environment,
            capture_output=True,
            timeout=body["timeout_sec"],
            check=False,
        )
        if child.returncode:
            raise ValueError(
                "choice replay process failed: "
                + child.stderr[-1000:].decode(errors="replace")
            )
        actual = json.loads(child.stdout)
        expected_task = copy.deepcopy(cases[row["case"]]["task"])
        expected_task["policy_origin"] = "scripted"
        metrics = {
            k: v
            for k, v in row["metrics"].items()
            if k not in ("decision_wall_sec", "process_peak_rss_bytes")
        }
        expected = {
            "record_sha256": row["record_sha256"],
            "scenario_sha256": digest(cases[row["case"]]["scenario"]),
            "task_sha256": digest(expected_task),
            "policy": policy_config(row["policy"]),
            "operator_enabled": False,
            "metrics": metrics,
            "witness": row["witness"],
        }
        if actual != expected or record["sha256"] != row["record_sha256"]:
            raise ValueError("choice native recipe/policy/outcome mismatch")
        return 1

    with ThreadPoolExecutor(max_workers=body["workers"]) as pool:
        checked = sum(pool.map(verify, report["rows"]))
    return {
        "valid": True,
        "native_replayed": checked,
        "rows": len(report["rows"]),
        "scope": "public policy decisions and native execution replayed; timing not resampled, no independent physics qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    group.add_argument("--verify", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    if args.verify:
        print(canonical_json(_verify_file(args.verify)))
    elif args.worker:
        if args.worker_output is None:
            parser.error("worker-output required")
        _worker(args.worker, args.worker_output)
    elif args.audit:
        print(canonical_json(audit(args.audit)))
    else:
        result = run_suite(args.output, quick=args.quick, workers=args.workers)
        print(
            canonical_json({"summary": result["summary"], "rows": len(result["rows"])})
        )


if __name__ == "__main__":
    main()
