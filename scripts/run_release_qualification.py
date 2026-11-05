"""Run frozen geometry groups independently; retain all shards and audit aggregation.

Invoke using the installed environment's Python -I. Concurrent timings describe
this shared host, not independent single-worker latency measurements.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from space_env.planning.mission_bundle import read_json
from space_env.planning.release_benchmark import Suite, teacher_report, verify_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest


def inventory(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("qualification symlinks forbidden")
    return {
        p.relative_to(root).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": file_digest(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "report.json"
    }


def execute(value, output, workers):
    spec = Suite.model_validate(value)
    root = Path(output)
    if root.exists():
        raise ValueError("qualification output must not exist")
    root.mkdir(parents=True)
    frozen = spec.model_dump(mode="json", by_alias=True)
    atomic_write(root / "spec.json", canonical_json(frozen))
    groups = list(dict.fromkeys(c.group for c in spec.cases))
    tasks = []
    for i, group in enumerate(groups):
        directory = root / "parts" / f"{i:03d}"
        directory.mkdir(parents=True)
        sub = dict(frozen)
        sub["id"] = f"{spec.id}-part-{i}"
        sub["cases"] = [c for c in frozen["cases"] if c["group"] == group]
        sub["max_total_artifact_bytes"] = max(
            1000000, spec.max_total_artifact_bytes // len(groups)
        )
        atomic_write(directory / "input.json", canonical_json(sub))
        tasks.append((i, group, directory))

    def job(task):
        i, group, directory = task
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (directory / "launcher.log").open("wb") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "space_env.planning.release_benchmark",
                    "run",
                    str(directory / "input.json"),
                    str(directory / "results"),
                ],
                stdout=log,
                stderr=log,
                env=env,
                check=False,
            )
        result = {
            "index": i,
            "group": group,
            "exit_code": proc.returncode,
            "audit": None,
        }
        if proc.returncode == 0:
            result["audit"] = verify_suite(directory / "results", native=False)
        atomic_write(directory / "status.json", canonical_json(result))
        return result

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(job, tasks):
            parts.append(part)
            atomic_write(
                root / "progress.json",
                canonical_json(
                    {
                        "parts_finished": len(parts),
                        "parts_expected": len(groups),
                        "parts": parts,
                    }
                ),
            )
    rows = []
    for part in parts:
        if part["exit_code"] == 0:
            rows += read_json(
                root / "parts" / f"{part['index']:03d}" / "results/report.json"
            )["rows"]
        else:
            for c in spec.cases:
                if c.group != part["group"]:
                    continue
                for policy in (
                    "priority_greedy",
                    "deadline_rule",
                    "bounded_order_search",
                ):
                    rows.append(
                        {
                            "case": c.id,
                            "policy": policy,
                            "error": "shard_exit_" + str(part["exit_code"]),
                            "total": len(
                                c.scenario["definition"]["lifecycle"]["requests"]
                            ),
                            "native_replay_verified": False,
                            "split": c.split,
                            "group": c.group,
                            "family": c.family,
                        }
                    )
    files = inventory(root)
    total = sum(v["bytes"] for v in files.values())
    body = {
        "schema": "space-env.release-qualification/1",
        "spec_sha256": digest(frozen),
        "parts": parts,
        "rows": rows,
        "teacher": teacher_report(spec, rows),
        "all_native_rows_verified": all(
            not r.get("error") and r.get("native_replay_verified") for r in rows
        ),
        "workers": workers,
        "artifact_bytes": total,
        "global_artifact_gate": total <= spec.max_total_artifact_bytes,
        "measurement_scope": "concurrent frozen geometry groups on a shared host; not independent performance repetitions",
        "expert_teacher_qualified": False,
        "files": files,
    }
    body["sha256"] = digest(body)
    atomic_write(root / "report.json", canonical_json(body))
    return body


def verify(output, native=False):
    root = Path(output)
    report = read_json(root / "report.json")
    spec = Suite.model_validate(read_json(root / "spec.json"))
    if report.get("schema") != "space-env.release-qualification/1":
        raise ValueError("invalid qualification schema")
    if report["sha256"] != digest(
        {k: v for k, v in report.items() if k != "sha256"}
    ) or report["files"] != inventory(root):
        raise ValueError("qualification report/inventory mismatch")
    if report["spec_sha256"] != digest(spec.model_dump(mode="json", by_alias=True)):
        raise ValueError("qualification spec differs")
    frozen = spec.model_dump(mode="json", by_alias=True)
    groups = list(dict.fromkeys(c.group for c in spec.cases))
    if len(report["parts"]) != len(groups):
        raise ValueError("qualification group count differs")
    if report["artifact_bytes"] != sum(m["bytes"] for m in report["files"].values()):
        raise ValueError("qualification artifact accounting differs")
    rows = []
    for i, part in enumerate(report["parts"]):
        if (
            type(part["index"]) is not int
            or part["index"] != i
            or part["group"] != groups[i]
        ):
            raise ValueError("qualification group identity differs")
        if part["exit_code"]:
            raise ValueError("qualification has failed shard; inspect retained logs")
        directory = root / "parts" / f"{part['index']:03d}" / "results"
        expected = dict(frozen)
        expected["id"] = f"{spec.id}-part-{i}"
        expected["cases"] = [c for c in frozen["cases"] if c["group"] == groups[i]]
        expected["max_total_artifact_bytes"] = max(
            1000000, spec.max_total_artifact_bytes // len(groups)
        )
        if (
            read_json(directory / "spec.json") != expected
            or read_json(directory.parent / "input.json") != expected
        ):
            raise ValueError("qualification shard does not match frozen parent cases")
        if read_json(directory.parent / "status.json") != part:
            raise ValueError("qualification status differs")
        audit = verify_suite(directory, native=native)
        if not audit["integrity_passed"]:
            raise ValueError("invalid shard audit")
        rows += read_json(directory / "report.json")["rows"]
    if rows != report["rows"] or teacher_report(spec, rows) != report["teacher"]:
        raise ValueError("qualification aggregation differs")
    if report["all_native_rows_verified"] is not all(
        not r.get("error") and r.get("native_replay_verified") for r in rows
    ):
        raise ValueError("native evidence gate differs")
    if report["global_artifact_gate"] is not (
        report["artifact_bytes"] <= spec.max_total_artifact_bytes
    ):
        raise ValueError("artifact gate differs")
    return {
        "integrity_passed": True,
        "native_audit_repeated": native,
        "policy_rows": len(rows),
        "all_native_rows_verified": report["all_native_rows_verified"],
        "teacher_gates_passed": report["teacher"]["passed_declared_gates"],
        "expert_teacher_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--artifact-gib", type=int, choices=[1, 4], default=1)
    parser.add_argument("--result-mb", type=int, choices=[64, 128], default=64)
    args = parser.parse_args()
    if args.command == "freeze":
        from space_env.planning.release_cases import release_suite

        value = release_suite()
        value["id"] += f"-{args.artifact_gib}g-{args.result_mb}m"
        value["budget"]["max_result_bytes"] = args.result_mb * 1000000
        value["max_total_artifact_bytes"] = args.artifact_gib * 1073741824
        value = Suite.model_validate(value).model_dump(mode="json", by_alias=True)
        if Path(args.input).exists():
            raise ValueError("frozen profile must not exist")
        atomic_write(args.input, canonical_json(value))
        print(
            canonical_json(
                {
                    "spec_sha256": digest(value),
                    "cases": len(value["cases"]),
                    "artifact_gib": args.artifact_gib,
                }
            )
        )
        return
    if args.command == "run":
        if not args.output:
            parser.error("run requires output")
        result = execute(read_json(args.input), args.output, args.workers)
        result = {
            k: result[k]
            for k in (
                "schema",
                "all_native_rows_verified",
                "teacher",
                "global_artifact_gate",
            )
        }
    else:
        result = verify(args.input, args.native)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
