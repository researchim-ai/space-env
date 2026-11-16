"""Frozen coupled-service/nonlinear-navigation/FDIR matrix, local execution only."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from space_env.industrial.agent import IndustrialEpisode
from space_env.industrial.native_proximity import (
    NativeProximitySession,
    native_proximity_policy,
    native_proximity_recipe,
)
from space_env.industrial.native_service import (
    NativeServiceSession,
    native_service_policy,
    native_service_recipe,
)
from space_env.industrial.operations import OperationsSession, operations_policy
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def suite():
    cases = []
    for profile in (
        "nominal",
        "outage",
        "receiver_failure",
        "cancel_inflight",
        "pointing",
        "low_memory",
        "prefix",
    ):
        cases.append(
            {
                "id": "native_service-" + profile,
                "kind": "native_service",
                "configuration": native_service_recipe(profile),
            }
        )
    for profile in ("nominal", "latency", "blind", "crossing", "retreat"):
        cases.append(
            {
                "id": "native_proximity-" + profile,
                "kind": "native_proximity",
                "configuration": native_proximity_recipe(profile),
            }
        )
    cases.append(
        {
            "id": "operations-permanent-fault",
            "kind": "operations",
            "configuration": {
                "events": [
                    {
                        "at_sec": 4.0,
                        "event": {
                            "type": "spacecraft_fault",
                            "payload": {"actor": "camera-00", "subsystem": "radio"},
                        },
                    }
                ],
                "permanent_faults": [["camera-00", "radio"]],
            },
        }
    )
    recovery = {
        "id": "operations-external-repair",
        "kind": "operations",
        "configuration": {
            "events": [
                {
                    "at_sec": 4.0,
                    "event": {
                        "type": "spacecraft_fault",
                        "payload": {"actor": "camera-00", "subsystem": "radio"},
                    },
                },
                {
                    "at_sec": 28.0,
                    "event": {
                        "type": "spacecraft_fault_cleared",
                        "payload": {"actor": "camera-00", "subsystem": "radio"},
                    },
                },
            ],
            "permanent_faults": [["camera-00", "radio"]],
        },
    }
    cases.append(recovery)
    return {
        "schema": "space-env.native-coupled-suite/1",
        "cases": cases,
        "policies": ["wait", "public"],
        "scope": "public synthetic regression variants, shared progenitors; no independent OOD, external engine ranking or flight calibration",
    }


def run_case(case, policy, *, split="test"):
    if policy not in {"wait", "public"}:
        raise ValueError("unknown native coupled policy")
    factories = {
        "native_service": NativeServiceSession,
        "native_proximity": NativeProximitySession,
        "operations": OperationsSession,
    }
    policies = {
        "native_service": native_service_policy,
        "native_proximity": native_proximity_policy,
        "operations": operations_policy,
    }
    session = factories[case["kind"]](case["configuration"])
    episode = IndustrialEpisode(
        session, split=split, policy_origin="scripted_public", max_calls=128
    )
    index = 0
    while not episode.done:
        obs = episode.observe()
        action = (
            policies[case["kind"]](obs)
            if policy == "public"
            else ({"kind": "coast"} if case["kind"] == "native_proximity" else {})
        )
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": obs["revision"],
                    "idempotency_key": str(index),
                    "action": action,
                }
            ),
        )
        index += 1
    return episode.record()


def _worker(job):
    case, policy, expected_sources = job
    if sources() != expected_sources:
        raise ValueError("worker source identity mismatch")
    try:
        return {
            "case": case["id"],
            "policy": policy,
            "status": "completed",
            "record": run_case(case, policy),
        }
    except Exception as exc:  # noqa: BLE001 -- retain every attempted matrix cell
        return {
            "case": case["id"],
            "policy": policy,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1024],
        }


def _write(path, value):
    atomic_write(path, canonical_json(value))


def run_study(output, *, workers=2):
    root = Path(output)
    if root.exists() or type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("choose a new output and one/two workers")
    root.mkdir(parents=True)
    manifest = {"suite": suite(), "sources": sources(), "workers": workers}
    _write(root / "manifest.json", manifest)
    jobs = [
        (c, p, manifest["sources"])
        for c in manifest["suite"]["cases"]
        for p in manifest["suite"]["policies"]
    ]
    rows = [None] * len(jobs)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(_worker, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            if result["status"] == "completed":
                record = result.pop("record")
                name = result["case"] + "--" + result["policy"] + ".json"
                _write(root / name, record)
                result.update(
                    record=name,
                    sha256=record["sha256"],
                    metrics=record["metrics"],
                    invalid_calls=sum(
                        "error" in e["response"] for e in record["events"]
                    ),
                )
            rows[index] = result
            _write(root / "progress.json", {"rows": rows})
    if manifest["sources"] != sources():
        raise ValueError("source changed during coupled study")
    body = {
        "schema": "space-env.native-coupled-study/1",
        "manifest_sha256": digest(manifest),
        "rows": rows,
        "training_measured": False,
        "external_llm_measured": False,
        "flight_qualified": False,
        "universal_superiority": False,
    }
    report = {**body, "sha256": digest(body)}
    _write(root / "report.json", report)
    return report


def _audit_worker(job):
    case, policy, record, expected_sources = job
    if sources() != expected_sources or canonical_json(
        run_case(case, policy)
    ) != canonical_json(record):
        raise ValueError("native policy and episode cold replay mismatch")
    return True


def audit(output):
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    if (
        canonical_json(manifest.get("suite")) != canonical_json(suite())
        or manifest.get("sources") != sources()
        or report.get("manifest_sha256") != digest(manifest)
        or report.get("sha256")
        != digest({k: v for k, v in report.items() if k != "sha256"})
        or any(
            report.get(k) is not False
            for k in (
                "training_measured",
                "external_llm_measured",
                "flight_qualified",
                "universal_superiority",
            )
        )
    ):
        raise ValueError("native study source, recipe, scope or seal mismatch")
    expected = [
        (c, p)
        for c in manifest["suite"]["cases"]
        for p in manifest["suite"]["policies"]
    ]
    if [(r["case"], r["policy"]) for r in report["rows"]] != [
        (c["id"], p) for c, p in expected
    ]:
        raise ValueError("native study matrix mismatch")
    jobs = []
    errors = 0
    for row, (case, policy) in zip(report["rows"], expected, strict=True):
        if row["status"] == "error":
            errors += 1
            continue
        name = case["id"] + "--" + policy + ".json"
        if row["status"] != "completed" or row["record"] != name:
            raise ValueError("invalid coupled result row")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("native record escapes study root")
        record = json.loads(path.read_text())
        if (
            row["sha256"] != record.get("sha256")
            or row["metrics"] != record.get("metrics")
            or row["invalid_calls"]
            != sum("error" in e["response"] for e in record["events"])
        ):
            raise ValueError("native result outcome mismatch")
        jobs.append((case, policy, record, manifest["sources"]))
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_audit_worker, jobs))
    return {
        "valid": True,
        "policy_and_episode_replayed": len(results),
        "errors_retained": errors,
        "flight_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = audit(args.audit) if args.audit else run_study(args.output)
    print(
        canonical_json(
            result
            if args.audit
            else {
                "attempts": len(result["rows"]),
                "errors": sum(r["status"] == "error" for r in result["rows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
