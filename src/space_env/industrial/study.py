"""Frozen operational/proximity/service matrix with retained failures and replay.

Run locally; no CI, paid APIs, external model inference or superiority claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from space_env.industrial import OperationsSession, ProximitySession, ServiceSession
from space_env.industrial.agent import IndustrialEpisode, verify_episode
from space_env.industrial.operations import operations_policy
from space_env.industrial.proximity import proximity_policy
from space_env.industrial.service import service_policy
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def suite():
    cases = []

    def add(kind, name, configuration):
        cases.append(
            {"id": kind + "-" + name, "kind": kind, "configuration": configuration}
        )

    def fault(subsystem="radio", actor="camera-00", at=4.0):
        return {
            "at_sec": at,
            "event": {
                "type": "spacecraft_fault",
                "payload": {"actor": actor, "subsystem": subsystem},
            },
        }

    add("operations", "new-demand", {"deferred_request_at_sec": 12.0})
    add("operations", "resettable-radio", {"events": [fault()]})
    add(
        "operations",
        "permanent-radio",
        {"events": [fault()], "permanent_faults": [["camera-00", "radio"]]},
    )
    add(
        "operations",
        "cpu-and-new-demand",
        {"events": [fault("cpu", at=8.0)], "deferred_request_at_sec": 16.0},
    )
    add(
        "operations",
        "ground-outage",
        {
            "events": [
                {
                    "at_sec": 8.0,
                    "event": {
                        "type": "station_outage_added",
                        "payload": {
                            "station": "Ground",
                            "start_sec": 8.0,
                            "end_sec": 28.0,
                        },
                    },
                }
            ]
        },
    )
    add("proximity", "inspection", {})
    add("proximity", "hold", {"mode": "hold"})
    add(
        "proximity",
        "delayed-measurements",
        {"measurement_latency_steps": 2, "dropout_probability": 0.2, "seed": 7},
    )
    add("proximity", "blind", {"dropout_probability": 1.0})
    add(
        "proximity",
        "retreat",
        {
            "mode": "retreat",
            "initial_truth_si": [145.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            "initial_estimate_si": [145.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        },
    )
    add(
        "proximity",
        "crossing",
        {
            "initial_truth_si": [100.0, 0.0, 0.0, -10.0, 0.0, 0.0],
            "initial_estimate_si": [100.0, 0.0, 0.0, -10.0, 0.0, 0.0],
            "max_impulse_m_s": 3.0,
            "delta_v_budget_m_s": 6.0,
        },
    )
    base = {
        "nodes": [
            {"id": "a"},
            {"id": "b"},
            {"id": "reserve", "reserve": True},
            {"id": "ground", "kind": "ground"},
        ],
        "links": [
            {"id": "a-b", "source": "a", "destination": "b", "bits_per_slot": 100},
            {"id": "b-g", "source": "b", "destination": "ground", "bits_per_slot": 100},
            {
                "id": "a-r",
                "source": "a",
                "destination": "reserve",
                "bits_per_slot": 100,
            },
            {
                "id": "r-g",
                "source": "reserve",
                "destination": "ground",
                "bits_per_slot": 100,
            },
        ],
        "flows": [
            {
                "id": "routine",
                "source": "a",
                "destination": "ground",
                "size_bits": 200,
                "deadline_step": 10,
                "priority": 2,
            },
            {
                "id": "urgent",
                "source": "a",
                "destination": "ground",
                "size_bits": 100,
                "release_step": 3,
                "deadline_step": 8,
                "priority": 9,
            },
        ],
        "horizon_steps": 12,
    }
    add("service", "nominal", copy.deepcopy(base))
    outage = copy.deepcopy(base)
    outage["events"] = [{"at_step": 2, "kind": "node_fault", "target": "b"}]
    add("service", "reserve-reroute", outage)
    recovery = copy.deepcopy(outage)
    recovery["events"].append({"at_step": 6, "kind": "node_recovered", "target": "b"})
    add("service", "custodian-recovery", recovery)
    congestion = copy.deepcopy(base)
    congestion["flows"][0].update(size_bits=1000, deadline_step=6)
    add("service", "congestion", congestion)
    cancelled = copy.deepcopy(base)
    cancelled["events"] = [
        {"at_step": 3, "kind": "flow_cancelled", "target": "routine"}
    ]
    add("service", "cancelled", cancelled)
    overflow = copy.deepcopy(base)
    overflow["nodes"][0]["queue_bits"] = 150
    add("service", "capacity-rejection", overflow)
    return {
        "schema": "space-env.industrial-suite/1",
        "cases": cases,
        "policies": ["wait", "public"],
        "scope": "public synthetic regression families; no independently held-out industrial population, model training or flight qualification",
    }


def run_case(case, policy, *, split="test"):
    factories = {
        "operations": OperationsSession,
        "proximity": ProximitySession,
        "service": ServiceSession,
    }
    policies = {
        "operations": operations_policy,
        "proximity": proximity_policy,
        "service": service_policy,
    }
    if policy not in {"wait", "public"}:
        raise ValueError("unknown study policy")
    session = factories[case["kind"]](case["configuration"])
    episode = IndustrialEpisode(session, split=split, policy_origin="scripted_public")
    index = 0
    while not episode.done:
        observation = episode.observe()
        action = (
            policies[case["kind"]](observation)
            if policy == "public"
            else ({"kind": "coast"} if case["kind"] == "proximity" else {})
        )
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": observation["revision"],
                    "idempotency_key": str(index),
                    "action": action,
                }
            ),
        )
        index += 1
    return episode.record()


def _write(path, body):
    atomic_write(path, canonical_json(body))


def run_study(output):
    output = Path(output)
    if output.exists():
        raise ValueError("study requires a new output directory")
    output.mkdir(parents=True)
    manifest = {"suite": suite(), "sources": sources()}
    _write(output / "manifest.json", manifest)
    rows = []
    for case in manifest["suite"]["cases"]:
        for policy in manifest["suite"]["policies"]:
            name = case["id"] + "--" + policy + ".json"
            row = {"case": case["id"], "policy": policy}
            try:
                record = run_case(case, policy)
                _write(output / name, record)
                row.update(
                    status="completed",
                    record=name,
                    sha256=record["sha256"],
                    metrics=record["metrics"],
                    calls=len(record["events"]),
                    invalid_calls=sum(
                        "error" in e["response"] for e in record["events"]
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- every failed matrix attempt is evidence
                # Retain failed attempts in the original declared denominator.
                row.update(
                    status="error", error_type=type(exc).__name__, error=str(exc)[:1024]
                )
            rows.append(row)
            _write(output / "progress.json", {"rows": rows})
    if manifest["sources"] != sources():
        raise ValueError("source changed during study")
    body = {
        "schema": "space-env.industrial-study/1",
        "manifest_sha256": digest(manifest),
        "rows": rows,
        "external_llm_measured": False,
        "training_measured": False,
        "flight_qualification": False,
        "universal_superiority": False,
    }
    report = {**body, "sha256": digest(body)}
    _write(output / "report.json", report)
    return report


def audit(output):
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    if (
        manifest != {"suite": suite(), "sources": sources()}
        or report.get("manifest_sha256") != digest(manifest)
        or report.get("sha256")
        != digest({k: v for k, v in report.items() if k != "sha256"})
        or any(
            report.get(k) is not False
            for k in (
                "external_llm_measured",
                "training_measured",
                "flight_qualification",
                "universal_superiority",
            )
        )
    ):
        raise ValueError("study identity, scope or integrity mismatch")
    expected = [
        (c["id"], p)
        for c in manifest["suite"]["cases"]
        for p in manifest["suite"]["policies"]
    ]
    if [(r["case"], r["policy"]) for r in report["rows"]] != expected:
        raise ValueError("study matrix mismatch")
    cases = {c["id"]: c for c in manifest["suite"]["cases"]}
    replayed, errors = 0, 0
    for row in report["rows"]:
        if row["status"] == "error":
            errors += 1
            continue
        if (
            row["status"] != "completed"
            or row["record"] != row["case"] + "--" + row["policy"] + ".json"
        ):
            raise ValueError("invalid study result row")
        path = (root / row["record"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("record escapes study root")
        record = json.loads(path.read_text())
        case = cases[row["case"]]
        factory = {
            "operations": OperationsSession,
            "proximity": ProximitySession,
            "service": ServiceSession,
        }[case["kind"]]
        if (
            record["initial"] != factory(case["configuration"]).checkpoint()
            or record["metrics"] != row["metrics"]
            or record["sha256"] != row["sha256"]
            or record["split"] != "test"
            or record["policy_origin"] != "scripted_public"
            or len(record["events"]) != row["calls"]
            or sum("error" in e["response"] for e in record["events"])
            != row["invalid_calls"]
        ):
            raise ValueError("study recipe or outcome mismatch")
        verify_episode(record)
        # Recompute the policy as well as native action replay: a forged policy
        # label must not silently become a benchmark result.
        if canonical_json(run_case(case, row["policy"])) != canonical_json(record):
            raise ValueError("study policy replay mismatch")
        replayed += 1
    return {
        "valid": True,
        "replayed": replayed,
        "errors_retained": errors,
        "flight_qualification": False,
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
                "rows": len(result["rows"]),
                "errors": sum(r["status"] == "error" for r in result["rows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
