"""Frozen native operations regression tasks, executed through actor-only tools."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, seal, sources, verify_record
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.session import TaskSession, score

FIXTURE_SHA256 = "e1ec2af29e9b9e70f85ec8f0ab959795cfc95fae76ee85bf85595271b45b8d75"
POLICIES = ("wait", "reference_schedule")
TASKS = {
    "resource-conflict": "Acquire and deliver all requests while sharing camera, CPU, radio and memory bandwidth. Native qualified Ground acknowledgements determine fulfillment.",
    "comms-recovery": "Deliver both requests despite the declared interrupted transfer. Use actor cancellation and an alternate custody path when needed. Native Ground acknowledgements determine fulfillment.",
    "momentum-unloading": "Unload actor a wheel momentum toward zero using the declared paid RCS command within three seconds. Native unloading status, remaining momentum, propellant and energy are evaluated; request delivery is a separate metric.",
}


def compile_suite():
    """Load immutable packaged inputs; no example imports or feasibility search."""
    fixture = files("space_env").joinpath("bundled_data/llm_operations_v1.json")
    suite = json.loads(fixture.read_text(encoding="utf-8"))
    if (
        suite.get("sha256") != FIXTURE_SHA256
        or digest({k: v for k, v in suite.items() if k != "sha256"}) != FIXTURE_SHA256
    ):
        raise ValueError("Frozen operations fixture digest mismatch")
    return suite


def validate_suite(suite):
    if canonical_json(suite) != canonical_json(compile_suite()):
        raise ValueError("Frozen operations suite mismatch")
    return suite


def native_evidence(session):
    report = session._runtime._backend.report()
    life = report.get("lifecycle") or {}
    # Evaluator-only ledgers never enter policy messages.
    return {
        "score": score(session._runtime),
        "time_sec": session._runtime.time_sec,
        "lifecycle": {
            k: life[k]
            for k in (
                "metrics",
                "requests",
                "resource_lanes",
                "unloading",
                "acknowledgements",
            )
            if k in life
        },
        "dtn": report.get("dtn"),
        "spacecraft_bus": report.get("spacecraft_bus"),
    }


def operational_goal(family, evidence):
    """Native predicates, independent of case labels or scripted action IDs."""
    if family == "momentum-unloading":
        operations = evidence["lifecycle"]["unloading"]["operations"]
        return (
            evidence["time_sec"] <= 3.0 + 1e-9
            and bool(operations)
            and all(
                op["status"] == "completed"
                and all(
                    abs(x - y) <= op["tolerance_nms"]
                    for x, y in zip(
                        op["remaining_momentum_nms"], op["target_momentum_nms"]
                    )
                )
                for op in operations.values()
            )
        )
    metrics = evidence["lifecycle"]["metrics"]
    return (
        bool(metrics["total_requests"])
        and metrics["fulfilled_requests"] == metrics["total_requests"]
    )


def export_qualification(record, family):
    """Operation accomplishment does not confer delivery SFT qualification."""
    return {
        "policy_origin": record["configuration"]["policy_origin"],
        "delivery_sft_eligible": record["metrics"]["sft_eligible"],
        "operation_expert_qualified": False,
        "training_exported": False,
        "split": record["configuration"]["split"],
        "operation_goal_is_delivery_goal": family != "momentum-unloading",
    }


def run_suite(output, *, suite=None):
    suite = compile_suite() if suite is None else validate_suite(suite)
    start_sources = sources()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "suite.json", canonical_json(suite))
    rows = []
    for case in suite["cases"]:
        for policy in POLICIES:
            row = {"case": case["id"], "family": case["family"], "policy": policy}
            try:
                session = TaskSession(case["scenario"], case["task"])
                episode = LLMEpisode(
                    session,
                    task_text=TASKS[case["family"]],
                    group=case["id"],
                    split="test",
                    max_calls=64,
                    policy_origin="scripted_privileged"
                    if policy == "reference_schedule"
                    else "scripted_public",
                )
                roles = {a.id: a.role for a in session._task.actions}
                # Both policies get the same physical time partition and initial state.
                steps = (
                    case["steps"]
                    if policy == "reference_schedule"
                    else [dict(s, action_ids=[]) for s in case["steps"]]
                )
                record = run_agent(episode, reference_policy(steps, roles))
                verify_record(record)
                evidence = native_evidence(episode._session)
                artifact = case["id"] + "--" + policy + ".json"
                atomic_write(output / artifact, canonical_json(record))
                row.update(
                    status="completed",
                    artifact=artifact,
                    episode_sha256=record["sha256"],
                    metrics=record["metrics"],
                    native=evidence,
                    operational_goal_met=operational_goal(case["family"], evidence),
                    export_qualification=export_qualification(record, case["family"]),
                )
            except Exception as exc:  # noqa: BLE001 -- retain every attempted cell
                row.update(
                    status="error",
                    metrics=None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            rows.append(row)
            atomic_write(output / "progress.json", canonical_json(seal({"rows": rows})))
    if sources() != start_sources:
        raise ValueError("Sources changed during suite; rerun after writers finish")
    report = seal(
        {
            "schema": "space-env.llm-operations-report/1",
            "suite_sha256": suite["sha256"],
            "model_sources": start_sources,
            "rows": rows,
            "external_llm_measured": False,
        }
    )
    atomic_write(output / "report.json", canonical_json(report))
    return report


def audit(directory):
    directory = Path(directory)
    suite = validate_suite(json.loads((directory / "suite.json").read_text()))
    report = json.loads((directory / "report.json").read_text())
    if (
        report != seal({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
        or report["suite_sha256"] != suite["sha256"]
    ):
        raise ValueError("Report seal/source mismatch")
    expected = [(c["id"], p) for c in suite["cases"] for p in POLICIES]
    if [(r["case"], r["policy"]) for r in report["rows"]] != expected:
        raise ValueError("Incomplete comparison matrix")
    for case, row in zip([c for c in suite["cases"] for _ in POLICIES], report["rows"]):
        if row["family"] != case["family"] or row["status"] not in (
            "completed",
            "error",
        ):
            raise ValueError("Result identity mismatch")
        if row["status"] == "error":
            if row["metrics"] is not None:
                raise ValueError("Error cannot claim metrics")
            continue
        name = row["case"] + "--" + row["policy"] + ".json"
        if row["artifact"] != name:
            raise ValueError("Artifact mismatch")
        record = json.loads((directory / name).read_text())
        verify_record(record)
        replay = LLMEpisode(
            TaskSession(case["scenario"], case["task"]), **record["configuration"]
        )
        roles = {a.id: a.role for a in replay._spec.actions}
        steps = (
            case["steps"]
            if row["policy"] == "reference_schedule"
            else [dict(s, action_ids=[]) for s in case["steps"]]
        )
        expected_record = run_agent(replay, reference_policy(steps, roles))
        origin = (
            "scripted_privileged"
            if row["policy"] == "reference_schedule"
            else "scripted_public"
        )
        if (
            record != expected_record
            or record["configuration"]
            != {
                "task_text": TASKS[case["family"]],
                "group": case["id"],
                "split": "test",
                "max_calls": 64,
                "max_dialogue_bytes": 16000000,
                "policy_origin": origin,
            }
            or row["episode_sha256"] != record["sha256"]
            or row["metrics"] != record["metrics"]
            or row["export_qualification"]
            != export_qualification(record, case["family"])
            or row["native"] != native_evidence(replay._session)
            or row["operational_goal_met"]
            != operational_goal(case["family"], row["native"])
        ):
            raise ValueError("Native result binding mismatch")
    return {
        "valid": True,
        "rows": len(expected),
        "replayed": sum(r["status"] == "completed" for r in report["rows"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    group.add_argument("--compile", type=Path)
    args = parser.parse_args()
    if args.compile:
        atomic_write(args.compile, canonical_json(compile_suite()))
    elif args.audit:
        print(canonical_json(audit(args.audit)))
    else:
        report = run_suite(args.output)
        print(
            canonical_json(
                {
                    "sha256": report["sha256"],
                    "rows": [
                        {k: r[k] for k in ("case", "policy", "status")}
                        for r in report["rows"]
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
