"""Frozen multi-family catalogue-execution tasks; no automatic LLM inference.

Families vary declared physical inputs, never filter on planner/agent success.
Previously public geometries are regression splits, not blind holdout or OOD.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from space_env.api.network_workspace import NetworkConfiguration
from space_env.llm.benchmark import POLICIES, public_policy, reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, seal, sources, verify_record
from space_env.planning.network_benchmark import geometry_key, network_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.workspace import MissionWorkspace

FAMILIES = (
    "relay-chain",
    "cpu-backlog",
    "memory-pressure",
    "station-blackout",
    "link-blackout",
    "deadline-pressure",
)
SCHEMA = "space-env.llm-mission-families/1"
TASK = "Deliver the declared observation/processing products to Ground before their deadlines. Maximize delivered priority. Only native Ground ACK completes a request."


def compile_suite():
    """36 literal workload variants, six public geometry groups, fixed ML splits."""
    cases = []
    for base in network_suite()["cases"]:
        if base["condition"] != "nominal":
            continue
        for family in FAMILIES:
            config = copy.deepcopy(base["configuration"])
            mission = config["mission"]
            mission["name"] = base["group"] + " / " + family
            if family == "relay-chain":
                config["links"] = config["links"][:2]
                config["max_hops"] = 2
            elif family == "cpu-backlog":
                for i, request in enumerate(mission["requests"]):
                    request["processing_sec"] = [35.0, 20.0, 45.0][i]
            elif family == "memory-pressure":
                mission["memory_mbit"] = 1.2
            elif family == "station-blackout":
                mission["outages"] = [
                    {"station": "Ground", "start_sec": 0.0, "end_sec": 120.0}
                ]
            elif family == "link-blackout":
                config["links"] = config["links"][:2]
                config["links"][0]["outages"] = [[0.0, 90.0]]
            elif family == "deadline-pressure":
                for i, request in enumerate(mission["requests"]):
                    request["deadline_sec"] = [30.0, 45.0, 60.0][i]
            normalized = NetworkConfiguration.model_validate(config).model_dump(
                mode="json", by_alias=True
            )
            cases.append(
                {
                    "id": base["group"] + "-" + family,
                    "group": base["group"],
                    "split": base["split"],
                    "family": family,
                    "geometry_sha256": geometry_key(normalized),
                    "configuration": normalized,
                }
            )
    return seal(
        {
            "schema": SCHEMA,
            "cases": cases,
            "scope": "fixed public regression geometries; six workload families, not blind holdout; operator-compiled catalogue execution",
        }
    )


def validate_suite(value):
    # Bind the published task set to this explicit version, including order and scope.
    if canonical_json(value) != canonical_json(compile_suite()):
        raise ValueError("Frozen family suite mismatch")
    geometries = {}
    for case in value["cases"]:
        key = case["geometry_sha256"]
        if key in geometries and geometries[key] != case["split"]:
            raise ValueError("Family geometry crosses split")
        geometries[key] = case["split"]
    return value


def summarize(suite, selected, rows):
    validate_suite(suite)
    cases = {c["id"]: c for c in suite["cases"]}
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= set(cases)
    ):
        raise ValueError("Invalid case selection")
    expected = {(c, p) for c in selected for p in POLICIES}
    if (
        len(rows) != len(expected)
        or {(r["case"], r["policy"]) for r in rows} != expected
    ):
        raise ValueError("Incomplete or duplicate family comparison matrix")
    for row in rows:
        case = cases[row["case"]]
        total = len(case["configuration"]["mission"]["requests"])
        priority = sum(
            r["priority"] for r in case["configuration"]["mission"]["requests"]
        )
        if (
            any(
                row[k] != case[k]
                for k in ("group", "split", "family", "geometry_sha256")
            )
            or row["priority_total"] != priority
            or row["request_total"] != total
        ):
            raise ValueError("Family result identity/denominator mismatch")
        if row["status"] not in {"completed", "error"}:
            raise ValueError("Unknown family result status")
        if row["status"] == "error":
            if row["metrics"] is not None:
                raise ValueError("Error result cannot claim mission metrics")
        else:
            m = row["metrics"]
            value = m["weighted_fulfilled"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= priority
                or type(m["fulfilled"]) is not int
                or not 0 <= m["fulfilled"] <= total
                or m["total"] != total
            ):
                raise ValueError("Invalid family result value")
    aggregates = []
    for family in FAMILIES:
        for split in ("train", "development", "test"):
            for policy in POLICIES:
                cells = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["split"] == split
                    and r["policy"] == policy
                ]
                if not cells:
                    continue
                completed = [r for r in cells if r["status"] == "completed"]
                lower = sum(
                    r["metrics"]["weighted_fulfilled"] / r["priority_total"]
                    for r in completed
                ) / len(cells)
                aggregates.append(
                    {
                        "family": family,
                        "split": split,
                        "policy": policy,
                        "cases": len(cells),
                        "geometries": len({r["geometry_sha256"] for r in cells}),
                        "completed": len(completed),
                        "successful": sum(r["metrics"]["success"] for r in completed),
                        "value_lower": lower,
                        "value_upper": lower
                        + (len(cells) - len(completed)) / len(cells),
                        "privileged": policy == "reference_schedule",
                    }
                )
    return {
        "aggregates": aggregates,
        "external_llm_measured": False,
        "competitor_superiority_established": False,
        "blind_holdout_claim": False,
    }


def run_suite(output, *, selected=None):
    suite = compile_suite()
    selected = (
        list(selected) if selected is not None else [c["id"] for c in suite["cases"]]
    )
    cases = {c["id"]: c for c in suite["cases"]}
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= set(cases)
    ):
        raise ValueError("Invalid case selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "suite.json", canonical_json(suite))
    rows = []
    for name in selected:
        case = cases[name]
        initial, meta, task, setup_error = None, None, None, None
        mission = None
        try:
            mission = MissionWorkspace.network(case["configuration"])
            initial = mission.operator.session
            meta = initial._scenario["metadata"]["network_workspace"]
            task = initial._task
        except Exception as exc:  # noqa: BLE001 -- preserve failed cells
            setup_error = type(exc).__name__
        for policy in POLICIES:
            row = {
                k: case[k]
                for k in ("id", "group", "split", "family", "geometry_sha256")
            }
            row["case"] = row.pop("id")
            row["policy"] = policy
            row["priority_total"] = sum(
                r["priority"] for r in case["configuration"]["mission"]["requests"]
            )
            row["request_total"] = len(case["configuration"]["mission"]["requests"])
            try:
                if setup_error:
                    raise ValueError("native setup failed: " + setup_error)
                agent = LLMEpisode(
                    initial,
                    task_text=TASK,
                    group=case["group"],
                    split=case["split"],
                    max_calls=64,
                    policy_origin="scripted_privileged"
                    if policy == "reference_schedule"
                    else "scripted_public",
                )
                choose = (
                    reference_policy(
                        meta["steps"], {a.id: a.role for a in task.actions}
                    )
                    if policy == "reference_schedule"
                    else public_policy(policy, task.decision_sec)
                )
                record = run_agent(agent, choose)
                verify_record(record)
                artifact = name + "--" + policy + ".json"
                atomic_write(output / artifact, canonical_json(record))
                row.update(
                    status="completed",
                    metrics=record["metrics"],
                    artifact=artifact,
                    episode_sha256=record["sha256"],
                    initial_checkpoint_sha256=record["initial_checkpoint"]["sha256"],
                )
            except Exception as exc:  # noqa: BLE001 -- no skipped failures or scripted fallback
                row.update(status="error", metrics=None, error_type=type(exc).__name__)
            rows.append(row)
            atomic_write(output / "progress.json", canonical_json(seal({"rows": rows})))
        if mission is not None:
            mission.close()
    report = seal(
        {
            "schema": "space-env.llm-family-report/1",
            "suite": suite,
            "selected": selected,
            "rows": rows,
            "model_sources": sources(),
            **summarize(suite, selected, rows),
        }
    )
    atomic_write(output / "report.json", canonical_json(report))
    return report


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if (
        report.get("schema") != "space-env.llm-family-report/1"
        or report.get("sha256")
        != digest({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
    ):
        raise ValueError("Family report seal/source mismatch")
    suite = validate_suite(report["suite"])
    if json.loads((directory / "suite.json").read_text()) != suite:
        raise ValueError("Published suite mismatch")
    summary = summarize(suite, report["selected"], report["rows"])
    cases = {c["id"]: c for c in suite["cases"]}
    checkpoints = {}
    for row in report["rows"]:
        if row["status"] == "error":
            continue
        name = row["case"] + "--" + row["policy"] + ".json"
        if row["artifact"] != name:
            raise ValueError("Unsafe episode artifact")
        record = json.loads((directory / name).read_text())
        verify_record(record)
        case = cases[row["case"]]
        config = record["initial_checkpoint"]["scenario"]["metadata"][
            "network_workspace"
        ]["configuration"]
        expected_origin = (
            "scripted_privileged"
            if row["policy"] == "reference_schedule"
            else "scripted_public"
        )
        if (
            config != case["configuration"]
            or record["configuration"]
            != {
                "task_text": TASK,
                "group": case["group"],
                "split": case["split"],
                "max_calls": 64,
                "max_dialogue_bytes": 16000000,
                "policy_origin": expected_origin,
            }
            or record["metrics"] != row["metrics"]
            or record["sha256"] != row["episode_sha256"]
            or record["initial_checkpoint"]["sha256"]
            != row["initial_checkpoint_sha256"]
        ):
            raise ValueError("Family episode/result binding mismatch")
        checkpoints.setdefault(row["case"], set()).add(row["initial_checkpoint_sha256"])
    if any(len(values) != 1 for values in checkpoints.values()) or any(
        report[k] != v for k, v in summary.items()
    ):
        raise ValueError("Family comparison checkpoint/summary mismatch")
    return {
        "valid": True,
        "rows": len(report["rows"]),
        "replayed": sum(r["status"] == "completed" for r in report["rows"]),
        "external_llm_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    group.add_argument("--compile", type=Path)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.compile:
        atomic_write(args.compile, canonical_json(compile_suite()))
    elif args.audit:
        print(canonical_json(audit(args.audit)))
    else:
        report = run_suite(args.output, selected=args.case)
        print(
            canonical_json(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"suite", "model_sources", "rows"}
                }
            )
        )


if __name__ == "__main__":
    main()
