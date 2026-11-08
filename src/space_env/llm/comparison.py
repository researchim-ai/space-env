"""Matched planner ablation on frozen mission families, no real LLM inference.

Compare serial, original exact_small, and finite-goal pointing_bounded. The
scripted reference executor is privileged in every cell, not an agent leaderboard.
"""

import argparse
import json
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, seal, sources, verify_record
from space_env.llm.mission_suite import TASK, compile_suite
from space_env.planning.pointing_envelope import diagnose_network
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.workspace import MissionWorkspace

METHODS = ("serial", "exact_small", "pointing_bounded")


def _diagnostic(meta):
    if meta["planning_method"] == "sequence_lanes":
        from space_env.planning.sequence_lanes import verify_sequence_schedule

        proof = verify_sequence_schedule(
            meta["forecast"], meta["sequence_geometry"], meta["schedule"]
        )
        return {
            "schema": "space-env.sequence-completeness/1",
            "admission": proof,
            "all_requests_catalogued": meta["schedule"]["predicted_fulfilled"]
            == meta["schedule"]["total"],
            "conditional_on_schedule": True,
        }
    return diagnose_network(meta["forecast"], meta["schedule"])


def run(output, selected, *, methods=METHODS):
    methods = list(methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods) <= {*METHODS, "sequence_lanes"}
    ):
        raise ValueError("Invalid comparison methods")
    suite = compile_suite()
    cases = {c["id"]: c for c in suite["cases"]}
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(cases)
    ):
        raise ValueError("Invalid pointing comparison selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in selected:
        case = cases[name]
        for method in methods:
            mission = None
            row = {
                "case": name,
                "method": method,
                "split": case["split"],
                "family": case["family"],
            }
            try:
                mission = MissionWorkspace.network(
                    case["configuration"], planning_method=method
                )
                session = mission.operator.session
                meta = session._scenario["metadata"]["network_workspace"]
                record = run_agent(
                    LLMEpisode(
                        session,
                        task_text=TASK,
                        group=case["group"],
                        split=case["split"],
                        max_calls=64,
                        policy_origin="scripted_privileged",
                    ),
                    reference_policy(
                        meta["steps"], {a.id: a.role for a in session._task.actions}
                    ),
                )
                verify_record(record)
                name_out = name + "--" + method + ".json"
                atomic_write(output / name_out, canonical_json(record))
                row.update(
                    status="completed",
                    artifact=name_out,
                    episode_sha256=record["sha256"],
                    metrics=record["metrics"],
                    planned=meta["schedule"]["predicted_fulfilled"],
                    optimal_in_forecast=meta["schedule"].get("optimal"),
                    search_budget_exhausted=meta["schedule"]["search_budget_exhausted"],
                    diagnostic=_diagnostic(meta),
                )
            except Exception as exc:  # noqa: BLE001 -- keep every comparison cell
                row.update(status="error", metrics=None, error_type=type(exc).__name__)
            finally:
                if mission is not None:
                    mission.close()
            rows.append(row)
            atomic_write(output / "progress.json", canonical_json(seal({"rows": rows})))
    report = seal(
        {
            "schema": "space-env.pointing-comparison/1",
            "suite": suite,
            "selected": selected,
            "methods": methods,
            "rows": rows,
            "model_sources": sources(),
            "scope": "matched physical configuration, three planners with respective catalogues and same privileged reference executor; no LLM, no blanket mission optimality; frozen regression families",
        }
    )
    atomic_write(output / "report.json", canonical_json(report))
    return report


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if (
        report.get("schema") != "space-env.pointing-comparison/1"
        or report.get("sha256")
        != digest({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
        or report["suite"] != compile_suite()
    ):
        raise ValueError("Pointing comparison seal/source/suite mismatch")
    cases = {c["id"]: c for c in report["suite"]["cases"]}
    selected = report["selected"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(cases)
    ):
        raise ValueError("Invalid selection")
    methods = report["methods"]
    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods) <= {*METHODS, "sequence_lanes"}
    ):
        raise ValueError("Invalid comparison methods")
    rows = report["rows"]
    if len(rows) != len(selected) * len(methods) or {
        (r["case"], r["method"]) for r in rows
    } != {(c, m) for c in selected for m in methods}:
        raise ValueError("Incomplete pointing comparison")
    for row in rows:
        case = cases[row["case"]]
        if (
            row["split"] != case["split"]
            or row["family"] != case["family"]
            or row["status"] not in {"completed", "error"}
        ):
            raise ValueError("Invalid comparison identity")
        if row["status"] == "error":
            if row["metrics"] is not None:
                raise ValueError("Error row cannot claim metrics")
            continue
        name = row["case"] + "--" + row["method"] + ".json"
        if row["artifact"] != name:
            raise ValueError("Unsafe artifact")
        record = json.loads((directory / name).read_text())
        verify_record(record)
        meta = record["initial_checkpoint"]["scenario"]["metadata"]["network_workspace"]
        expected_config = {
            "task_text": TASK,
            "group": case["group"],
            "split": case["split"],
            "max_calls": 64,
            "max_dialogue_bytes": 16000000,
            "policy_origin": "scripted_privileged",
        }
        if (
            meta["configuration"] != case["configuration"]
            or meta["planning_method"] != row["method"]
            or record["configuration"] != expected_config
            or record["sha256"] != row["episode_sha256"]
            or record["metrics"] != row["metrics"]
        ):
            raise ValueError("Native/configuration result mismatch")
        # Independently rebuild the optimizer/envelope, not merely trust metadata.
        mission = MissionWorkspace.network(
            case["configuration"], planning_method=row["method"]
        )
        try:
            expected_meta = mission.operator.session._scenario["metadata"][
                "network_workspace"
            ]
            if canonical_json(expected_meta) != canonical_json(meta):
                raise ValueError("Planner reproduction mismatch")
        finally:
            mission.close()
        if (
            row["diagnostic"] != _diagnostic(meta)
            or row["planned"] != meta["schedule"]["predicted_fulfilled"]
            or row["optimal_in_forecast"] != meta["schedule"].get("optimal")
            or row["search_budget_exhausted"]
            != meta["schedule"]["search_budget_exhausted"]
        ):
            raise ValueError("Diagnostic or optimality mismatch")
    paired = []
    for name in selected:
        cells = {r["method"]: r for r in rows if r["case"] == name}
        complete = all(r["status"] == "completed" for r in cells.values())
        paired.append(
            {
                "case": name,
                "measured": complete,
                "fulfilled": {m: r["metrics"]["fulfilled"] for m, r in cells.items()}
                if complete
                else None,
                "priority": {
                    m: r["metrics"]["weighted_fulfilled"] for m, r in cells.items()
                }
                if complete
                else None,
            }
        )
    return {
        "valid": True,
        "replayed": sum(r["status"] == "completed" for r in rows),
        "cases": len(selected),
        "paired": paired,
        "external_llm_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--audit", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--method", action="append", choices=(*METHODS, "sequence_lanes")
    )
    args = parser.parse_args()
    if args.audit:
        result = audit(args.audit)
    else:
        selected = args.case or [c["id"] for c in compile_suite()["cases"][:6]]
        run(args.output, selected, methods=args.method or METHODS)
        result = audit(args.output)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
