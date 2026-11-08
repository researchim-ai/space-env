"""Paired native runs of conservative and fixed-start terminal compaction.

Run explicitly locally; failures are retained. No CI or hardware claims.
"""

import argparse
import json
from pathlib import Path

from space_env.planning.network_benchmark import NetworkSuite
from space_env.planning.network_demand_benchmark import demand_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.network_evidence import (
    run_scale_case,
    seal,
    source_manifest,
    validate_row,
)


def audit(path):
    from space_env.planning.terminal_sequence import verify_terminal_schedule

    def admit(value):
        if value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("Invalid artifact checksum")

    stored = json.loads(path.read_text())
    admit(stored)
    report = stored.get("report", stored)
    admit(report)
    spec = NetworkSuite.model_validate(report["suite"])
    if report["schema"] != "space-env.terminal-sequence-comparison/1" or report[
        "suite_sha256"
    ] != digest(report["suite"]):
        raise ValueError("Invalid report identity")
    expected = {
        (c.id, m) for c in spec.cases for m in ("exact_small", "sequence_compact")
    }
    rows = report["rows"]
    if (
        len(rows) != len(expected)
        or {(r["case"], r["method"]) for r in rows} != expected
    ):
        raise ValueError("Incomplete/duplicate comparison matrix")
    for row in rows:
        case = next(c for c in spec.cases if c.id == row["case"])
        work = {
            "id": case.id + "--" + row["method"],
            "configuration": case.configuration.model_dump(mode="json", by_alias=True),
            "recover_after": None,
            "planning_method": row["method"],
        }
        if row["workload"] != work:
            raise ValueError("Workload binding mismatch")
        validate_row(
            row["result"], work, expected_source_sha256=digest(report["model_sources"])
        )
    paired = []
    for case in spec.cases:
        a, b = [
            next(
                r["result"]
                for r in rows
                if r["case"] == case.id and r["method"] == method
            )
            for method in ("exact_small", "sequence_compact")
        ]
        sequence = b.get("terminal_sequence")
        if b["status"] == "completed":
            if (
                not sequence
                or sequence["global_sequence_optimality_established"] is not False
            ):
                raise ValueError("Invalid sequence scope")
            if sequence["saved_reservation_sec"] != sum(
                (r["before_cells"] - r["after_cells"])
                * case.configuration.mission.decision_sec
                for r in sequence["rows"]
            ):
                raise ValueError("Invalid reservation summary")
        paired.append(
            {
                "case": case.id,
                "same_forecast": a.get("forecast_sha256") == b.get("forecast_sha256"),
                "same_native_value": a.get("weighted_fulfilled")
                == b.get("weighted_fulfilled")
                if a["status"] == b["status"] == "completed"
                else None,
                "saved_reservation_sec": (sequence or {}).get("saved_reservation_sec"),
                "forecast_native_match": b.get("forecast_weighted_fulfilled")
                == b.get("weighted_fulfilled")
                if b["status"] == "completed"
                else None,
            }
        )
    if paired != report["paired"]:
        raise ValueError("Paired summary mismatch")
    for proof in stored.get("proofs", []):
        verify_terminal_schedule(
            proof["forecast"],
            proof["base_schedule"],
            proof["geometry"],
            proof["schedule"],
        )
        row = next(
            r
            for r in rows
            if r["case"] == proof["case"] and r["method"] == "sequence_compact"
        )
        if row["result"]["schedule_sha256"] != proof["schedule"]["sha256"]:
            raise ValueError("Reproduced plan/native binding mismatch")
    return {
        "valid": True,
        "native_rows": len(rows),
        "plans_reproduced": len(stored.get("proofs", [])),
        "source_matches_current": report["model_sources"] == source_manifest(),
        "native_replayed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    suite = demand_suite()
    rows = []
    for case in NetworkSuite.model_validate(suite).cases:
        for method in ("exact_small", "sequence_compact"):
            work = {
                "id": case.id + "--" + method,
                "configuration": case.configuration.model_dump(
                    mode="json", by_alias=True
                ),
                "recover_after": None,
                "planning_method": method,
            }
            result = run_scale_case(work, timeout_sec=120)
            validate_row(result, work)
            row = {
                "case": case.id,
                "method": method,
                "workload": work,
                "result": result,
            }
            rows.append(row)
            atomic_write(
                args.output / (work["id"] + ".json"), canonical_json(seal(row)) + "\n"
            )
            print(
                case.id,
                method,
                result["status"],
                (result.get("terminal_sequence") or {}).get("saved_reservation_sec"),
                flush=True,
            )
    completed = all(r["result"]["status"] == "completed" for r in rows)
    paired = []
    for case in NetworkSuite.model_validate(suite).cases:
        a, b = [r["result"] for r in rows if r["case"] == case.id]
        paired.append(
            {
                "case": case.id,
                "same_forecast": a.get("forecast_sha256") == b.get("forecast_sha256"),
                "same_native_value": a.get("weighted_fulfilled")
                == b.get("weighted_fulfilled")
                if a["status"] == b["status"] == "completed"
                else None,
                "saved_reservation_sec": (b.get("terminal_sequence") or {}).get(
                    "saved_reservation_sec"
                ),
                "forecast_native_match": b.get("forecast_weighted_fulfilled")
                == b.get("weighted_fulfilled")
                if b["status"] == "completed"
                else None,
            }
        )
    report = seal(
        {
            "schema": "space-env.terminal-sequence-comparison/1",
            "suite": suite,
            "suite_sha256": digest(suite),
            "model_sources": source_manifest(),
            "rows": rows,
            "paired": paired,
            "scope": "fixed starts and selected requests; single-terminal sequence bounds; multi-terminal conservative fallback; reserved time savings, not additional deliveries, global optimality or hardware calibration",
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report) + "\n")
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
