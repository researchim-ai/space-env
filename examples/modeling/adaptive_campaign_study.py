"""Paired fixed/adaptive polling on identical continuous synthetic missions."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from examples.modeling.reactive_campaign_study import cases, run_reactive
from space_env.industrial.agent import export_training, verify_episode
from space_env.simulation.snapshot import atomic_write, canonical_json


def accounting(record):
    resources = record["metrics"]["resource_accounting"]
    energy = resources["energy"]
    bus = resources["bus"]
    errors_wh = [abs(row["balance_error_wh"]) for row in energy.values()]
    errors_wh.extend(
        abs(value)
        for row in bus.values()
        for key, value in row.items()
        if key.endswith("error_wh")
    )
    errors_j = [abs(row["thermal_balance_error_j"]) for row in bus.values()]
    return {
        "consumed_wh": sum(row["consumed_wh"] for row in energy.values()),
        "max_balance_error_wh": max(errors_wh, default=0.0),
        "max_thermal_balance_error_j": max(errors_j, default=0.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, pairs = [], []
    for case in cases():
        directory = args.output / case["id"]
        directory.mkdir()
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        results = {}
        initial = None
        for adaptive in [False, True]:
            name = "adaptive" if adaptive else "fixed"
            contract = {
                "minimum_completed_requests": 3,
                "completion_action_ids": ["control"],
                "poll_sec": 0.25,
                "adaptive_wait": adaptive,
                "max_wait_sec": 1.0,
            }
            record = run_reactive(case["configuration"], public_contract=contract)
            if initial is not None and initial != record["initial"]["configuration"]:
                raise ValueError("paired native initial configuration differs")
            initial = record["initial"]["configuration"]
            verified = verify_episode(record)
            trace = verify_policy_trace(record, contract)
            atomic_write(directory / (name + ".json"), canonical_json(record))
            export_training(record, directory / (name + "-rl.jsonl"))
            if record["metrics"]["success"] and all(
                e["response"].get("feedback", {}).get("accepted")
                for e in record["events"]
            ):
                export_training(record, directory / (name + "-sft.jsonl"), format="sft")
            row = {
                "case": case["id"],
                "policy": name,
                "contract": contract,
                "calls": len(record["events"]),
                "metrics": record["metrics"],
                "accounting": accounting(record),
                "verified": verified,
                "policy_trace": trace,
                "sha256": record["sha256"],
            }
            rows.append(row)
            results[name] = row
            print(case["id"], name, row["calls"], row["metrics"]["success"], flush=True)
        old, new = results["fixed"], results["adaptive"]
        pairs.append(
            {
                "case": case["id"],
                "expected_success": case["expected_reactive_success"],
                "both_expected_outcomes": all(
                    r["metrics"]["success"] == case["expected_reactive_success"]
                    for r in results.values()
                ),
                "calls_saved": old["calls"] - new["calls"],
                "additional_mission_sec": new["metrics"]["time_sec"]
                - old["metrics"]["time_sec"],
                "additional_consumed_wh": new["accounting"]["consumed_wh"]
                - old["accounting"]["consumed_wh"],
            }
        )
    report = {
        "scope": "Six hand-authored short paired variants; not multi-day validation, model inference, training or independent physical calibration.",
        "rows": rows,
        "pairs": pairs,
    }
    atomic_write(args.output / "report.json", canonical_json(report))
    print(canonical_json({"pairs": pairs}))


if __name__ == "__main__":
    main()
