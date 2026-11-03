"""Execute assumed battery variants of the synthetic EO/CPU/relay/ACK mission."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from space_env.scaling.ensemble import (
    run_ensemble_qualification,
    verify_ensemble_qualification,
)
from space_env.simulation.snapshot import atomic_write, canonical_json


def build_spec(*, rejected=False, new_process=False):
    path = Path(__file__).resolve().parents[1] / "planning/demo.py"
    loader = importlib.util.spec_from_file_location("ensemble_planning_host", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, task = module.build("nominal")
    task["runtime_audit"] = "compact"
    parameter = {
        "id": "battery",
        "actor": "a",
        "target": "battery_capacity_scale",
        "mode": "uniform",
        "low": 0.9,
        "high": 1.1,
        "rationale": "assumed host capacity perturbation, not measured uncertainty",
    }
    if rejected:
        parameter.update(target="solar_efficiency_scale", low=100.0, high=101.0)
    result = {
        "id": "native-ensemble-smoke",
        "workers": 2,
        "resume_after_decisions": 1,
        "workload": {
            "id": "host",
            "family": "EO-CPU-relay-ACK",
            "scenario": scenario,
            "task": task,
            "source": "synthetic constructive mission; not an independently located fixed-site qualification",
            "policy": {"name": "route_aware_rolling", "max_nodes": 256},
            "repeats": 1,
            "timeout_sec": 180,
            "min_completion_fraction": 1.0,
        },
        "sensitivity": {
            "id": "native-battery-assumptions",
            "seed": 29,
            "samples": 2,
            "parameters": [parameter],
        },
    }
    if new_process:
        result["resume_in_new_process"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rejected", action="store_true")
    parser.add_argument("--new-process", action="store_true")
    args = parser.parse_args()
    result = run_ensemble_qualification(
        build_spec(rejected=args.rejected, new_process=args.new_process), args.directory
    )
    audit = verify_ensemble_qualification(args.directory)
    assert audit["verified"] and not audit["native_reexecuted"]
    if args.rejected:
        assert (
            not result["passed"]
            and result["summary"]["counts"]["structural_rejected"] == 2
        )
        assert result["summary"]["baseline"]["status"] == "passed"
    else:
        assert result["passed"] and result["summary"]["counts"]["passed"] == 2
    atomic_write(args.directory / "audit.json", canonical_json(audit))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "expected_negative": args.rejected,
                "counts": result["summary"]["counts"],
                "baseline": result["summary"]["baseline"]["status"],
                "artifact_audit": audit["verified"],
                "sha256": result["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
