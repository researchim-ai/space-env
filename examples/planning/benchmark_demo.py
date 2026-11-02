"""A portable positive/blocked benchmark; the negative case must remain visible."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.planning.benchmark import run_benchmark, verify_evaluation
from space_env.runtime import reference_scenario
from space_env.simulation.snapshot import atomic_write, canonical_json


def build():
    path = Path(__file__).resolve().parents[1] / "reference_operations/demo.py"
    spec = importlib.util.spec_from_file_location("published_reference_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    physical = module.build_spec()
    physical["horizon_sec"] = 8.0
    for product in physical["products"]:
        product["deadline_sec"] = 8.0
    task = {
        "id": "fixed-ground-baselines",
        "roles": {a: {"actors": [a]} for a in ["a", "b"]},
        "decision_sec": 1.0,
        "max_decision_sec": 4.0,
        "max_decisions": 16,
        "actions": [
            {
                "id": "ground-" + a,
                "role": a,
                "kind": "submit",
                "commands": [
                    {
                        "command_id": a,
                        "actor_id": a,
                        "kind": "downlink",
                        "duration_sec": 4.0,
                        "destination_id": "Ground",
                        "product_id": "scene-" + a,
                    }
                ],
            }
            for a in ["a", "b"]
        ],
    }
    blocked = copy.deepcopy(physical)
    blocked["outages"] = [{"station": "Ground", "start_sec": 0.0, "end_sec": 8.0}]
    return {
        "id": "B07-positive-and-infeasible",
        "cases": [
            {
                "id": "positive",
                "family": "fixed-ground",
                "split": "holdout",
                "episode_seeds": [0],
                "scenario": reference_scenario(
                    physical, id="ground", lineage="B07-ground"
                ),
                "task": task,
            },
            {
                "id": "blocked",
                "family": "closed-ground",
                "split": "ood",
                "episode_seeds": [0],
                "scenario": reference_scenario(
                    blocked, id="ground", lineage="B07-closed"
                ),
                "task": task,
            },
        ],
        "policies": [
            {"name": n, "seed": 19, "max_nodes": 64}
            for n in [
                "wait",
                "random_masked",
                "priority",
                "earliest_deadline",
                "route_aware_rolling",
            ]
        ],
        "teacher_gates": {"min_cases_per_family": 1},
    }


def main(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build()
    atomic_write(directory / "manifest.json", canonical_json(manifest))
    report = run_benchmark(manifest, artifacts=directory / "episodes")
    atomic_write(directory / "evaluation.json", canonical_json(report))
    assert report["passed_execution"] and len(report["rows"]) == 10
    assert all(
        r["metrics"]["fulfilled"] == 0 for r in report["rows"] if r["case"] == "blocked"
    )
    assert report["teacher"]["status"] == "failed_declared_gates"
    assert verify_evaluation(report)["ok"]
    result = {
        "episodes": 10,
        "execution_verified": True,
        "semantic_replay": True,
        "teacher": report["teacher"]["status"],
        "expert_qualified": False,
        "blocked_cases_retained": 5,
        "summary": report["summary"],
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main(Path(sys.argv[1]))
