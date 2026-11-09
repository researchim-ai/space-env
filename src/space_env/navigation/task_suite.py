"""Fixed public sensor-aware delivery tasks; not random/OOD or calibrated hardware."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import (
    LLMEpisode,
    export_training,
    seal,
    sources,
    verify_record,
)
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks import TaskSession, TaskSpec
from space_env.tasks.modeling_suite import compile_suite as physical_suite


def compile_suite():
    original = next(c for c in physical_suite()["cases"] if c["id"] == "orbit-nominal")
    cases = []
    for name in ["nominal", "delayed-drop", "no-contact"]:
        scenario = copy.deepcopy(original["scenario"])
        definition = scenario["definition"]
        definition["operations"]["products"] = [
            {
                "id": "ready-data",
                "actor": "a",
                "size_bits": 1e6,
                "ready_sec": 0.0,
                "deadline_sec": 10.0,
                "destination": "Ground",
            }
        ]
        if name == "no-contact":
            definition["operations"]["stations"][0]["lat_deg"] = 70.0
        sensors = {}
        for i, actor in enumerate(["a", "b"]):
            initial = definition["orbits"][actor]["initial"]
            x = np.array([*initial["r_m"], *initial["v_m_s"]])
            x[:3] += [30.0, -10.0, 5.0]
            sensors[actor] = {
                "kind": "cartesian_position",
                "initial_estimate_si": x.tolist(),
                "initial_covariance_si": np.diag(
                    [900.0, 900.0, 900.0, 1.0, 1.0, 1.0]
                ).tolist(),
                "standard_deviation": [3.0, 3.0, 3.0],
                "latency_sec": 0.25 if name == "delayed-drop" else 0.0,
                "dropout_probability": 0.25 if name == "delayed-drop" else 0.0,
                "seed": 10 + i,
            }
        task = {
            "id": "sensor-delivery-" + name,
            "roles": {"a": {"actors": ["a"]}, "b": {"actors": ["b"]}},
            "actions": [
                {
                    "id": "deliver",
                    "role": "a",
                    "kind": "submit",
                    "commands": [
                        {
                            "command_id": "deliver",
                            "actor_id": "a",
                            "kind": "downlink",
                            "destination_id": "Ground",
                            "product_id": "ready-data",
                            "duration_sec": 2.0,
                        }
                    ],
                }
            ],
            "decision_sec": 1.0,
            "max_decision_sec": 5.0,
            "max_decisions": 32,
            "navigation_sensors": sensors,
            "model_requirements": [
                {"feature": f, "actors": ["a", "b"]}
                for f in ["navigation_measurements", "state_estimation"]
            ],
            "goal": {
                "deadline_sec": 5.0,
                "conditions": [
                    {
                        "id": "ground-product",
                        "kind": "product_delivery",
                        "actor": "a",
                        "product_ids": ["ready-data"],
                    }
                ],
            },
        }
        task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
        cases.append(
            {
                "id": name,
                "scenario": scenario,
                "task": task,
                "task_text": "Deliver the declared ready onboard data to Ground within its deadline. Orbit coordinates are delayed noisy sensor estimates. Actions masks are declarative, not a promise of admission.",
                "steps": [{"action_ids": ["deliver"], "advance_sec": 2.0}]
                + [{"action_ids": ["wait"], "advance_sec": 1.0} for _ in range(5)],
            }
        )
    return cases


def execute(case, policy, split="test"):
    s = TaskSession(case["scenario"], case["task"])
    ep = LLMEpisode(
        s,
        task_text=case["task_text"],
        group="sensor-" + case["id"],
        split=split,
        policy_origin="scripted_privileged"
        if policy == "reference"
        else "scripted_public",
    )
    chooser = (
        reference_policy(case["steps"], {a.id: a.role for a in s._task.actions})
        if policy == "reference"
        else public_policy("wait", 1.0)
    )
    record = run_agent(ep, chooser)
    verify_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("choose output or audit")
    cases = compile_suite()
    if args.audit:
        report = json.loads((args.audit / "report.json").read_text())
        if report.get("model_sources") != sources():
            raise ValueError("sensor task source mismatch")
        if report != seal({k: v for k, v in report.items() if k != "sha256"}):
            raise ValueError("sensor task report seal mismatch")
        if [(r["case"], r["policy"]) for r in report["rows"]] != [
            (c["id"], p) for c in cases for p in ["wait", "reference"]
        ]:
            raise ValueError("sensor matrix mismatch")
        for c in cases:
            for p in ["wait", "reference"]:
                expected = execute(c, p, split="train")
                saved = json.loads(
                    (args.audit / (c["id"] + "--" + p + ".json")).read_text()
                )
                if expected != saved:
                    raise ValueError("sensor task replay mismatch")
                row = next(
                    r
                    for r in report["rows"]
                    if r["case"] == c["id"] and r["policy"] == p
                )
                if (
                    row["episode_sha256"] != saved["sha256"]
                    or row["metrics"] != saved["metrics"]
                ):
                    raise ValueError("sensor task metrics mismatch")
        print(canonical_json({"valid": True, "native_replayed": 6}))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for c in cases:
            for p in ["wait", "reference"]:
                record = execute(c, p, split="train")
                stem = c["id"] + "--" + p
                atomic_write(args.output / (stem + ".json"), canonical_json(record))
                export_training(record, args.output / (stem + "-rl.jsonl"))
                if record["metrics"]["sft_eligible"]:
                    export_training(
                        record, args.output / (stem + "-sft.jsonl"), format="sft"
                    )
                rows.append(
                    {
                        "case": c["id"],
                        "policy": p,
                        "episode_sha256": record["sha256"],
                        "metrics": record["metrics"],
                    }
                )
        report = seal(
            {
                "schema": "space-env.sensor-tasks-study/1",
                "model_sources": sources(),
                "rows": rows,
                "scope": "six public scripted train episodes; one input geometry; no learned model or blind evaluation",
            }
        )
        atomic_write(args.output / "report.json", canonical_json(report))
        print(
            canonical_json(
                {
                    "sha256": report["sha256"],
                    "successes": sum(r["metrics"]["success"] for r in rows),
                }
            )
        )


if __name__ == "__main__":
    main()
