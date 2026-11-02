"""Small real-runtime corpus with train/holdout/OOD and a retained negative case."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.factory import build_dataset, verify_bundle
from space_env.runtime import reference_scenario
from space_env.simulation.snapshot import atomic_write, canonical_json


def load_example(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location("factory_published_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_spec():
    physical = load_example("reference_operations/demo.py").build_spec()
    physical["horizon_sec"] = 8.0
    for p in physical["products"]:
        p["deadline_sec"] = 8.0
    task = {
        "id": "ground",
        "roles": {a: {"actors": [a]} for a in ["a", "b"]},
        "decision_sec": 2.0,
        "max_decision_sec": 4.0,
        "max_decisions": 8,
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
    templates = []
    for i, split in enumerate(["train", "holdout", "holdout", "train", "ood", "ood"]):
        native = copy.deepcopy(physical)
        native["stations"][0]["lat_deg"] += 0.02 * i
        if i == 3:
            native["outages"] = [
                {"station": "Ground", "start_sec": 0.0, "end_sec": 8.0}
            ]
        templates.append(
            {
                "id": f"mission-{i}",
                "family": "closed-ground"
                if i == 3
                else "small-memory"
                if split == "ood"
                else "ground",
                "kind": "shared_terminal",
                "source": "synthetic published fixed-GCRS demo; no calibration",
                "scenario": reference_scenario(native, id="input", lineage="input"),
                "task": task,
                "split": split,
                "samples": 2,
                "distributions": [
                    {
                        "parameter": "product_bits_scale",
                        "unit": "ratio",
                        "values": [0.65 + 0.025 * i],
                        "source": "synthetic declared workload strata",
                    },
                    {
                        "parameter": "memory_scale",
                        "unit": "ratio",
                        "values": [0.8 if split == "ood" else 1.0],
                        "source": "synthetic capacity stratum",
                    },
                ],
            }
        )
    return {
        "id": "B08-ground-corpus",
        "seed": 41,
        "templates": templates,
        "teacher_gates": {"min_cases_per_family": 2},
        "export_roles": ["@central", "a"],
    }


def eo_spec():
    scenario, task = load_example("planning/demo.py").build("nominal")
    scenario["definition"]["horizon_sec"] = 20.0
    for request in scenario["definition"]["lifecycle"]["requests"]:
        request["deadline_sec"] = 20.0
    return {
        "id": "B08-coupled-EO",
        "seed": 17,
        "policies": [{"name": "route_aware_rolling", "max_nodes": 256}],
        "templates": [
            {
                "id": "eo",
                "family": "EO-CPU-ISL-ground",
                "kind": "relay",
                "source": "synthetic B07 coupled mission, not hardware calibrated",
                "scenario": scenario,
                "task": task,
                "split": "train",
                "distributions": [
                    {
                        "parameter": "product_bits_scale",
                        "unit": "ratio",
                        "values": [0.75],
                        "source": "synthetic resource scaling",
                    }
                ],
            }
        ],
        "quality": {"require_teacher_gates": False},
        "export_roles": ["@central"],
    }


def main(directory, eo=False):
    directory = Path(directory)
    spec = eo_spec() if eo else build_spec()
    directory.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(directory.with_suffix(".spec.json"), canonical_json(spec))
    result = build_dataset(spec, directory)
    assert result["ok"], json.loads((directory / "report.json").read_text())
    assert verify_bundle(directory)["ok"]
    report = json.loads((directory / "report.json").read_text())
    if eo:
        evaluation = json.loads((directory / "benchmark.json").read_text())
        assert evaluation["rows"][0]["metrics"]["fulfilled"] == 2
    else:
        assert report["episodes"] == 36 and report["coverage"]["geometry_groups"] == 6
        assert report["teacher"]["status"] == "passed_declared_gates"
        assert any(
            "incomplete_teacher_episode" in e["excluded"]
            for a in report["audit"]
            for e in a["exports"]
        )
        assert any(
            "duplicate_trajectory" in e["excluded"]
            for a in report["audit"]
            for e in a["exports"]
        )
    print(
        json.dumps(
            result | {"expert_qualified": False, "verified": True, "coupled_eo": eo}
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--eo" in sys.argv)
