"""Declared synthetic scale probes, separate from operational calibration."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import reference_scenario
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location("scale_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed(count, horizon=4.0):
    native = load("reference_operations/demo.py").build_spec()
    trajectory, vehicle = native["ephemerides"]["a"], native["vehicles"]["a"]
    names = [f"sat-{i:03}" for i in range(count)]
    native.update(
        ephemerides={a: copy.deepcopy(trajectory) for a in names},
        vehicles={a: copy.deepcopy(vehicle) for a in names},
        horizon_sec=horizon,
        outages=[],
    )
    # Identical ephemerides deliberately stress resource/action/telemetry fanout;
    # this is not an orbital separation or conjunction qualification workload.
    native["stations"][0]["terminals"] = count
    native["products"] = [
        {"id": "p-" + a, "actor": a, "size_bits": 100000.0, "deadline_sec": horizon}
        for a in names
    ]
    task = {
        "id": f"fixed-{count}",
        "runtime_audit": "compact",
        "roles": {a: {"actors": [a]} for a in names},
        "max_joint_actions": count,
        "max_actions": count + 1,
        "max_entities": 4 * count + 16,
        "max_relations": 16 * count + 16,
        "decision_sec": 2.0,
        "max_decision_sec": 4.0,
        "max_decisions": 8,
        "actions": [
            {
                "id": "send-" + a,
                "role": a,
                "kind": "submit",
                "commands": [
                    {
                        "command_id": a,
                        "actor_id": a,
                        "kind": "downlink",
                        "duration_sec": 2.0,
                        "destination_id": "Ground",
                        "product_id": "p-" + a,
                    }
                ],
            }
            for a in names
        ],
    }
    return {
        "id": f"fixed-{count}",
        "family": "synthetic-fixed-fanout",
        "source": "Published reference ephemeris cloned for workload fanout only; no independent orbital geometry or calibrated fleet",
        "scenario": reference_scenario(
            native, id=f"fixed-{count}", lineage=f"scale-fixed-{count}"
        ),
        "task": task,
        "policy": {"name": "priority"},
        "min_completion_fraction": 1.0,
    }


def manifest(full=False):
    workloads = [fixed(n) for n in ([1, 4, 16, 64] if full else [4])]
    if full:
        eo = load("factory/demo.py").eo_spec()["templates"][0]
        for mode in ["full", "compact"]:
            t = copy.deepcopy(eo["task"])
            t["runtime_audit"] = mode
            workloads.append(
                {
                    "id": "coupled-" + mode,
                    "family": "coupled-EO-CPU-DTN-ACK",
                    "source": "Published synthetic three-actor EO/CPU/relay/thermal mission, 20 SI seconds",
                    "scenario": eo["scenario"],
                    "task": t,
                    "policy": {"name": "route_aware_rolling", "max_nodes": 256},
                    "min_completion_fraction": 1.0,
                }
            )
        for mode in ["loss", "ttl"]:
            scenario, task = load("planning/demo.py").build(mode)
            task["runtime_audit"] = "compact"
            workloads.append(
                {
                    "id": "fault-" + mode,
                    "family": "coupled-fault",
                    "source": "Published synthetic loss/TTL witness; infeasibility retained, not filtered",
                    "scenario": scenario,
                    "task": task,
                    "policy": {"name": "route_aware_rolling", "max_nodes": 256},
                    "min_completion_fraction": 0.0,
                    "timeout_sec": 600,
                }
            )
    return {"id": "b09-published-probes", "workloads": workloads}


if __name__ == "__main__":
    value = manifest("--full" in sys.argv)
    atomic_write(sys.argv[1], canonical_json(value))
    print(json.dumps({"workloads": len(value["workloads"]), "sha256": digest(value)}))
