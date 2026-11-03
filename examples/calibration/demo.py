"""Published component points on a synthetic host; native ensemble/restore smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.calibration import (
    compile_ensemble,
    hardware_catalog,
    solar_resources,
    wheel_model,
)
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve().parents[1] / "rigid_mission/demo.py"
    loader = importlib.util.spec_from_file_location("synthetic_host_example", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, _ = module.build_scenario("tuned")
    scenario["id"] = "published-component-synthetic-host"
    scenario["lineage"] = (
        "Q5-C1 mixed evidence; synthetic host, not calibrated spacecraft"
    )
    assumptions = {
        "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "efficiency": 0.8,
        "idle_power_w": 0.1,
        "torque_loss_w_per_nm2": 1.0,
        "rationale": "synthetic host inertia and constant motor losses; only mechanical component points are published",
    }
    wheel = wheel_model(["cw0017"] * 3, assumptions)
    scenario["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]["model"] = wheel[
        "model"
    ]
    solar = solar_resources(
        "azur-3g30-bare",
        scenario["definition"]["operations"]["vehicles"]["a"],
        area_m2=0.1,
        packing_factor=0.9,
        assumption="synthetic constant BOL efficiency approximation; reference AM0 temperature, geometry and losses are not flight calibrated",
    )
    scenario["definition"]["operations"]["vehicles"]["a"] = solar["vehicle_resources"]
    scenario["sources"] += [
        {
            "id": s["id"],
            "kind": s["kind"],
            "sha256": s["content_sha256"],
            "uri": s["uri"],
            "license": s["redistribution"],
        }
        for s in hardware_catalog()["sources"].values()
    ]
    spec = {
        "id": "published-host-sensitivity",
        "seed": 29,
        "samples": 3,
        "parameters": [
            {
                "id": "inertia",
                "actor": "a",
                "target": "body_inertia_scale",
                "mode": "uniform",
                "low": 0.9,
                "high": 1.1,
                "rationale": "assumed host inertia variation; no manufacturer tolerance implied",
            },
            {
                "id": "battery",
                "actor": "a",
                "target": "battery_capacity_scale",
                "mode": "uniform",
                "low": 0.8,
                "high": 1.0,
                "rationale": "assumed capacity variation for a paired mission sensitivity experiment",
            },
        ],
    }
    ensemble = compile_ensemble(scenario, spec)
    assert ensemble["rejected"] == 0 and ensemble["structural_candidates"] == 3
    atomic_write(directory / "ensemble.json", canonical_json(ensemble))
    results = []
    for member in ensemble["members"]:
        runtime = MissionRuntime(member["scenario"], access=AccessPolicy(truth=True))
        runtime.submit(
            [
                {
                    "command_id": "wheel",
                    "actor_id": "a",
                    "kind": "wheel_torque",
                    "duration_sec": 0.5,
                    "wheel_torque": {
                        "frame": "BODY",
                        "motor_torque_nm": [0.0, 0.0, 0.0001],
                    },
                }
            ],
            expected_revision=runtime.revision,
            idempotency_key="wheel",
        )
        runtime.advance(
            0.25, expected_revision=runtime.revision, idempotency_key="prefix"
        )
        prefix = directory / f"prefix-{member['index']}.json"
        runtime.save(prefix)
        cold = MissionRuntime.load(prefix, access=AccessPolicy(truth=True))
        for run in (runtime, cold):
            run.advance(
                0.25, expected_revision=run.revision, idempotency_key="continuation"
            )
        assert cold.checkpoint() == runtime.checkpoint()
        final = directory / f"final-{member['index']}.json"
        cold.save(final)
        assert (
            MissionRuntime.load(final, access=AccessPolicy(truth=True)).checkpoint()
            == cold.checkpoint()
        )
        state = cold.observe("a")["actors"]["a"]["spacecraft_bus"]["rigid_attitude"]
        assert abs(state["wheel_momentum_nms"][2] - 0.00005) < 1e-10
        assert state["dc_energy_j"] > 0
        results.append(
            {
                "index": member["index"],
                "scenario_sha256": member["scenario_sha256"],
                "native_restore_replay": True,
                "wheel_momentum_nms": state["wheel_momentum_nms"],
                "dc_energy_j": state["dc_energy_j"],
            }
        )
    result = {
        "schema": "space-env.calibration-native-smoke/1",
        "passed": True,
        "members": results,
        "ensemble_sha256": ensemble["sha256"],
        "wheel_adapter_sha256": wheel["sha256"],
        "solar_adapter_sha256": solar["sha256"],
        "scope": "three short native mixed-evidence sensitivity cases; not a calibrated mission or a teacher-quality benchmark",
    }
    result["sha256"] = digest(result)
    atomic_write(directory / "receipt.json", canonical_json(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
