"""Paid RCS/wheel unloading, mid-operation replay and subsequent EO mission."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.runtime.cli import apply_plan


def build_scenario(mode="nominal"):
    if mode not in {"nominal", "limited"}:
        raise ValueError("mode must be nominal or limited")
    path = Path(__file__).resolve().parents[1] / "rigid_mission/demo.py"
    loader = importlib.util.spec_from_file_location("rigid_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    previous, (first, second) = module.build_scenario("tuned")
    template = MissionRuntime(previous, access=AccessPolicy(truth=True))
    definition = previous["definition"]
    q = pointing_quaternion(
        template._backend.mission.ephemerides["a"],
        template._backend.acquisition_targets["site"],
        3.0,
    )
    a = definition["spacecraft_bus"]["a"]["attitude"]
    a["initial_quaternion_wxyz"] = q.tolist()
    a["rigid"]["initial_wheel_momentum_nms"] = [0.08, 0.0, 0.0]
    authority = 0.1 if mode == "nominal" else 0.02
    definition["unloading"] = {
        "actors": {
            "a": {
                "model_id": "synthetic-balanced-rcs/1",
                "parameter_status": "synthetic",
                "source": "declared ideal nozzle-pair assumptions; no flight calibration",
                "propellant_pool": "shared-lumped",
                "max_total_nozzle_thrust_n": 1.0,
                "max_torque_nm": [authority] * 3,
                "lever_arm_m": [0.5] * 3,
                "isp_s": 60.0,
                "startup_sec": 0.2,
                "startup_power_w": 3.0,
                "firing_power_w": 4.0,
            }
        },
        "max_segments": 100,
    }
    operations = [
        {
            "op": "submit",
            "idempotency_key": "start-unloading",
            "commands": [
                {
                    "command_id": "unload",
                    "actor_id": "a",
                    "kind": "momentum_unload",
                    "duration_sec": 3.0,
                    "momentum_unload": {
                        "frame": "BODY",
                        "target_momentum_nms": [0.0, 0.0, 0.0],
                        "tolerance_nms": 1e-5,
                    },
                }
            ],
        },
        {"op": "advance", "idempotency_key": "partial-unloading", "dt_sec": 0.5},
        {"op": "advance", "idempotency_key": "finish-unloading", "dt_sec": 2.5},
    ]
    operations += first["operations"] + second["operations"]
    for index, row in enumerate(operations):
        row["expected_revision"] = index
    return spacecraft_scenario(
        definition, id="unloading-" + mode, lineage="B04-D2b3"
    ), ({"operations": operations[:2]}, {"operations": operations[2:]})


def main(directory, mode="nominal"):
    scenario, (prefix, resume) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in [("scenario", scenario), ("plan", prefix), ("resume", resume)]:
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for operation in resume["operations"]:
        apply_plan(model, {"operations": [operation]})
        if operation["idempotency_key"] == "partial-tracking":
            model.save(directory / "tracking.json")
            model = MissionRuntime.load(
                directory / "tracking.json", access=AccessPolicy(truth=True)
            )
    model.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )
    report = model.observe(view="truth")["backend"]["report"]
    life = report["lifecycle"]
    unloading = life["unloading"]
    op = unloading["operations"]["unload"]
    rigid = report["spacecraft_bus"]["states"]["a"]["rigid_attitude"]
    assert op["status"] == ("completed" if mode == "nominal" else "failed")
    if mode == "limited":
        assert op["reason"] == "unload_window_ended"
        assert op["remaining_momentum_nms"][0] > 0.02
    assert op["consumed_propellant_kg"] > 0
    assert life["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert abs(unloading["fuel_balance_error_kg"]["a"]) < 1e-9
    assert abs(unloading["energy_balance_error_j"]["unload"]) < 1e-8
    assert max(abs(v) for v in rigid["angular_momentum_balance_error_gcrs_nms"]) < 1e-8
    assert abs(rigid["kinetic_work_balance_error_j"]) < 1e-8
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "metrics": life["metrics"],
                "unloading": unloading,
                "rigid_attitude": rigid,
                "geometry_revision": report["geometry_revision"],
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "scope": "resource-accounted ideal balanced RCS; single declared tank and constant inertia applicability",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "nominal")
