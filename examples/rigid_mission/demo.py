"""EO mission with paid wheel control, persistent spin and checkpoint replay."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.runtime.cli import apply_plan


def build_scenario(mode="tuned"):
    if mode not in {"tuned", "weak"}:
        raise ValueError("mode must be tuned or weak")
    path = Path(__file__).resolve().parents[1] / "resource_lanes/demo.py"
    loader = importlib.util.spec_from_file_location("lanes_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    previous, _ = module.build_scenario("wide")
    definition = previous["definition"]
    first = definition["lifecycle"]["requests"][1]
    second = copy.deepcopy(first)
    second.update(
        id="after-wheel", raw_product_id="raw-after", delivery_product_id="raw-after"
    )
    definition["lifecycle"]["requests"] = [first, second]
    definition["spacecraft_bus"]["a"]["attitude"]["rigid"] = {
        "model": {
            "model_id": "synthetic-wheel-bus/1",
            "source": "declared engineering assumptions; not a flight controller",
            "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            "wheel_inertia_kg_m2": [0.01, 0.01, 0.01],
            "max_torque_nm": [0.1, 0.1, 0.1],
            "max_momentum_nms": [0.2, 0.2, 0.2],
        },
        "supply_w": 12.0,
        "payload_max_body_rate_rad_s": 0.1,
    }
    if mode == "tuned":
        definition["spacecraft_bus"]["a"]["attitude"]["rigid"].update(
            proportional_nm_per_rad=10.0,
            damping_nm_per_rad_s=3.0,
        )
    scenario = spacecraft_scenario(
        definition, id="rigid-eo-" + mode, lineage="B04-D2b1"
    )
    ops = []

    def add(op, key, **fields):
        ops.append(
            {"op": op, "expected_revision": len(ops), "idempotency_key": key, **fields}
        )

    def image(id, request):
        return {
            "command_id": id,
            "actor_id": "a",
            "kind": "imaging",
            "duration_sec": 2.0,
            "request_id": request,
        }

    def radio(id, product):
        return {
            "command_id": id,
            "actor_id": "a",
            "kind": "downlink",
            "duration_sec": 1.0,
            "destination_id": "Ground",
            "product_id": product,
        }

    add("submit", "image-before", commands=[image("image-before", "customer-a")])
    add("advance", "capture-before", dt_sec=2.0)
    add(
        "submit",
        "wheel-and-cpu",
        commands=[
            {
                "command_id": "wheel",
                "actor_id": "a",
                "kind": "wheel_torque",
                "duration_sec": 1.0,
                "wheel_torque": {"frame": "BODY", "motor_torque_nm": [0.05, 0.0, 0.0]},
            },
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "transform",
                "duration_sec": 2.0,
                "job_id": "features-a",
            },
        ],
    )
    add("advance", "partial-wheel-and-cpu", dt_sec=0.25)
    add("cancel", "cancel-wheel", command_id="wheel")
    add("advance", "finish-cpu-while-coasting", dt_sec=1.75)
    add("submit", "deliver-before", commands=[radio("radio-before", "final-a")])
    add("advance", "first-ack", dt_sec=0.5)
    add("submit", "image-after", commands=[image("image-after", "after-wheel")])
    add("advance", "partial-tracking", dt_sec=1.0)
    add("advance", "capture-after", dt_sec=1.0)
    add("submit", "deliver-after", commands=[radio("radio-after", "raw-after")])
    add("advance", "second-ack", dt_sec=1.0)
    return scenario, ({"operations": ops[:4]}, {"operations": ops[4:]})


def main(directory, mode="tuned"):
    scenario, (prefix, continuation) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ]:
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for op in continuation["operations"]:
        apply_plan(model, {"operations": [op]})
        if op["idempotency_key"] == "partial-tracking":
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
    rigid = report["spacecraft_bus"]["states"]["a"]["rigid_attitude"]
    assert report["lifecycle"]["metrics"] == {
        "fulfilled_requests": 2 if mode == "tuned" else 1,
        "total_requests": 2,
    }
    if mode == "weak":
        assert (
            report["lifecycle"]["requests"]["after-wheel"]["reason"]
            == "coverage_requirement"
        )
        assert report["lifecycle"]["artifacts"]["raw-after"]["status"] == "delivered"
    assert abs(rigid["dc_heat_work_balance_error_j"]) < 1e-8
    assert abs(rigid["kinetic_work_balance_error_j"]) < 1e-8
    assert report["spacecraft_bus"]["operations"]["wheel"]["status"] == "cancelled"
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "request_outcomes": {
                    k: {f: v.get(f) for f in ("status", "reason", "delivered_sec")}
                    for k, v in report["lifecycle"]["requests"].items()
                },
                "metrics": report["lifecycle"]["metrics"],
                "rigid_attitude": rigid,
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "scope": "mission wheel control; body thrust and external unloading unsupported on rigid actors",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "tuned")
