"""Useful EO work before/after a paid body burn; chemical/electric variants."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.runtime.cli import apply_plan


def build_scenario(kind="chemical"):
    if kind not in {"chemical", "electric"}:
        raise ValueError("kind must be chemical or electric")
    path = Path(__file__).resolve().parents[1] / "payload_quality/demo.py"
    loader = importlib.util.spec_from_file_location("quality_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    previous, _ = module.build_scenario("optical")
    definition = previous["definition"]
    first = definition["lifecycle"]["requests"][0]
    after = copy.deepcopy(first)
    after.update(
        id="after-request", raw_product_id="raw-after", delivery_product_id="raw-after"
    )
    definition["lifecycle"]["requests"] = [first, after]
    definition["execution"] = {
        "actors": {
            a: {"memory_bandwidth_bps": 10000000.0} for a in previous["actor_order"]
        }
    }
    engine = {
        "kind": kind,
        "model_id": f"synthetic-{kind}/1",
        "source_status": "synthetic",
        "source": "declared demo assumptions; not a flight engine model",
        "max_thrust_n": 1.0,
        "max_gimbal_deg": 20.0,
        "gimbal_rate_deg_s": 5.0,
        "gimbal_settling_sec": 0.5,
        "gimbal_power_w": 2.0,
        "ignition_sec": 0.5,
        "ignition_power_w": 3.0,
        "firing_aux_power_w": 2.0,
        "chemical_wall_heat_w_at_max": 4.0 if kind == "chemical" else 0.0,
    }
    if kind == "electric":
        engine["electric_efficiency"] = 0.5
    definition["actuation"] = {"engines": {"a": engine}, "max_segments": 32}
    return spacecraft_scenario(
        definition, id=f"body-thrust-{kind}", lineage="B04-D1"
    ), plans(kind)


def plans(kind):
    operations = []

    def add(op, key, **fields):
        operations.append(
            {
                "op": op,
                "expected_revision": len(operations),
                "idempotency_key": key,
                **fields,
            }
        )

    angle = math.radians(10.0)
    add(
        "submit",
        "first-image",
        commands=[
            {
                "command_id": "image-before",
                "actor_id": "a",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "customer-a",
            }
        ],
    )
    add("advance", "acquire-before", dt_sec=2.0)
    add(
        "submit",
        "engine-and-cpu",
        commands=[
            {
                "command_id": "engine",
                "actor_id": "a",
                "kind": "body_burn",
                "duration_sec": 5.0,
                "body_burn": {
                    "frame": "BODY",
                    "direction_body": [math.sin(angle), 0.0, math.cos(angle)],
                    "thrust_n": 1.0 if kind == "chemical" else 0.01,
                    "firing_sec": 1.0,
                },
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
    add("advance", "partial-gimbal", dt_sec=0.5)
    add("advance", "finish-gimbal-and-cpu", dt_sec=1.5)
    add(
        "submit",
        "deliver-before",
        commands=[
            {
                "command_id": "radio-before",
                "actor_id": "a",
                "kind": "downlink",
                "duration_sec": 1.0,
                "product_id": "final-a",
                "destination_id": "Ground",
            }
        ],
    )
    add("advance", "settle-ignite-and-radio", dt_sec=1.0)
    add("advance", "partial-firing", dt_sec=0.5)
    add("advance", "finish-firing", dt_sec=0.5)
    add(
        "submit",
        "second-image",
        commands=[
            {
                "command_id": "image-after",
                "actor_id": "a",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "after-request",
            }
        ],
    )
    add("advance", "acquire-after", dt_sec=2.0)
    add(
        "submit",
        "deliver-after",
        commands=[
            {
                "command_id": "radio-after",
                "actor_id": "a",
                "kind": "downlink",
                "duration_sec": 1.0,
                "product_id": "raw-after",
                "destination_id": "Ground",
            }
        ],
    )
    add("advance", "after-ack", dt_sec=1.0)
    return {"operations": operations[:4]}, {"operations": operations[4:]}


def main(directory, kind="chemical"):
    scenario, (prefix, continuation) = build_scenario(kind)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))
    initial_mass = model.observe("a")["actors"]["a"]["orbit"]["mass_kg"]
    apply_plan(model, prefix)
    assert model.observe("a")["actors"]["a"]["orbit"]["mass_kg"] == initial_mass
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for operation in continuation["operations"]:
        apply_plan(model, {"operations": [operation]})
        if operation["idempotency_key"] == "partial-firing":
            model.save(directory / "partial-firing.json")
            model = MissionRuntime.load(
                directory / "partial-firing.json", access=AccessPolicy(truth=True)
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
    engine = life["actuation"]
    assert life["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert engine["operations"]["engine"]["status"] == "completed"
    assert engine["operations"]["engine"]["firing_sec"] == 1.0
    assert abs(engine["fuel_balance_error_kg"]["a"]) < 1e-12
    assert all(
        abs(value) < 1e-9
        for value in engine["electrical_heat_balance_error_j"].values()
    )
    print(
        json.dumps(
            {
                "ok": True,
                "kind": kind,
                "metrics": life["metrics"],
                "actuation": engine,
                "geometry_revision": report["geometry_revision"],
                "mass_kg": report["mass_kg"],
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "scope": "center-of-mass engine with held body attitude; no torque/wheel or flight calibration",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "chemical")
