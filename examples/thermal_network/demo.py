"""EO mission under nominal, hot-CPU or cold-start thermal conditions."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.runtime.cli import apply_plan


def thermal_model(mode="nominal"):
    def node(**kw):
        return {
            "initial_k": 285.0,
            "capacity_j_per_k": 20.0,
            "trip_k": 330.0,
            "resume_k": 310.0,
            **kw,
        }

    nodes = {
        "hull": node(),
        "cpu": node(capacity_j_per_k=10.0),
        "camera": node(),
        "panel": node(
            emission_area_m2=0.02, emissivity=0.8, solar_area_m2=0.01, absorptivity=0.6
        ),
        "radiator": node(
            emission_area_m2=0.02, emissivity=0.85, normal_body=[1.0, 0.0, 0.0]
        ),
    }
    if mode == "hot":
        nodes["cpu"] = node(capacity_j_per_k=0.2, trip_k=286.0, resume_k=285.1)
    if mode == "cold":
        nodes["panel"].update(
            initial_k=260.0,
            capacity_j_per_k=1.0,
            cold_trip_k=270.0,
            cold_resume_k=280.0,
            heater_power_w=20.0,
            heater_on_below_k=280.0,
            heater_off_above_k=285.0,
            resume_k=300.0,
            trip_k=310.0,
        )
    return {
        "model_id": "synthetic-5node/1",
        "parameter_status": "synthetic",
        "source": "declared lumped-node/radiation assumptions; not calibrated flight hardware",
        "nodes": nodes,
        "links": [
            {"a": "hull", "b": "cpu", "conductance_w_per_k": 0.1},
            {"a": "hull", "b": "camera", "conductance_w_per_k": 0.1},
            {"a": "hull", "b": "radiator", "conductance_w_per_k": 0.2},
        ],
        "routing": {
            "base": {"hull": 1.0},
            "cpu": {"cpu": 1.0},
            "payload": {"camera": 1.0},
            "radio": {"cpu": 1.0},
            "adcs": {"hull": 1.0},
            "propulsion": {"hull": 1.0},
            "conversion": {"hull": 1.0},
            "pv_extraction": {"panel": 1.0},
        },
    }


def build_scenario(mode="nominal"):
    if mode not in {"nominal", "hot", "cold"}:
        raise ValueError("mode must be nominal, hot or cold")
    path = Path(__file__).resolve().parents[1] / "resource_lanes/demo.py"
    loader = importlib.util.spec_from_file_location("thermal_lanes_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    previous, _ = module.build_scenario("wide")
    d = previous["definition"]
    first = d["lifecycle"]["requests"][1]
    second = copy.deepcopy(first)
    second.update(
        id="after-thermal", raw_product_id="raw-after", delivery_product_id="raw-after"
    )
    d["lifecycle"]["requests"] = [first, second]
    d["spacecraft_bus"]["a"]["thermal_network"] = thermal_model(mode)
    if mode == "cold":
        template = MissionRuntime(previous, access=AccessPolicy(truth=True))
        q = pointing_quaternion(
            template._backend.mission.ephemerides["a"],
            template._backend.acquisition_targets["site"],
            1.5,
        )
        d["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = q.tolist()
    operations = []

    def add(op, key, **kw):
        operations.append(
            {
                "op": op,
                "expected_revision": len(operations),
                "idempotency_key": key,
                **kw,
            }
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

    if mode == "cold":
        add("advance", "paid-cold-warmup", dt_sec=1.5)
        add(
            "inject",
            "recover-cold",
            events=[
                {
                    "id": "recovery",
                    "type": "spacecraft_recover",
                    "time_sec": 1.5,
                    "source": "thermal-demo",
                    "reason": "all node temperatures satisfy hysteresis after paid warmup",
                    "payload": {"actor": "a"},
                }
            ],
        )
    add("submit", "image-first", commands=[image("image-first", "customer-a")])
    add("advance", "partial-first-image", dt_sec=1.0)
    prefix_count = len(operations)
    add("advance", "finish-first-image", dt_sec=1.0)
    add(
        "submit",
        "cpu-first",
        commands=[
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "transform",
                "duration_sec": 2.0,
                "job_id": "features-a",
            }
        ],
    )
    add("advance", "partial-cpu", dt_sec=0.25)
    add("advance", "finish-cpu", dt_sec=1.75)
    if mode == "hot":
        add("advance", "cool-with-latched-safe", dt_sec=1.0)
    else:
        add("submit", "deliver-first", commands=[radio("radio-first", "final-a")])
        add("advance", "first-ack", dt_sec=0.5)
        add("submit", "image-second", commands=[image("image-second", "after-thermal")])
        add("advance", "finish-second", dt_sec=2.0)
        add("submit", "deliver-second", commands=[radio("radio-second", "raw-after")])
        add("advance", "second-ack", dt_sec=1.0)
    return spacecraft_scenario(d, id="thermal-" + mode, lineage="B04-E"), (
        {"operations": operations[:prefix_count]},
        {"operations": operations[prefix_count:]},
    )


def main(directory, mode="nominal"):
    scenario, (prefix, resume) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in [("scenario", scenario), ("plan", prefix), ("resume", resume)]:
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    m = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(m, prefix)
    m.save(directory / "prefix.json")
    m = MissionRuntime.load(directory / "prefix.json", access=AccessPolicy(truth=True))
    for operation in resume["operations"]:
        apply_plan(m, {"operations": [operation]})
        if operation["idempotency_key"] == "partial-cpu":
            m.save(directory / "cpu.json")
            m = MissionRuntime.load(
                directory / "cpu.json", access=AccessPolicy(truth=True)
            )
    m.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == m.checkpoint()
    )
    report = m.observe(view="truth")["backend"]["report"]
    bus = report["spacecraft_bus"]
    t = bus["states"]["a"]["thermal_network"]
    life = report["lifecycle"]
    assert life["metrics"] == {
        "fulfilled_requests": 0 if mode == "hot" else 2,
        "total_requests": 2,
    }
    assert max(abs(v) for v in t["node_balance_error_j"].values()) < 1e-6
    assert abs(bus["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-6
    if mode == "hot":
        assert (
            bus["states"]["a"]["safe"]
            and "overtemperature:cpu" in bus["states"]["a"]["safe_reasons"]
        )
        assert "final-a" not in life["artifacts"]
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "metrics": life["metrics"],
                "safe": bus["states"]["a"]["safe"],
                "safe_reasons": bus["states"]["a"]["safe_reasons"],
                "thermal_network": t,
                "bus_ledgers": bus["ledgers"],
                "job_state": life["jobs"]["features-a"],
                "scope": "5-node engineering thermal network with sampled radiation and paid heaters; no flight calibration",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "nominal")
