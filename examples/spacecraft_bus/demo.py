"""Coupled spacecraft: finite slew, checkpoint, acquisition, CPU and delivery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from space_env.core.attitude import align_z, boresight
from space_env.core.frames import ecef_to_geodetic
from space_env.reference import EOPSeries, Epoch, FrameContext
from space_env.reference.spacecraft_bus import bus_preset, pointing_quaternion
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    lifecycle_scenario,
    spacecraft_scenario,
)
from space_env.runtime.cli import apply_plan


def build_spec():
    path = Path(__file__).resolve().parents[1] / "unified_runtime/dynamic_demo.py"
    loader = importlib.util.spec_from_file_location("dynamic_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = module.build_spec()
    spec["operations"]["products"] = []
    eop, initial = spec["eop"], spec["orbits"]["a"]["initial"]
    context = FrameContext(
        EOPSeries(
            eop["rows"],
            source=eop["source"],
            allow_predictions=eop["allow_predictions"],
        )
    )
    lat, lon, _ = ecef_to_geodetic(
        context.matrix(Epoch.utc(initial["epoch_utc"])) @ initial["r_m"]
    )
    spec["acquisition_targets"] = [
        {"id": "site", "lat_deg": lat, "lon_deg": lon},
        {"id": "far-side", "lat_deg": -lat, "lon_deg": ((lon + 360) % 360) - 180},
    ]
    spec["acquisition_sensors"] = {a: {"max_off_nadir_deg": 35.0} for a in ("a", "b")}
    spec["lifecycle"] = {
        "payloads": {a: {"power_w": 10.0} for a in ("a", "b")},
        "requests": [
            {
                "id": "customer-a",
                "target_id": "site",
                "eligible_actors": ["a"],
                "deadline_sec": 20.0,
                "acquisition_sec": 2.0,
                "raw_product_id": "raw-a",
                "raw_size_bits": 1000000,
                "delivery_product_id": "final-a",
                "destination_id": "Ground",
            },
            {
                "id": "infeasible",
                "target_id": "far-side",
                "eligible_actors": ["b"],
                "deadline_sec": 20.0,
                "acquisition_sec": 1.0,
                "raw_product_id": "raw-b",
                "raw_size_bits": 1000000,
                "delivery_product_id": "raw-b",
                "destination_id": "Ground",
            },
        ],
        "jobs": [
            {
                "id": "features-a",
                "inputs": ["raw-a"],
                "outputs": [{"id": "final-a", "size_bits": 500000}],
                "work_sec": 1.5,
                "switching_sec": 0.5,
                "workspace_bits": 100000,
                "power_w": 5.0,
                "transformer": "geometry-features/1",
            }
        ],
    }
    return spec


def main(directory: Path):
    definition = build_spec()
    template = MissionRuntime(lifecycle_scenario(definition, id="pointing", lineage="B04-A"))
    definition["spacecraft_bus"] = {a: bus_preset("synthetic-agile/1") for a in ("a", "b")}
    goals = {}
    for actor, config in definition["spacecraft_bus"].items():
        goals[actor] = pointing_quaternion(template._backend.mission.ephemerides[actor],
                                          template._backend.acquisition_targets["site" if actor == "a" else "far-side"], 0.0)
        direction = boresight(goals[actor])
        axis = np.cross(direction, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(direction, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(10.0)
        offset = direction * np.cos(angle) + np.cross(axis, direction) * np.sin(angle)
        config["attitude"]["initial_quaternion_wxyz"] = (align_z(offset) if actor == "a" else goals[actor]).tolist()
        config["attitude"]["max_rate_deg_s"] = 10.0
        definition["operations"]["vehicles"][actor]["solar_mode"] = "body_fixed_bus"
    scenario = spacecraft_scenario(definition, id="coupled-spacecraft", lineage="B04-A")
    prefix = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "acquire",
                "commands": [
                    {
                        "command_id": "image-a",
                        "actor_id": "a",
                        "kind": "imaging",
                        "duration_sec": 2.0,
                        "request_id": "customer-a",
                    },
                    {
                        "command_id": "image-b",
                        "actor_id": "b",
                        "kind": "imaging",
                        "duration_sec": 1.0,
                        "request_id": "infeasible",
                    },
                ],
            },
            {
                "op": "advance",
                "expected_revision": 1,
                "idempotency_key": "partial-image",
                "dt_sec": 1.0,
            },
        ]
    }
    continuation = {
        "operations": [
            {
                "op": "advance",
                "expected_revision": 2,
                "idempotency_key": "finish-image",
                "dt_sec": 1.0,
            },
            {
                "op": "submit",
                "expected_revision": 3,
                "idempotency_key": "transform",
                "commands": [
                    {
                        "command_id": "cpu",
                        "actor_id": "a",
                        "kind": "transform",
                        "duration_sec": 2.0,
                        "job_id": "features-a",
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 4,
                "idempotency_key": "partial-cpu",
                "dt_sec": 1.0,
            },
            {
                "op": "advance",
                "expected_revision": 5,
                "idempotency_key": "finish-cpu",
                "dt_sec": 1.0,
            },
            {
                "op": "submit",
                "expected_revision": 6,
                "idempotency_key": "downlink",
                "commands": [
                    {
                        "command_id": "radio",
                        "actor_id": "a",
                        "kind": "downlink",
                        "duration_sec": 1.0,
                        "product_id": "final-a",
                        "destination_id": "Ground",
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 7,
                "idempotency_key": "deliver",
                "dt_sec": 1.0,
            },
        ]
    }
    old_operations = prefix["operations"] + continuation["operations"]
    for operation in old_operations:
        operation["expected_revision"] += 3
    prefix = {"operations": [
        {"op": "submit", "expected_revision": 0, "idempotency_key": "slew",
         "commands": [{"command_id": "point-a", "actor_id": "a", "kind": "slew",
                       "duration_sec": 3.0, "slew": {"frame": "GCRS",
                         "quaternion_wxyz": goals["a"].tolist()}}]},
        {"op": "advance", "expected_revision": 1, "idempotency_key": "partial-slew", "dt_sec": 0.5}]}
    continuation = {"operations": [
        {"op": "advance", "expected_revision": 2, "idempotency_key": "finish-slew", "dt_sec": 2.5},
        *old_operations]}
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    initial_report = model.observe(view="truth")["backend"]["report"]["lifecycle"]
    assert not initial_report["artifacts"] and all(
        row["initial"] == 0 for row in initial_report["ledger"].values()
    )
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for row in continuation["operations"]:
        apply_plan(model, {"operations": [row]})
        if row["idempotency_key"] == "partial-cpu":
            model.save(directory / "processing.json")
            model = MissionRuntime.load(
                directory / "processing.json", access=AccessPolicy(truth=True)
            )
    model.save(directory / "final.json")
    restored = MissionRuntime.load(
        directory / "final.json", access=AccessPolicy(truth=True)
    )
    assert restored.checkpoint() == model.checkpoint()
    report = model.observe(view="truth")["backend"]["report"]["lifecycle"]
    assert report["metrics"] == {"fulfilled_requests": 1, "total_requests": 2}
    assert report["requests"]["infeasible"]["reason"] == "geometry"
    assert "raw-b" not in report["artifacts"]
    assert all(abs(r["balance_error_bits"]) < 1e-8 for r in report["ledger"].values())
    bus = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]
    assert all(abs(row[k]) < 1e-7 for row in bus["ledgers"].values() for k in row if "error" in k)
    print(
        json.dumps(
            {
                "ok": True,
                "profile": model.report()["profile"],
                "metrics": report["metrics"],
                "spacecraft_bus": bus,
                "ledger": report["ledger"],
                "acknowledgements": report["acknowledgements"],
                "failed_request_reason": report["requests"]["infeasible"]["reason"],
                "representation": report["representation"],
                "qualification_status": "needs_external_evidence",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
