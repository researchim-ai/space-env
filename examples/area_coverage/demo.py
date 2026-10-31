"""No preloaded raw: polygon coverage through acquisition, transform and receipt."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.spacecraft_bus import bus_preset, pointing_quaternion
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    lifecycle_scenario,
    spacecraft_scenario,
)
from space_env.runtime.cli import apply_plan


def build_scenario():
    path = Path(__file__).resolve().parents[1] / "end_to_end_eo/demo.py"
    loader = importlib.util.spec_from_file_location("eo_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    definition = module.build_spec()
    template = MissionRuntime(
        lifecycle_scenario(definition, id="pointing", lineage="B04-B1")
    )
    definition["spacecraft_bus"] = {
        actor: bus_preset("synthetic-agile/1") for actor in ("a", "b")
    }
    for actor, config in definition["spacecraft_bus"].items():
        target = template._backend.acquisition_targets[
            "site" if actor == "a" else "far-side"
        ]
        config["attitude"]["initial_quaternion_wxyz"] = pointing_quaternion(
            template._backend.mission.ephemerides[actor], target, 0.0
        ).tolist()
        definition["operations"]["vehicles"][actor]["solar_mode"] = "body_fixed_bus"
    target = next(
        row for row in definition["acquisition_targets"] if row["id"] == "site"
    )
    lon, lat = target["lon_deg"], target["lat_deg"]
    shell = [
        [(lon + dx + 180) % 360 - 180, lat + dy]
        for dx, dy in (
            (-0.005, -0.005),
            (0.005, -0.005),
            (0.005, 0.005),
            (-0.005, 0.005),
            (-0.005, -0.005),
        )
    ]
    definition["lifecycle"]["areas_of_interest"] = [
        {
            "id": "customer-area",
            "anchor_lon_deg": lon,
            "anchor_lat_deg": lat,
            "shell": shell,
        }
    ]
    definition["lifecycle"]["requests"][0]["coverage"] = {
        "aoi_id": "customer-area",
        "min_fraction": 0.95,
    }
    definition["lifecycle"]["payloads"]["a"]["footprint"] = {
        "half_angle_x_deg": 1.0,
        "half_angle_y_deg": 1.0,
    }
    return spacecraft_scenario(definition, id="area-eo", lineage="B04-B1")


def plans():
    prefix = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "image",
                "commands": [
                    {
                        "actor_id": "a",
                        "command_id": "area-image",
                        "kind": "imaging",
                        "duration_sec": 2.0,
                        "request_id": "customer-a",
                    },
                    {
                        "actor_id": "b",
                        "command_id": "bad-image",
                        "kind": "imaging",
                        "duration_sec": 1.0,
                        "request_id": "infeasible",
                    },
                ],
            },
            {
                "op": "advance",
                "expected_revision": 1,
                "idempotency_key": "partial",
                "dt_sec": 1.0,
            },
        ]
    }
    continuation = {
        "operations": [
            {
                "op": "advance",
                "expected_revision": 2,
                "idempotency_key": "complete-image",
                "dt_sec": 1.0,
            },
            {
                "op": "submit",
                "expected_revision": 3,
                "idempotency_key": "cpu",
                "commands": [
                    {
                        "actor_id": "a",
                        "command_id": "features",
                        "kind": "transform",
                        "duration_sec": 2.0,
                        "job_id": "features-a",
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 4,
                "idempotency_key": "complete-cpu",
                "dt_sec": 2.0,
            },
            {
                "op": "submit",
                "expected_revision": 5,
                "idempotency_key": "radio",
                "commands": [
                    {
                        "actor_id": "a",
                        "command_id": "delivery",
                        "kind": "downlink",
                        "duration_sec": 1.0,
                        "product_id": "final-a",
                        "destination_id": "Ground",
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 6,
                "idempotency_key": "receipt",
                "dt_sec": 1.0,
            },
        ]
    }
    return prefix, continuation


def main(directory):
    scenario = build_scenario()
    prefix, continuation = plans()
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    apply_plan(model, continuation)
    model.save(directory / "final.json")
    restored = MissionRuntime.load(
        directory / "final.json", access=AccessPolicy(truth=True)
    )
    assert model.checkpoint() == restored.checkpoint()
    report = model.observe(view="truth")["backend"]["report"]["lifecycle"]
    assert report["metrics"] == {"fulfilled_requests": 1, "total_requests": 2}
    assert report["requests"]["infeasible"]["reason"] == "geometry"
    assert report["area_coverage"]["delivered"]["customer-area"]["fraction"] >= 0.95
    assert all(row["balance_error_bits"] == 0 for row in report["ledger"].values())
    print(
        json.dumps(
            {
                "ok": True,
                "metrics": report["metrics"],
                "area_coverage": report["area_coverage"],
                "ledger": report["ledger"],
                "acknowledgements": report["acknowledgements"],
                "geometry_manifest": model._backend.lifecycle.coverage_geometry_manifest,
                "scope": "sampled geometric coverage; not optical/SAR image quality",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
