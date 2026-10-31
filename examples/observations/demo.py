"""Same-spacecraft revisit/stereo or two-spacecraft ground delivery bundle."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.cli import apply_plan


def build_scenario(mode="revisit"):
    if mode not in {"revisit", "stereo-fleet"}:
        raise ValueError("mode must be revisit or stereo-fleet")
    path = Path(__file__).resolve().parents[1] / "payload_quality/demo.py"
    loader = importlib.util.spec_from_file_location("quality_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, _ = module.build_scenario("optical")
    scenario.update(id=f"observations-{mode}", lineage="B04-B3")
    spec = scenario["definition"]["lifecycle"]
    first = spec["requests"][0]
    second = copy.deepcopy(first)
    second.update(id="customer-b", raw_product_id="raw-b")
    spec["requests"] = [first, second]
    spec["jobs"][0]["inputs"] = ["raw-a", "raw-b"]
    if mode == "revisit":
        first["observations"] = {
            "revisit": {"min_interval_sec": 0.5, "max_interval_sec": 10.0},
            "stereo": {
                "min_convergence_deg": 0.1,
                "max_convergence_deg": 10.0,
                "max_interval_sec": 10.0,
            },
        }
        spec["jobs"][0]["inputs"] = ["raw-a", "raw-b"]
    else:
        template = MissionRuntime(scenario)
        state = template._backend.mission.ephemerides["a"].state(3.0)
        scenario["definition"]["orbits"]["b"]["initial"].update(
            r_m=state[:3].tolist(), v_m_s=state[3:].tolist()
        )
        first["delivery_product_id"] = "raw-a"
        first["observations"] = {
            "stereo": {
                "min_convergence_deg": 0.1,
                "max_convergence_deg": 10.0,
                "max_interval_sec": 2.0,
                "different_actors": True,
            },
            "support_product_ids": ["raw-b"],
        }
        second.update(eligible_actors=["b"], delivery_product_id="raw-b")
        spec["jobs"] = []
        spec["payloads"]["b"] = copy.deepcopy(spec["payloads"]["a"])
        scenario["definition"]["acquisition_sensors"]["b"] = copy.deepcopy(
            scenario["definition"]["acquisition_sensors"]["a"]
        )
        template = MissionRuntime(scenario)
        target = AcquisitionTarget.model_validate(
            next(
                t
                for t in scenario["definition"]["acquisition_targets"]
                if t["id"] == "site"
            )
        )
        scenario["definition"]["spacecraft_bus"]["b"]["attitude"][
            "initial_quaternion_wxyz"
        ] = pointing_quaternion(
            template._backend.mission.ephemerides["b"], target, 0.0
        ).tolist()
    return scenario, plans(mode)


def plans(mode):
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

    def image(actor, request, id):
        return {
            "command_id": id,
            "actor_id": actor,
            "kind": "imaging",
            "request_id": request,
            "duration_sec": 2.0,
        }

    def radio(actor, product, id):
        return {
            "command_id": id,
            "actor_id": actor,
            "kind": "downlink",
            "product_id": product,
            "destination_id": "Ground",
            "duration_sec": 1.0,
        }

    commands = [image("a", "customer-a", "image-a")]
    if mode == "stereo-fleet":
        commands.append(image("b", "customer-b", "image-b"))
    add("submit", "images", commands=commands)
    add("advance", "partial-images", dt_sec=1.0)
    add("advance", "complete-images", dt_sec=1.0)
    if mode == "revisit":
        add("submit", "second-image", commands=[image("a", "customer-b", "image-b")])
        add("advance", "second-complete", dt_sec=2.0)
        add(
            "submit",
            "features",
            commands=[
                {
                    "command_id": "cpu",
                    "actor_id": "a",
                    "kind": "transform",
                    "job_id": "features-a",
                    "duration_sec": 2.0,
                }
            ],
        )
        add("advance", "cpu-time", dt_sec=2.0)
    if mode == "stereo-fleet":
        add("advance", "station-reopens", dt_sec=2.0)
    add(
        "submit",
        "radio-a",
        commands=[
            radio("a", "raw-a" if mode == "stereo-fleet" else "final-a", "radio-a")
        ],
    )
    add("advance", "partial-radio", dt_sec=0.25)
    add("advance", "finish-radio", dt_sec=0.75)
    if mode == "stereo-fleet":
        add("submit", "radio-b", commands=[radio("b", "raw-b", "radio-b")])
        add("advance", "support-time", dt_sec=1.0)
    return {"operations": operations[:2]}, {"operations": operations[2:]}


def main(directory, mode="revisit"):
    scenario, (prefix, continuation) = build_scenario(mode)
    directory.mkdir(parents=True, exist_ok=True)
    for name, row in (
        ("scenario", scenario),
        ("plan", prefix),
        ("resume", continuation),
    ):
        (directory / f"{name}.json").write_text(json.dumps(row, allow_nan=False))
    model = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(model, prefix)
    model.save(directory / "prefix.json")
    model = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    for operation in continuation["operations"]:
        apply_plan(model, {"operations": [operation]})
        if operation["idempotency_key"] == "partial-radio":
            model.save(directory / "partial-radio.json")
            model = MissionRuntime.load(
                directory / "partial-radio.json", access=AccessPolicy(truth=True)
            )
    model.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )
    report = model.observe(view="truth")["backend"]["report"]["lifecycle"]
    assert report["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert report["observation_requirements"]["customer-a"]["delivered_and_fulfilled"]
    assert all(row["balance_error_bits"] == 0 for row in report["ledger"].values())
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "metrics": report["metrics"],
                "requirements": report["observation_requirements"],
                "ledger": report["ledger"],
                "acknowledgements": report["acknowledgements"],
                "scope": "sampled static-surface geometry; no stereo reconstruction or flight calibration",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "revisit")
