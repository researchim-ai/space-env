"""EO work around rotating/off-center chemical or electric thrust, with replay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.runtime.cli import apply_plan


def _load(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def build_scenario(kind="chemical"):
    root = Path(__file__).resolve().parents[1]
    body = _load("body_demo", root / "actuation/demo.py")
    wheels = _load("wheels_demo", root / "rigid_mission/demo.py")
    original, (prefix, continuation) = body.build_scenario(kind)
    wheel_scenario, _ = wheels.build_scenario("tuned")
    definition = original["definition"]
    definition["spacecraft_bus"]["a"]["attitude"]["rigid"] = wheel_scenario[
        "definition"
    ]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    definition["actuation"]["engines"]["a"]["rigid_coupling"] = {
        "lever_arm_body_m": [0.2, 0.0, 0.0],
        "max_mass_loss_fraction": 0.001,
    }
    for operation in prefix["operations"] + continuation["operations"]:
        for command in operation.get("commands", []):
            if command["kind"] == "body_burn":
                command["body_burn"]["thrust_n"] = 0.01
    return spacecraft_scenario(definition, id="rotating-" + kind, lineage="B04-D2b2"), (
        prefix,
        continuation,
    )


def main(directory, kind="chemical"):
    scenario, (prefix, resume) = build_scenario(kind)
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
    rigid = report["spacecraft_bus"]["states"]["a"]["rigid_attitude"]
    assert life["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert engine["operations"]["engine"]["status"] == "completed"
    assert abs(engine["fuel_balance_error_kg"]["a"]) < 1e-10
    assert abs(engine["electrical_heat_balance_error_j"]["engine"]) < 1e-8
    assert max(abs(v) for v in rigid["angular_momentum_balance_error_gcrs_nms"]) < 1e-8
    assert abs(rigid["kinetic_work_balance_error_j"]) < 1e-8
    assert any(
        row.get("direction_model") == "space-env.rotating-body-thrust/1"
        for row in engine["intervals"]
    )
    print(
        json.dumps(
            {
                "ok": True,
                "kind": kind,
                "metrics": life["metrics"],
                "actuation": engine,
                "rigid_attitude": rigid,
                "geometry_revision": report["geometry_revision"],
                "bus_ledgers": report["spacecraft_bus"]["ledgers"],
                "scope": "deterministic rotating thrust/lever moment; small mass loss with constant inertia, no exhaust qualification",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "chemical")
