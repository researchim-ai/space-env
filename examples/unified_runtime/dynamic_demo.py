"""Two command-driven GCRS satellites: impulse, partial burn, resume and delivery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, dynamic_reference_scenario
from space_env.runtime.cli import apply_plan


def build_spec():
    path = Path(__file__).resolve().parents[1] / "reference_operations/demo.py"
    loader = importlib.util.spec_from_file_location("physical_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    fixed = module.build_spec()
    first = next(iter(fixed["ephemerides"].values()))
    return {
        "schema": "space-env.reference-dynamic-spec/1",
        "horizon_sec": 30.0,
        "eop": first["manifest"]["eop"],
        "orbits": {
            a: {
                "initial": {
                    "frame": "GCRS",
                    "epoch_utc": record["epoch_utc"],
                    "r_m": record["states_si"][0][:3],
                    "v_m_s": record["states_si"][0][3:],
                },
                "forces": {"j2": True},
                "propulsion": {
                    "dry_mass_kg": 90.0,
                    "propellant_kg": 10.0,
                    "isp_s": 300.0,
                    "max_thrust_n": 2.0,
                },
                "integration": {"sample_sec": 1.0, "max_step_sec": 5.0},
            }
            for a, record in fixed["ephemerides"].items()
        },
        "operations": {
            k: v for k, v in fixed.items() if k not in {"ephemerides", "horizon_sec"}
        },
    }


def main(directory: Path):
    scenario = dynamic_reference_scenario(
        build_spec(), id="maneuver-demo", lineage="B02"
    )
    prefix = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "impulse",
                "commands": [
                    {
                        "command_id": "kick",
                        "actor_id": "a",
                        "kind": "impulse",
                        "duration_sec": 0.0,
                        "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 0.0, 10.0]},
                    }
                ],
            },
            {
                "op": "submit",
                "expected_revision": 1,
                "idempotency_key": "burn",
                "commands": [
                    {
                        "command_id": "engine",
                        "actor_id": "a",
                        "kind": "finite_burn",
                        "duration_sec": 4.0,
                        "burn": {
                            "frame": "GCRS",
                            "direction_gcrs": [0.0, 1.0, 0.0],
                            "thrust_n": 2.0,
                        },
                    }
                ],
            },
            {
                "op": "advance",
                "expected_revision": 2,
                "idempotency_key": "partial",
                "dt_sec": 1.0,
            },
        ]
    }
    resume = {
        "operations": [
            {
                "op": "advance",
                "expected_revision": 3,
                "idempotency_key": "finish-burn",
                "dt_sec": 3.0,
            },
            {
                "op": "submit",
                "expected_revision": 4,
                "idempotency_key": "downlinks",
                "commands": [
                    {
                        "command_id": f"radio-{actor}",
                        "actor_id": actor,
                        "kind": "downlink",
                        "duration_sec": 5.0,
                        "destination_id": "Ground",
                    }
                    for actor in ("a", "b")
                ],
            },
            {
                "op": "advance",
                "expected_revision": 5,
                "idempotency_key": "deliver",
                "dt_sec": 5.0,
            },
        ]
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("scenario", scenario), ("plan", prefix), ("resume", resume)):
        (directory / f"{name}.json").write_text(json.dumps(data, allow_nan=False))
    runtime = MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))
    apply_plan(runtime, prefix)
    runtime.save(directory / "prefix.json")
    restored = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    apply_plan(restored, resume)
    restored.save(directory / "final.json")
    replayed = MissionRuntime.load(directory / "final.json")
    assert replayed.report() == restored.report()
    assert restored.report()["metrics"]["on_time_bits"] == 5e6
    mass = restored.observe("a")["actors"]["a"]["orbit"]["mass_kg"]
    assert mass < 100
    print(
        json.dumps(
            {
                "ok": True,
                "report": restored.report(),
                "mass_a_kg": mass,
                "qualification": "needs_external_evidence",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
