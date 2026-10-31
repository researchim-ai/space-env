"""Command-driven nominal orbits with optional Cartesian uncertainty prediction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from space_env.reference.covariance import NavigationSettings
from space_env.runtime import AccessPolicy, MissionRuntime, dynamic_reference_scenario
from space_env.runtime.cli import apply_plan


def main(directory: Path):
    path = Path(__file__).with_name("dynamic_demo.py")
    loader = importlib.util.spec_from_file_location("dynamic_demo", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    definition = module.build_spec()
    settings = NavigationSettings(
        initial_covariance_si=np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01]).tolist(),
        acceleration_psd_m2_s3=(np.eye(3) * 1e-6).tolist(),
        impulse_velocity_cov_m2_s2=(np.eye(3) * 0.0025).tolist(),
    )
    for orbit in definition["orbits"].values():
        orbit["navigation"] = settings.model_dump(mode="json")
    scenario = dynamic_reference_scenario(
        definition, id="navigation-demo", lineage="B02-Q1"
    )
    scenario["required_capabilities"] = ["state.covariance_propagation", "state.stm"]
    prefix = {
        "operations": [
            {
                "op": "submit",
                "expected_revision": 0,
                "idempotency_key": "kick",
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
                "idempotency_key": "half",
                "dt_sec": 2.0,
            },
        ]
    }
    resume = {
        "operations": [
            {
                "op": "cancel",
                "expected_revision": 3,
                "idempotency_key": "cancel",
                "command_id": "engine",
            },
            {
                "op": "advance",
                "expected_revision": 4,
                "idempotency_key": "coast",
                "dt_sec": 3.0,
            },
        ]
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("scenario", scenario), ("plan", prefix), ("resume", resume)):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    runtime = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    apply_plan(runtime, prefix)
    runtime.save(directory / "prefix.json")
    restored = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    apply_plan(restored, resume)
    restored.save(directory / "final.json")
    replayed = MissionRuntime.load(
        directory / "final.json", access=AccessPolicy(truth=True)
    )
    assert replayed.checkpoint() == restored.checkpoint()
    rows = {}
    for actor, view in restored.observe()["actors"].items():
        covariance = np.asarray(view["orbit"]["covariance_si"])
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
        rows[actor] = {
            "position_stddev_rss_m": float(np.sqrt(np.trace(covariance[:3, :3]))),
            "velocity_stddev_rss_m_s": float(np.sqrt(np.trace(covariance[3:, 3:]))),
        }
    print(
        json.dumps(
            {
                "ok": True,
                "time_sec": restored.time_sec,
                "uncertainty": rows,
                "meaning": "linearized prediction; no measurement update or sampled execution error",
                "qualification_status": "needs_external_evidence",
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
