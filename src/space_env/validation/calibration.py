"""Q5-C foundation acceptance: published points and synthetic workflow, not flight fits."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from space_env.calibration import (
    calibrate,
    hardware_catalog,
    measurement_payload,
    sample_parameters,
    solar_resources,
    wheel_model,
)
from space_env.reference.rigid_attitude import (
    RigidAttitudeModel,
    RigidAttitudeState,
    advance_rigid_attitude,
)
from space_env.simulation.snapshot import digest

FIXTURE = (
    Path(__file__).resolve().parents[1] / "calibration/data/workflow_fixtures_v1.json"
)


FIXTURE_SHA256 = "85dbd577b3a34f591230c809c711bbfd8e4c8f6689c95592e8d79ec0b112bcec"


def calibration_workflow_report():
    catalog = hardware_catalog()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise ValueError(
            "packaged synthetic workflow fixture differs from frozen bytes"
        )
    fixture = json.loads(FIXTURE.read_text())
    if fixture.get(
        "schema"
    ) != "space-env.calibration-workflow-fixtures/1" or fixture.get("sha256") != digest(
        {k: v for k, v in fixture.items() if k != "sha256"}
    ):
        raise ValueError("corrupt synthetic calibration workflow fixture")
    fits = [calibrate(c["spec"]) for c in fixture["cases"]]
    if len(fits) != 3 or {r["model"] for r in fits} != {
        "pv_efficiency",
        "inverse_inertia",
        "inverse_heat_capacity",
    }:
        raise ValueError("every declared calibration workflow model is required")
    # Frozen positive/negative workflow checks, never rebrand these as measured data.
    negative = copy.deepcopy(fixture["cases"][0]["spec"])
    for row in negative["observations"]:
        if row["split"] == "holdout":
            row["y"] += 5.0
    negative["source"]["content_sha256"] = digest(measurement_payload(negative))
    bad = calibrate(negative)
    checks = [
        {"name": "synthetic_fits_pass", "passed": all(r["passed"] for r in fits)},
        {"name": "biased_holdout_rejected", "passed": not bad["passed"]},
        {
            "name": "holdout_does_not_change_fit",
            "passed": bad["parameter"] == fits[0]["parameter"],
        },
    ]
    adapters = []
    for id, p in catalog["profiles"].items():
        if p["kind"] == "reaction_wheel":
            adapter = wheel_model(
                [id] * 3,
                {
                    "inertia_kg_m2": [
                        [2.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0],
                        [0.0, 0.0, 4.0],
                    ],
                    "efficiency": 0.8,
                    "idle_power_w": 0.1,
                    "torque_loss_w_per_nm2": 1.0,
                    "rationale": "synthetic host inertia and motor losses for a numerical adapter smoke only",
                },
            )
            cfg = RigidAttitudeModel.model_validate(adapter["model"])
            duration = 2 * cfg.max_momentum_nms[2] / cfg.max_torque_nm[2]
            state, _ = advance_rigid_attitude(
                cfg,
                RigidAttitudeState(),
                duration,
                motor_torque_nm=(0.0, 0.0, cfg.max_torque_nm[2]),
            )
            passed = abs(state.wheel_momentum_nms[2] - cfg.max_momentum_nms[2]) <= 1e-10
            checks.append({"name": id + "-scenario-cap", "passed": passed})
            adapters.append(
                {
                    "profile": id,
                    "adapter_sha256": adapter["sha256"],
                    "model": adapter["model"],
                    "observed_momentum_nms": list(state.wheel_momentum_nms),
                    "scope": adapter["qualification_status"],
                }
            )
        else:
            adapter = solar_resources(
                id,
                {"memory_bits": 1e6, "battery_wh": 20.0},
                area_m2=0.1,
                packing_factor=0.9,
                assumption="BOL reference-point approximation with constant efficiency and ideal irradiance scaling",
            )
            checks.append(
                {
                    "name": id + "-efficiency-mapping",
                    "passed": abs(
                        adapter["vehicle_resources"]["solar_efficiency"]
                        - p["values_si"]["efficiency"] * 0.9
                    )
                    < 1e-15,
                }
            )
            adapters.append({"profile": id, "adapter": adapter})
    sensitivity = {
        "id": "workflow-sensitivity",
        "seed": 17,
        "samples": 16,
        "parameters": [
            {
                "id": "battery",
                "target": "battery_capacity_scale",
                "actor": "a",
                "mode": "uniform",
                "low": 0.8,
                "high": 1.1,
                "rationale": "assumed aging sensitivity, not manufacturer tolerance",
            },
            {
                "id": "inertia",
                "target": "body_inertia_scale",
                "actor": "a",
                "mode": "truncated_normal",
                "low": 0.8,
                "high": 1.2,
                "mean": 1.0,
                "sigma": 0.05,
                "rationale": "assumed geometry sensitivity, not a fitted hardware distribution",
            },
        ],
    }
    draws = sample_parameters(sensitivity)
    checks.append(
        {
            "name": "sensitivity_repeatability",
            "passed": draws == sample_parameters(sensitivity),
        }
    )
    checks.append(
        {
            "name": "all_sensitivity_draws_in_support",
            "passed": all(not r["errors"] for r in draws["members"]),
        }
    )
    # Independent analytic WLS information check, not coverage proof.
    pv = fixture["cases"][0]["spec"]
    train = [r for r in pv["observations"] if r["split"] == "train"]
    standard_error = 1 / np.sqrt(sum((r["x"] / r["sigma_y"]) ** 2 for r in train))
    checks.append(
        {
            "name": "known_noise_information",
            "passed": abs(
                float(standard_error) - fits[0]["parameter"]["standard_error"]
            )
            < 1e-15,
        }
    )
    learned_inertia = fits[1]["derived_reciprocal"]["estimate"]
    cfg = RigidAttitudeModel(
        model_id="synthetic-fit-runtime/1",
        source="synthetic fixture fitted inverse inertia; no hardware calibration",
        inertia_kg_m2=(
            (learned_inertia, 0.0, 0.0),
            (0.0, learned_inertia, 0.0),
            (0.0, 0.0, learned_inertia),
        ),
        wheel_inertia_kg_m2=(0.01,) * 3,
        max_torque_nm=(0.01,) * 3,
        max_momentum_nms=(0.1,) * 3,
    )
    state, _ = advance_rigid_attitude(
        cfg, RigidAttitudeState(), 1.0, external_torque_body_nm=(0.0, 0.0, 0.02)
    )
    checks.append(
        {
            "name": "synthetic_fitted_inertia_in_native_kernel",
            "passed": abs(
                state.omega_body_rad_s[2] - 0.02 * fits[1]["parameter"]["estimate"]
            )
            < 1e-10,
        }
    )
    from space_env.reference.thermal_network import (
        HEAT_SOURCES,
        ThermalNetworkSettings,
        advance_network,
    )

    config = ThermalNetworkSettings(
        model_id="synthetic-fit-thermal/1",
        parameter_status="synthetic",
        source="synthetic fitted heat capacity; isolated net heat assumption",
        nodes={
            "node": {
                "initial_k": 300.0,
                "capacity_j_per_k": fits[2]["derived_reciprocal"]["estimate"],
                "trip_k": 1000.0,
                "resume_k": 900.0,
            }
        },
        routing={k: {"node": 1.0} for k in HEAT_SOURCES},
        earth_environment=False,
    )
    heat = advance_network(
        config, {"node": 300.0}, 1.0, internal_w={"node": 5.0}, external_w={"node": 0.0}
    )
    checks.append(
        {
            "name": "synthetic_fitted_heat_capacity_in_native_kernel",
            "passed": abs(
                heat["temperatures_k"]["node"]
                - 300
                - 5 * fits[2]["parameter"]["estimate"]
            )
            < 1e-8,
        }
    )
    result = {
        "schema": "space-env.calibration-workflow-validation/1",
        "passed": all(c["passed"] for c in checks),
        "industrial_readiness": False,
        "hardware_calibration_complete": False,
        "qualification_status": "published-reference-points-and-synthetic-workflow-only",
        "checks": checks,
        "catalog_sha256": catalog["sha256"],
        "sources": catalog["sources"],
        "source_bytes_revalidated_in_run": False,
        "fits": fits,
        "negative_holdout": bad,
        "adapters": adapters,
        "sensitivity": draws,
        "remaining": [
            "independent laboratory/flight acquisition data with authenticated provenance",
            "measurement noise/error/correlation assessment and wider operating domain",
            "native coupled scenario ensemble qualification and customer mission holdout",
        ],
    }
    from space_env import __version__

    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"}
    }
    try:
        installed = version("space-env")
    except PackageNotFoundError:
        installed = None
    result["build"] = {
        "package_version": __version__,
        "installed_distribution_version": installed,
        "distribution_matches_source_version": installed == __version__,
        "files_sha256": hashes,
        "source_data_sha256": digest(hashes),
        "dependencies": {
            n: version(n) for n in ("numpy", "pydantic", "scipy", "pyerfa")
        },
        "scope": "package source/data inventory, not an author signature or native mission qualification ID",
    }
    result["sha256"] = digest(result)
    return result
