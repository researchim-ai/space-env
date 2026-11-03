"""Frozen numerical evidence for variable-capacity thermal nodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from space_env.reference.materials import (
    material_evidence,
    specific_enthalpy_delta_j_per_kg,
    specific_heat_j_per_kg_k,
)
from space_env.reference.thermal_network import (
    HEAT_SOURCES,
    ThermalNetworkSettings,
    advance_network,
)
from space_env.simulation.snapshot import digest

FIXTURE = (
    Path(__file__).resolve().parents[1] / "reference/fixtures/material_vectors_v1.json"
)
FIXTURE_SHA256 = "f2277f0967503b40c91400f8f6a632eee40957c5ec0db73d38028352fe8cb148"


def material_validation_report():
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise ValueError("material fixture bytes changed")
    fixture = json.loads(FIXTURE.read_text())
    if fixture["sha256"] != digest({k: v for k, v in fixture.items() if k != "sha256"}):
        raise ValueError("material fixture body changed")
    checks = []

    def check(case, metric, measured, tolerance):
        checks.append(
            {
                "case": case,
                "metric": metric,
                "measured": float(measured),
                "tolerance": tolerance,
                "passed": bool(abs(measured) <= tolerance),
            }
        )

    for p in fixture["points"]:
        check(
            str(p["temperature_k"]),
            "cp_relative_error",
            specific_heat_j_per_kg_k(p["temperature_k"]) / p["cp_j_per_kg_k"] - 1,
            5e-10,
        )
    for p in fixture["energies"]:
        check(
            f"{p['start_k']}-{p['end_k']}",
            "enthalpy_error_j_per_kg",
            specific_enthalpy_delta_j_per_kg(p["start_k"], p["end_k"])
            - p["enthalpy_j_per_kg"],
            1e-4,
        )
    # Heat exactly equal to independent integral: the endpoint is known without
    # importing native quadrature into the oracle or solving the native ODE twice.
    for solver in ("DOP853", "Radau"):
        for p in fixture["energies"][:5]:
            config = ThermalNetworkSettings(
                model_id="material-isolated-oracle/1",
                parameter_status="synthetic",
                source="published cp correlation; synthetic isolated mass and heat input",
                solver=solver,
                earth_environment=False,
                max_step_sec=0.1,
                nodes={
                    "body": {
                        "initial_k": float(p["start_k"]),
                        "trip_k": 300.0,
                        "resume_k": 299.0,
                        "material_capacity": {
                            "mass_kg": 1.0,
                            "assumption": "isolated homogeneous alloy specimen",
                        },
                    }
                },
                routing={k: {"body": 1.0} for k in HEAT_SOURCES},
            )
            # Endpoint at 300 K would allow RK stages beyond the valid domain;
            # the final [250,300] integral is checked above, not used as an ODE run.
            if p["end_k"] == 300:
                continue
            r = advance_network(
                config,
                {"body": float(p["start_k"])},
                1.0,
                internal_w={"body": p["enthalpy_j_per_kg"]},
                external_w={"body": 0.0},
            )
            case = f"{solver}/{p['start_k']}-{p['end_k']}"
            check(
                case, "endpoint_error_k", r["temperatures_k"]["body"] - p["end_k"], 2e-5
            )
            check(
                case, "node_energy_residual_j", r["node_balance_error_j"]["body"], 1e-4
            )
    from space_env import __version__

    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"}
    }
    result = {
        "schema": "space-env.material-validation/1",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "material": material_evidence(),
        "fixture_sha256": FIXTURE_SHA256,
        "build": {
            "package_version": __version__,
            "files_sha256": hashes,
            "source_data_sha256": digest(hashes),
        },
        "industrial_readiness": False,
        "hardware_calibration_complete": False,
        "qualification_status": "published-correlation-numerical-verification-only",
        "budget_basis": "fixed numerical comparison budgets, not physical prediction uncertainty",
        "unqualified": [
            "raw measurement calibration and specimen variability",
            "outside 4--300 K",
            "temperature-dependent conductivity, phase changes and thermoelasticity",
            "flight hardware and full mission qualification",
        ],
    }
    result["sha256"] = digest(result)
    return result
