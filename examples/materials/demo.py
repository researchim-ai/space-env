"""Native opt-in variable heat capacity, cold replay and domain-escape rollback."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from space_env.reference.materials import material_evidence
from space_env.reference.thermal_network import HEAT_SOURCES
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve().parents[1] / "rigid_mission/demo.py"
    loader = importlib.util.spec_from_file_location("material_synthetic_host", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenario, _ = module.build_scenario("tuned")
    scenario["id"] = "material-enthalpy-native"
    scenario["lineage"] = (
        "published material cp; synthetic homogeneous node, not flight calibrated"
    )
    scenario["definition"]["operations"]["vehicles"]["a"].update(
        solar_area_m2=0.0, base_power_w=10.0
    )
    scenario["definition"]["spacecraft_bus"]["a"]["thermal_network"] = {
        "model_id": "mixed-enthalpy-demo/1",
        "parameter_status": "synthetic",
        "source": "NIST cp; synthetic mass, homogeneous lump and internal heat routing",
        "earth_environment": False,
        "nodes": {
            "body": {
                "initial_k": 250.0,
                "resume_k": 290.0,
                "trip_k": 295.0,
                "material_capacity": {
                    "mass_kg": 0.01,
                    "assumption": "synthetic homogeneous Al6061-T6 thermal lump",
                },
            }
        },
        "links": [],
        "routing": {k: {"body": 1.0} for k in HEAT_SOURCES},
    }
    evidence = material_evidence()["source"]
    scenario["sources"].append(
        {
            "id": "nist-al6061-t6-cp",
            "kind": evidence["kind"],
            "sha256": evidence["content_sha256"],
            "uri": evidence["uri"],
            "license": "source referenced; original HTML not redistributed",
        }
    )
    atomic_write(directory / "scenario.json", canonical_json(scenario))
    runtime = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    runtime.advance(0.3, expected_revision=runtime.revision, idempotency_key="prefix")
    runtime.save(directory / "prefix.json")
    cold = MissionRuntime.load(
        directory / "prefix.json", access=AccessPolicy(truth=True)
    )
    assert cold.checkpoint() == runtime.checkpoint()
    for model in (runtime, cold):
        model.advance(
            0.4, expected_revision=model.revision, idempotency_key="continuation"
        )
    assert runtime.checkpoint() == cold.checkpoint()
    cold.save(directory / "final.json")
    assert (
        MissionRuntime.load(
            directory / "final.json", access=AccessPolicy(truth=True)
        ).checkpoint()
        == cold.checkpoint()
    )
    view = cold.observe("a")["actors"]["a"]["spacecraft_bus"]["thermal_network"]
    assert 250.0 < view["temperatures_k"]["body"] < 295.0
    assert max(abs(x) for x in view["node_balance_error_j"].values()) < 1e-5
    negative = copy.deepcopy(scenario)
    node = negative["definition"]["spacecraft_bus"]["a"]["thermal_network"]["nodes"][
        "body"
    ]
    node.update(initial_k=299.9, resume_k=305.0, trip_k=310.0)
    node["material_capacity"]["mass_kg"] = 1e-5
    bad = MissionRuntime(negative, access=AccessPolicy(truth=True))
    before = bad.checkpoint()
    try:
        bad.advance(
            0.5,
            expected_revision=bad.revision,
            idempotency_key="outside-material-domain",
        )
    except RuntimeContractError as exc:
        assert "no extrapolation" in str(exc)
    else:
        raise AssertionError("material domain escape unexpectedly accepted")
    assert bad.checkpoint() == before
    result = {
        "schema": "space-env.material-native-smoke/1",
        "passed": True,
        "native_restore_replay": True,
        "domain_escape_atomic_rollback": True,
        "thermal_observation": view,
        "scenario_sha256": digest(scenario),
        "industrial_readiness": False,
        "hardware_calibration_complete": False,
        "scope": "short synthetic host with a published material correlation; no mission service or flight qualification",
    }
    result["sha256"] = digest(result)
    atomic_write(directory / "receipt.json", canonical_json(result))
    print(
        json.dumps(
            {
                "passed": True,
                "native_restore_replay": True,
                "domain_escape_atomic_rollback": True,
                "sha256": result["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
