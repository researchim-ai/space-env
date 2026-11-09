"""Inventory source availability, not physical fidelity or competitor superiority.

Run from a source checkout. AST indexing is structural; model depth and integration
are manually reviewed in docs/MODELING_COVERAGE_AND_ROADMAP.md. No simulator runs,
external installs, model calls or network requests are performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

FAMILIES = {
    "orbit_dynamics": [
        "reference/orbit.py",
        "reference/dynamic.py",
        "reference/frames.py",
    ],
    "maneuvers": [
        "reference/thrust.py",
        "reference/actuation.py",
        "reference/unloading.py",
    ],
    "attitude": [
        "reference/rigid_attitude.py",
        "reference/rigid_bus.py",
        "reference/spacecraft_bus.py",
    ],
    "relative_motion": ["core/relative.py", "tasks/goals.py"],
    "navigation_prediction": ["reference/covariance.py"],
    "conjunction_legacy": [
        "core/encounter.py",
        "core/probability.py",
        "envs/conjunction_env.py",
    ],
    "power": ["core/power.py", "reference/operations.py"],
    "thermal": ["reference/thermal_network.py", "reference/materials.py"],
    "earth_observation": [
        "reference/acquisition.py",
        "reference/area_coverage.py",
        "reference/observations.py",
    ],
    "payload_quality": ["reference/payload_quality.py"],
    "radio_ground": [
        "reference/radio.py",
        "reference/radio_network.py",
        "reference/booking.py",
    ],
    "inter_satellite_custody": ["reference/dtn.py"],
    "onboard_products": ["reference/lifecycle.py", "reference/dependencies.py"],
    "dynamic_planning": [
        "planning/sequence_lanes.py",
        "planning/sequence_recovery.py",
        "api/workspace_dynamic.py",
    ],
    "agent_data": [
        "tasks/session.py",
        "tasks/dataset.py",
        "llm/episode.py",
        "llm/corpus.py",
    ],
    "scaling": ["scaling/sequence.py", "scaling/full_orbit.py", "scaling/ensemble.py"],
    "external_adapters": [
        "integrations/orekit_prop.py",
        "integrations/spice_ephem.py",
        "integrations/sgp4_prop.py",
    ],
}


def inventory(root):
    package = root / "src/space_env"
    modules = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        modules[str(path.relative_to(package))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "classes": [
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            ],
            "functions": [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
        }
    missing = sorted(
        {p for files in FAMILIES.values() for p in files if p not in modules}
    )
    if missing:
        raise ValueError("Missing reviewed modules: " + ", ".join(missing))
    hashes = {p: row["sha256"] for p, row in modules.items()}
    receipts = {}
    for name in ["PHYSICAL_MISSION_GOALS_V1", "RELATIVE_MISSION_GOALS_V1"]:
        path = root / "docs/acceptance" / (name + ".json")
        receipt = json.loads(path.read_text())
        receipts[name] = {
            "receipt_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_sources_equal_current": receipt.get("model_sources") == hashes,
            "meaning": "Source equality only; tests/audits were not rerun by this inventory.",
        }
    body = {
        "schema": "space-env.modeling-inventory/1",
        "as_of": "2026-11-09",
        "scope": "Structural checkout inventory plus manually reviewed family paths. Availability is not integration, installed optional dependency availability, correctness, accuracy or superiority.",
        "module_count": len(modules),
        "family_count": len(FAMILIES),
        "families": FAMILIES,
        "modules": modules,
        "historical_receipts": receipts,
        "new_simulation_tests_run": False,
        "external_comparator_executed": False,
    }
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return body | {"inventory_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "module_count",
                    "family_count",
                    "historical_receipts",
                    "inventory_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
