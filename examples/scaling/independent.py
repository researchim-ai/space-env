"""Frozen independent geographic sites, multi-plane delivery and negative cases."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.scaling.independent import assess_geometry, independent_workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def configuration():
    return {
        "id": "independent-sites",
        "planes": [
            {
                "id": "plane-a",
                "inclination_deg": 55.0,
                "raan_deg": 0.0,
                "phases_deg": [0.0, 8.0],
            },
            {
                "id": "plane-b",
                "inclination_deg": 75.0,
                "raan_deg": 120.0,
                "phases_deg": [90.0, 98.0],
            },
        ],
        "stations": [
            {"name": "equator", "lat_deg": 0.0, "lon_deg": 0.0},
            {"name": "north", "lat_deg": 45.0, "lon_deg": 30.0},
            {"name": "south", "lat_deg": -30.0, "lon_deg": 150.0},
        ],
        "horizon_sec": 6000.0,
        "min_delivered": 1,
    }


def manifest(*, small=False):
    cfg = configuration()
    if small:
        for p in cfg["planes"]:
            p["phases_deg"] = p["phases_deg"][:1]
    workloads = []
    assessments = {}
    for mode in ["nominal", "outage", "no-contact"]:
        c = copy.deepcopy(cfg)
        c["id"] = "independent-" + mode
        if mode == "outage":
            c.update(mode="all_outage", min_delivered=0)
        if mode == "no-contact":
            c.update(
                stations=[
                    {
                        "name": "polar-high",
                        "lat_deg": 90.0,
                        "lon_deg": 0.0,
                        "min_elevation_deg": 89.0,
                    }
                ],
                min_delivered=0,
            )
        w = independent_workload(c)
        w["comparison_group"] = mode
        assessments[mode] = assess_geometry(w["scenario"])
        workloads.append(w)
        restored = copy.deepcopy(w)
        restored.update(
            id=w["id"] + "-restored",
            resume_after_decisions=20,
            resume_in_new_process=True,
        )
        workloads.append(restored)
    return {
        "id": "independent-fixed-sites/1",
        "workloads": workloads,
        "workers": 2,
    }, assessments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    if args.directory.exists() and any(args.directory.iterdir()):
        raise ValueError("example output must be empty")
    value, assessment = manifest(small=args.small)
    args.directory.mkdir(parents=True, exist_ok=True)
    atomic_write(args.directory / "assessments.json", canonical_json(assessment))
    result = run_scale(value, args.directory / "native")
    receipt = {
        "schema": "space-env.independent-site-example/1",
        "passed": result["ok"],
        "assessment_sha256": digest(assessment),
        "native_result_sha256": result["sha256"],
        "scope": "synthetic independent site/plane delivery, not closed-loop pointing/EO or flight calibration",
    }
    receipt["sha256"] = digest(receipt)
    atomic_write(args.directory / "receipt.json", canonical_json(receipt))
    print(canonical_json(receipt))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
