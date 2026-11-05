"""Predeclared mission families: no outcome-dependent geometry or split search."""

from __future__ import annotations

import copy

from space_env.scaling.coupled import coupled_workload
from space_env.simulation.snapshot import digest

FAMILIES = (
    "nominal",
    "outage",
    "deadline",
    "energy",
    "processing",
    "contention",
    "catalogue",
    "long",
)
GROUPS = (
    ("train", "train-orbits", 55.0, 73.0, 20.0, 140.0, 17.0, 43.0),
    ("development", "dev-orbits", 61.0, 81.0, -55.0, 73.0, 101.0, 211.0),
    ("test", "test-orbits-a", 49.0, 68.0, 90.0, -110.0, 155.0, 301.0),
    ("test", "test-orbits-b", 63.0, 84.0, -27.0, 161.0, 73.0, 253.0),
)


def release_suite(*, smoke=False):
    """32 fixed cases, or five explicitly incomplete smoke cases; synthetic only."""
    cases = []
    for index, (split, group, i1, i2, r1, r2, p1, p2) in enumerate(GROUPS):
        for family in FAMILIES:
            if smoke and not (family == "nominal" or family == "outage" and index == 2):
                continue
            four = family == "catalogue"
            cfg = {
                "id": f"release-{index}-{family}",
                "planes": [
                    {
                        "id": "a",
                        "inclination_deg": i1,
                        "raan_deg": r1,
                        "phases_deg": [p1, (p1 + 120) % 360] if four else [p1],
                    },
                    {
                        "id": "b",
                        "inclination_deg": i2,
                        "raan_deg": r2,
                        "phases_deg": [p2, (p2 + 120) % 360] if four else [p2],
                    },
                ],
                "targets": [
                    {
                        "id": f"t{k:02d}",
                        "lat_deg": float((-1) ** k * 15),
                        "lon_deg": float(-165 + 30 * k),
                    }
                    for k in range(12)
                ],
                "stations": [
                    {
                        "name": f"g{k:02d}",
                        "lat_deg": float((-1) ** k * 10),
                        "lon_deg": float(-165 + 30 * k),
                    }
                    for k in range(12)
                ],
                "horizon_sec": 600.0,
                "decision_sec": 5.0,
                "mode": "all_outage" if family == "outage" else "nominal",
            }
            w = coupled_workload(cfg)
            d = w["scenario"]["definition"]
            for bus in d["spacecraft_bus"].values():
                bus["attitude"].pop("rigid", None)
            if family == "deadline":
                for request in d["lifecycle"]["requests"]:
                    request["deadline_sec"] = 20.0
            if family == "energy":
                for vehicle in d["operations"]["vehicles"].values():
                    vehicle["battery_wh"] = 0.2
            if family == "processing":
                for job in d["lifecycle"]["jobs"]:
                    job["work_sec"] = 45.0
            if family == "contention":
                for request in d["lifecycle"]["requests"]:
                    request["destination_id"] = "g00"
                for action in w["task"]["actions"]:
                    if action["commands"][0]["kind"] == "downlink":
                        action["commands"][0]["destination_id"] = "g00"
            if family == "long":
                d["horizon_sec"] = 1800.0
                for request in d["lifecycle"]["requests"]:
                    request["deadline_sec"] = 1800.0
                w["task"]["max_decisions"] = 361
            w["scenario"]["lineage"] = f"release-v1/{group}"
            w["scenario"]["seed"] = 1000 + index
            geometry = {
                "orbits": d["orbits"],
                "targets": d["acquisition_targets"],
                "stations": d["operations"]["stations"],
            }
            cases.append(
                {
                    "id": cfg["id"],
                    "family": family,
                    "split": split,
                    "group": group,
                    "geometry_sha256": digest(geometry),
                    "scenario": w["scenario"],
                    "task": w["task"],
                    "goals": {
                        "min_fulfilled": 1,
                        "require_no_failed_operations": True,
                        "coupled": {"native_spec_sha256": digest(d), "min_acquired": 0},
                    },
                    "negative": family in ("outage", "deadline"),
                }
            )
    return {
        "schema": "space-env.release-suite/1",
        "id": "release-v1-smoke" if smoke else "release-v1",
        "cases": copy.deepcopy(cases),
        "budget": {
            "max_search_nodes": 1000000,
            "max_orders": 8,
            "timeout_sec": 300,
            "max_result_bytes": 64000000,
            "max_rss_bytes": 2147483648,
        },
        "gates": {
            "min_test_groups": 2,
            "min_completion_fraction": 0.1,
            "min_paired_margin": 0.0,
            "max_rejections": 0,
        },
        "scope": "synthetic kinematic metadata missions; fixed geometry before execution; not flight calibration",
    }
