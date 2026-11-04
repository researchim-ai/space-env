"""Frozen geographic validation cases for the bounded ready-data scheduler.

Sites/planes are declared before evaluation; no outcome search or pruning.
Holdout labels follow the existing lineage-family split function.
"""

import argparse
import copy
from pathlib import Path

from space_env.eval.statistics import assign_split
from space_env.planning.forecast_benchmark import run_forecast_benchmark
from space_env.scaling.independent import independent_workload
from space_env.simulation.snapshot import atomic_write, canonical_json


def configuration():
    base = {
        "id": "forecast-dev",
        "planes": [
            {"id": "a", "inclination_deg": 55.0, "raan_deg": 0.0, "phases_deg": [0.0]},
            {
                "id": "b",
                "inclination_deg": 75.0,
                "raan_deg": 120.0,
                "phases_deg": [90.0],
            },
        ],
        "stations": [
            {"name": f"g{i}", "lat_deg": 0.0, "lon_deg": float(lon)}
            for i, lon in enumerate(range(-180, 180, 45))
        ],
        "horizon_sec": 6000.0,
        "decision_sec": 60.0,
        "product_bits": 20000000.0,
        "min_delivered": 1,
    }
    a = copy.deepcopy(base)
    a["id"] = "forecast-northern-shift"
    a["planes"][0].update(inclination_deg=48.0, raan_deg=-73.0, phases_deg=[47.0])
    a["planes"][1].update(inclination_deg=82.0, raan_deg=11.0, phases_deg=[199.0])
    b = copy.deepcopy(base)
    b["id"] = "forecast-opposite-phase"
    b["planes"][0].update(inclination_deg=61.0, raan_deg=37.0, phases_deg=[211.0])
    b["planes"][1].update(inclination_deg=89.0, raan_deg=-91.0, phases_deg=[303.0])
    negative = copy.deepcopy(base)
    negative.update(id="forecast-outage", mode="all_outage", min_delivered=0)
    cases = []
    for cfg, split, control in [
        (base, "train", False),
        (a, "holdout", False),
        (b, "holdout", False),
        (negative, "holdout", True),
    ]:
        w = independent_workload(cfg)
        # Select only a split key, never coordinates or a service outcome.
        for i in range(100):
            lineage = cfg["id"] + "-fixed-split-" + str(i)
            if assign_split(lineage, "") == split:
                break
        else:
            raise ValueError("split key budget exhausted")
        w["scenario"]["lineage"] = lineage
        w["task"]["policy_origin"] = "scripted"
        cases.append(
            {
                "id": cfg["id"],
                "family": "independent-ideal-ready-downlink",
                "scenario": w["scenario"],
                "task": w["task"],
                "split": split,
                "negative": control,
                "min_fulfilled": 1,
            }
        )
    return {
        "id": "frozen-ready-data-forecast-v1",
        "cases": cases,
        "baselines": ["priority", "earliest_deadline"],
        "teacher": {
            "min_completion_fraction": 0.1,
            "max_rejections": 0,
            "min_cases_per_family": 2,
            "min_margin": 0.0,
            "require_negative_control": True,
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--write-spec", action="store_true")
    p.add_argument("--spec", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must not exist")
    if args.spec:
        from space_env.planning.mission_bundle import read_json

        frozen = read_json(args.spec)
    else:
        frozen = configuration()
    if args.write_spec:
        atomic_write(args.output, canonical_json(frozen))
    else:
        result = run_forecast_benchmark(frozen, args.output)
        print(
            canonical_json(
                {
                    "execution_integrity_passed": result["execution_integrity_passed"],
                    "teacher_status": result["teacher"]["status"],
                    "sha256": result["sha256"],
                }
            )
        )
        raise SystemExit(
            0
            if result["execution_integrity_passed"]
            and result["teacher"]["eligible_for_sft_selection"]
            else 1
        )
