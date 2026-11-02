"""Generate frozen distinct-orbit actor/task load profiles."""

import argparse
import json

from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--products", type=int, default=100)
    parser.add_argument("--planes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--outage", action="store_true")
    parser.add_argument("--horizon-sec", type=float, default=None)
    parser.add_argument("--decision-sec", type=float, default=2.0)
    args = parser.parse_args()
    workload = fleet_workload(
        {
            "id": f"fleet-{args.actors}-{args.products}"
            + ("-outage" if args.outage else ""),
            "actors": args.actors,
            "products": args.products,
            "planes": args.planes,
            "repeats": args.repeats,
            "horizon_sec": args.horizon_sec,
            "decision_sec": args.decision_sec,
            "mode": "ground_outage" if args.outage else "nominal",
        }
    )
    workload.update(max_step_p95_sec=30.0, max_execution_sec=300.0)
    manifest = {"id": "b09-q-independent-fleet", "workloads": [workload]}
    atomic_write(args.output, canonical_json(manifest))
    print(
        json.dumps(
            {
                "sha256": digest(manifest),
                "actors": args.actors,
                "products": args.products,
            }
        )
    )


if __name__ == "__main__":
    main()
