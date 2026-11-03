"""Freeze the new 2s-control formation case; retain the historical 4s baseline."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from space_env.scaling.benchmark import ScaleSpec, run_scale
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.snapshot import atomic_write, canonical_json


def manifest():
    straight = campaign_workload(
        {
            "id": "formation-12-40-ground-2s",
            "cameras": 8,
            "relays": 4,
            "waves": 5,
            "wave_gap_sec": 20.0,
            "horizon_sec": 120.0,
            "ground_control_sec": 2.0,
            "repeats": 2,
        }
    )
    straight.update(
        timeout_sec=2400,
        max_execution_sec=1200.0,
        max_step_p95_sec=60.0,
        max_checkpoint_bytes=536870912,
        max_peak_rss_bytes=6442450944,
        comparison_group="short-ground-recovery",
    )
    resumed = copy.deepcopy(straight)
    resumed.update(
        id="formation-12-40-ground-2s-resumed", repeats=1, resume_after_decisions=7
    )
    return ScaleSpec.model_validate(
        {
            "id": "ground-windows-v2-12-40",
            "workers": 3,
            "workloads": [straight, resumed],
        }
    ).model_dump(mode="json", by_alias=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error("manifest path must be new")
    value = manifest()
    atomic_write(args.manifest, canonical_json(value))
    if args.run:
        result = run_scale(value, args.run)
        print(
            canonical_json(
                {
                    "ok": result["ok"],
                    "sha256": result["sha256"],
                    "qualification_domains": result["qualification_domains"],
                }
            )
        )
        raise SystemExit(0 if result["ok"] else 1)
    print(args.manifest)


if __name__ == "__main__":
    main()
