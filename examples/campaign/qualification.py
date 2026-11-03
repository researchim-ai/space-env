"""Freeze coupled formation workloads before running the standard scale harness."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from space_env.scaling.benchmark import ScaleSpec, run_scale
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.snapshot import atomic_write, canonical_json


def manifest(
    *,
    workers=1,
    wave_gap=12.0,
    cameras=4,
    relays=2,
    waves=3,
    horizon=90.0,
    repeats=2,
    ground_control_sec=4.0,
    ack_release_actions=False,
    modes=("nominal", "ground_recovery", "ack_loss", "ground_outage", "ttl"),
):
    workloads = []
    for mode in modes:
        w = campaign_workload(
            {
                "id": "campaign-" + mode,
                "cameras": cameras,
                "relays": relays,
                "waves": waves,
                "wave_gap_sec": wave_gap,
                "horizon_sec": horizon,
                "repeats": repeats,
                "mode": mode,
                "ground_control_sec": ground_control_sec,
                "ack_release_actions": ack_release_actions,
            }
        )
        if mode == "nominal":
            w["comparison_group"] = "nominal-restart"
            workloads.append(w)
            resumed = copy.deepcopy(w)
            resumed.update(id=w["id"] + "-resumed", resume_after_decisions=3)
            workloads.append(resumed)
        else:
            workloads.append(w)
    return ScaleSpec.model_validate(
        {
            "id": "coupled-campaign-v2"
            if ground_control_sec != 4.0 or ack_release_actions
            else "coupled-campaign-v1",
            "workloads": workloads,
            "workers": workers,
        }
    ).model_dump(mode="json", by_alias=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("--run", metavar="OUTPUT")
    p.add_argument("--wave-gap", type=float, default=12.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--relays", type=int, default=2)
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--horizon", type=float, default=90.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--ground-control-sec", type=float, default=4.0)
    p.add_argument("--ack-release-actions", action="store_true")
    p.add_argument(
        "--mode",
        action="append",
        choices=["nominal", "ground_recovery", "ack_loss", "ground_outage", "ttl"],
    )
    args = p.parse_args()
    target = Path(args.manifest)
    if target.exists():
        raise ValueError("manifest output must be a new path")
    value = manifest(
        workers=args.workers,
        wave_gap=args.wave_gap,
        cameras=args.cameras,
        relays=args.relays,
        waves=args.waves,
        horizon=args.horizon,
        repeats=args.repeats,
        ground_control_sec=args.ground_control_sec,
        ack_release_actions=args.ack_release_actions,
        modes=tuple(args.mode)
        if args.mode
        else ("nominal", "ground_recovery", "ack_loss", "ground_outage", "ttl"),
    )
    atomic_write(target, canonical_json(value))
    if args.run:
        result = run_scale(value, args.run)
        print(
            canonical_json(
                {"ok": result["ok"], "sha256": result["sha256"], "output": args.run}
            )
        )
        raise SystemExit(0 if result["ok"] else 1)
    print(target)


if __name__ == "__main__":
    main()
