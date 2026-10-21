#!/usr/bin/env python3
"""Пишет эпизод цепочки аппаратов в JSON для viz/index.html."""
import argparse
from pathlib import Path

from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet
from space_env.viz.export import export_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fleet episode for the Three.js viewer")
    parser.add_argument("-o", "--output", default="viz/sample.json")
    parser.add_argument("--satellites", type=int, default=2)
    parser.add_argument("--separation-deg", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--dt", type=float, default=60.0)
    args = parser.parse_args()
    fleet = Fleet.string_of_pearls(
        args.satellites,
        args.separation_deg,
        scenario=ScenarioConfig(),
    )
    document = export_trajectory(fleet, args.steps, args.dt, Path(args.output))
    print(f"Wrote {len(document['frames'])} frames to {args.output}")


if __name__ == "__main__":
    main()
