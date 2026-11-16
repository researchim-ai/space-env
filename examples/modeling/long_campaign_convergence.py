"""Compare sealed public orbit traces; physical replay is verified separately."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from space_env.simulation.snapshot import canonical_json, digest


def compare_records(coarse, fine):
    for record in [coarse, fine]:
        if record["sha256"] != digest(
            {k: v for k, v in record.items() if k != "sha256"}
        ):
            raise ValueError("episode seal mismatch")
    expected = copy.deepcopy(coarse["initial"]["configuration"])
    for orbit in expected["scenario"]["definition"]["orbits"].values():
        orbit["integration"].update(sample_sec=1.0, max_step_sec=5.0)
    if expected != fine["initial"]["configuration"]:
        raise ValueError("paired configuration differs beyond orbit settings")

    def states(record):
        first = json.loads(record["messages"][1]["content"])["observation"]
        views = [first, *[e["response"]["observation"] for e in record["events"]]]
        return {
            v["native"]["time_sec"]: {
                r["id"]: [
                    r["values"][k]
                    for k in [
                        "r_x_m",
                        "r_y_m",
                        "r_z_m",
                        "v_x_m_s",
                        "v_y_m_s",
                        "v_z_m_s",
                    ]
                ]
                for r in v["native"]["entities"]
                if r["kind"] == "actor"
            }
            for v in views
        }

    left, right = states(coarse), states(fine)
    times = sorted(left.keys() & right.keys())
    if not times:
        raise ValueError("no common observation times")
    differences = {
        actor: {"max_position_m": 0.0, "max_velocity_m_s": 0.0}
        for actor in left[times[0]]
    }
    for t in times:
        if left[t].keys() != right[t].keys():
            raise ValueError("different actor sets")
        for actor, p in left[t].items():
            q = right[t][actor]
            differences[actor]["max_position_m"] = max(
                differences[actor]["max_position_m"], math.dist(p[:3], q[:3])
            )
            differences[actor]["max_velocity_m_s"] = max(
                differences[actor]["max_velocity_m_s"], math.dist(p[3:], q[3:])
            )
    return {
        "scope": "Difference of sealed traces using the same J2 model; not independent accuracy or a continuous-time error bound.",
        "matched_time_grid": left.keys() == right.keys(),
        "common_observations": len(times),
        "coarse_observations": len(left),
        "fine_observations": len(right),
        "actors": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    coarse = json.loads((args.directory / "two-hours" / "episode.json").read_text())
    fine = json.loads(
        (args.directory / "orbit-fine-two-hours" / "episode.json").read_text()
    )
    print(canonical_json(compare_records(coarse, fine)))


if __name__ == "__main__":
    main()
