"""Bounded native multi-spacecraft phasing with independent RK4 endpoint targets.

One minute of central-gravity finite-thrust execution, not multi-day operational
constellation deployment. This task is separate from the slotted service model;
no fabricated transfer of power, mass or clocks between executors is performed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, verify_record
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import TaskSession
from space_env.tasks.deployment import deployment_recipe


def run(actors=2):
    case = deployment_recipe(actors)
    rows = []
    for name in ("wait", "reference"):
        session = TaskSession(case["scenario"], case["task"])
        episode = LLMEpisode(
            session,
            task_text="Reach every declared own-actor orbital state at 60 seconds with the fuel reserve preserved.",
            group="central-local-phasing/v1",
            split="test",
            policy_origin="scripted_privileged"
            if name == "reference"
            else "scripted_public",
        )
        chooser = (
            public_policy("wait", 5.0)
            if name == "wait"
            else reference_policy(
                case["reference_steps"],
                {a["id"]: a["role"] for a in case["task"]["actions"]},
            )
        )
        record = run_agent(episode, chooser)
        verify_record(record)
        rows.append({"policy": name, "record": record})
    body = {
        "schema": "space-env.native-local-phasing/1",
        "actors": actors,
        "recipe": case,
        "rows": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_deployment_qualification": False,
    }
    return {**body, "sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.actors)
    atomic_write(args.output, canonical_json(result))
    print(canonical_json({"actors": args.actors, "attempts": len(result["rows"])}))


if __name__ == "__main__":
    main()
