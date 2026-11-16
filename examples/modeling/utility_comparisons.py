"""Reproduce public/inverse-priority and native fuel-safety comparison records.

The inverse-priority policy is an intentional negative control, not a strong
optimisation competitor. No model inference or training is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from space_env.industrial.agent import IndustrialEpisode, verify_episode
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def run(configuration, policy):
    if policy not in {"wait", "public", "low_priority", "unsafe_fuel"}:
        raise ValueError("unknown comparison policy")
    episode = IndustrialEpisode(
        OrbitSession(configuration),
        split=configuration["generation"]["split"],
        policy_origin="scripted_public",
        max_calls=24,
        task_text="Reach the published orbital-state targets with fuel reserves preserved. Choose at most one visible manoeuvre per spacecraft. Use current ideal state, published engine constants and declared tools; no future execution or reference schedule is available.",
    )
    index = 0
    if policy == "unsafe_fuel":
        obs = episode.observe()
        first = orbit_policy(obs)["action_ids"][0]
        row = next(r for r in obs["catalogue"] if r["id"] == first)
        actor = row["commands"][0]["actor_id"]
        duration = row["commands"][0]["duration_sec"]
        opposite = next(
            r["id"]
            for r in obs["catalogue"]
            if r["id"] not in ("wait", first) and r["commands"][0]["actor_id"] == actor
        )
    while not episode.done:
        obs = episode.observe()
        if policy == "wait":
            action = {}
        elif policy == "low_priority":
            view = copy.deepcopy(obs)
            view["reward_protocol"]["condition_weights"] = {
                k: 1001 - v
                for k, v in view["reward_protocol"]["condition_weights"].items()
            }
            action = orbit_policy(view)
        elif policy == "unsafe_fuel":
            action = (
                {"action_ids": [first]}
                if index == 0
                else {"action_ids": [opposite]}
                if obs["time_sec"] == duration
                else {}
            )
        else:
            action = orbit_policy(obs)
        episode.call(
            str(index),
            canonical_json(
                {
                    "action": action,
                    "expected_revision": obs["revision"],
                    "idempotency_key": str(index),
                }
            ),
        )
        index += 1
    return episode.record()


def compare(corpus, output):
    manifest = json.loads((Path(corpus) / "manifest.json").read_text())
    if (
        manifest.get("sha256")
        != digest({k: v for k, v in manifest.items() if k != "sha256"})
        or manifest["sources"] != sources()
    ):
        raise ValueError("invalid corpus or source identity")
    if (
        not manifest.get("competition")
        or not manifest.get("utility")
        or not 1 <= len(manifest["cases"]) <= 96
    ):
        raise ValueError("choose a bounded utility competition corpus")
    if len({case["id"] for case in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("duplicate case identifiers")
    if any(
        not re.fullmatch(r"(?:train|development|test)-[0-9]{4}", case["id"])
        for case in manifest["cases"]
    ):
        raise ValueError("invalid case identifier")
    root = Path(output)
    if root.exists():
        raise ValueError("choose a new output directory")
    root.mkdir(parents=True)
    four = {
        "actors": 4,
        "generation": {
            "split": "test",
            "seed": 0,
            "windows": True,
            "competition": True,
        },
        "utility": {
            "priorities": {f"slot-sat-{i:02}": 9 if i == 3 else 1 for i in range(4)}
        },
    }
    safety = {
        "actors": 2,
        "generation": {"split": "train", "seed": 0},
        "utility": {"priorities": {"slot-sat-00": 1000, "slot-sat-01": 1}},
    }
    setup = {
        "sources": sources(),
        "competition_manifest_sha256": manifest["sha256"],
        "four": four,
        "safety": safety,
        "baseline": "Inverse-priority ranking with the same public endpoint predictor; negative control, not a strong optimisation comparator.",
    }
    atomic_write(root / "manifest.json", canonical_json(setup))
    jobs = (
        [(c["id"], c["configuration"], "low_priority") for c in manifest["cases"]]
        + [("four", four, p) for p in ("wait", "public", "low_priority")]
        + [("unsafe_fuel", safety, "unsafe_fuel")]
    )
    rows = []
    for case, configuration, policy in jobs:
        row = {"case": case, "policy": policy}
        try:
            record = run(configuration, policy)
            verification = verify_episode(record)
            name = case + "--" + policy + ".json"
            atomic_write(root / name, canonical_json(record))
            row.update(
                status="completed",
                record=name,
                verification=verification,
                metrics=record["metrics"],
            )
        except Exception as exc:  # noqa: BLE001 -- retain every attempt
            row.update(
                status="error", error_type=type(exc).__name__, error=str(exc)[:1024]
            )
        rows.append(row)
        atomic_write(root / "progress.json", canonical_json({"rows": rows}))
    if sources() != setup["sources"]:
        raise ValueError("source changed during comparison")
    body = {
        "schema": "space-env.utility-comparisons/1",
        "manifest_sha256": digest(setup),
        "rows": rows,
    }
    result = {**body, "sha256": digest(body)}
    atomic_write(root / "report.json", canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.corpus, args.output)
    print(
        canonical_json(
            {
                "attempts": len(result["rows"]),
                "errors": sum(r["status"] == "error" for r in result["rows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
