"""Reproduce mandatory-goal controls, current dispatch bounds and budget failures.

Priority-only deliberately ignores mandatory ranking. No model inference/training.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from space_env.industrial.agent import IndustrialEpisode, verify_episode
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.industrial.phasing_corpus import task_brief
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def run(configuration, policy, *, max_calls=24):
    if policy not in {
        "public",
        "wait",
        "priority_only",
        "unsafe_fuel",
        "invalid_budget",
    }:
        raise ValueError("unknown contract comparison policy")
    episode = IndustrialEpisode(
        OrbitSession(configuration),
        split=configuration["generation"]["split"],
        policy_origin="scripted_public",
        max_calls=max_calls,
        task_text=task_brief(configuration),
    )
    if policy == "unsafe_fuel":
        obs = episode.observe()
        first = orbit_policy(obs)["action_ids"][0]
        command = next(r for r in obs["catalogue"] if r["id"] == first)["commands"][0]
        actor = command["actor_id"]
        duration = command["duration_sec"]
        opposite = next(
            r["id"]
            for r in obs["catalogue"]
            if r["id"] not in ("wait", first) and r["commands"][0]["actor_id"] == actor
        )
    index = 0
    while not episode.done:
        obs = episode.observe()
        if policy == "invalid_budget":
            episode.call(str(index), "{}")
            index += 1
            continue
        if policy == "priority_only":
            view = copy.deepcopy(obs)
            view.pop("task_objective", None)
            action = orbit_policy(view)
        elif policy == "wait":
            action = {}
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
        manifest["sha256"]
        != digest({k: v for k, v in manifest.items() if k != "sha256"})
        or manifest["sources"] != sources()
    ):
        raise ValueError("invalid corpus or source identity")
    if (
        manifest.get("acceptance") != "mandatory"
        or not manifest.get("competition")
        or not manifest.get("utility")
        or not 1 <= len(manifest["cases"]) <= 96
    ):
        raise ValueError("choose a bounded mandatory acceptance corpus")
    ids = [c["id"] for c in manifest["cases"]]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"(?:train|development|test)-[0-9]{4}", name) for name in ids
    ):
        raise ValueError("invalid or duplicate case identifiers")
    root = Path(output)
    if root.exists():
        raise ValueError("choose a new output directory")
    root.mkdir(parents=True)
    jobs = [
        (c["id"], c["configuration"], "priority_only", 24) for c in manifest["cases"]
    ]
    extras = {}
    for mode in ("priority", "mandatory", "infeasible"):
        configuration = {
            "actors": 4,
            "generation": {
                "split": "test",
                "seed": 0,
                "windows": True,
                "competition": True,
            },
            "utility": {
                "priorities": {f"slot-sat-{i:02}": 9 if i == 0 else 1 for i in range(4)}
            },
            "acceptance": {
                "minimum_fraction": 1 / 12 if mode == "mandatory" else 0.75,
                "mandatory": ["slot-sat-01"]
                if mode == "mandatory"
                else [f"slot-sat-{i:02}" for i in range(4)]
                if mode == "infeasible"
                else ["slot-sat-00"],
            },
        }
        extras["four-" + mode] = configuration
        jobs.append(("four-" + mode, configuration, "public", 24))
        if mode == "mandatory":
            jobs.append(("four-" + mode, configuration, "priority_only", 24))
    capacity = {
        "actors": 2,
        "generation": {
            "split": "test",
            "seed": 0,
            "windows": True,
            "competition": True,
        },
        "utility": {"priorities": {"slot-sat-00": 9, "slot-sat-01": 1}},
        "acceptance": {
            "minimum_fraction": 1.0,
            "mandatory": ["slot-sat-00", "slot-sat-01"],
        },
        "dispatch_events": [{"at_sec": 5.0, "kind": "capacity", "capacity": 2}],
    }
    safety = {
        "actors": 2,
        "generation": {"split": "train", "seed": 0},
        "utility": {"priorities": {"slot-sat-00": 1000, "slot-sat-01": 1}},
        "acceptance": {"minimum_fraction": 0.9, "mandatory": ["slot-sat-00"]},
    }
    budget = copy.deepcopy(manifest["cases"][0]["configuration"])
    extras.update(capacity=capacity, safety=safety, budget=budget)
    jobs += [
        ("capacity", capacity, "public", 24),
        ("unsafe_fuel", safety, "unsafe_fuel", 24),
        ("budget_valid", budget, "wait", 1),
        ("budget_invalid", budget, "invalid_budget", 1),
    ]
    setup = {
        "sources": sources(),
        "corpus_manifest_sha256": manifest["sha256"],
        "extras": extras,
        "scope": "Priority-only is an intentional negative control. Two budget cases are truncated episodes, not completed missions.",
    }
    atomic_write(root / "manifest.json", canonical_json(setup))
    rows = []
    for case, configuration, policy, budget_count in jobs:
        row = {"case": case, "policy": policy, "max_calls": budget_count}
        try:
            record = run(configuration, policy, max_calls=budget_count)
            verification = verify_episode(record)
            name = case + "--" + policy + ".json"
            atomic_write(root / name, canonical_json(record))
            row.update(
                status="completed",
                record=name,
                sha256=record["sha256"],
                verification=verification,
                metrics=record["metrics"],
                call_budget_exhausted=record["call_budget_exhausted"],
            )
        except Exception as exc:  # noqa: BLE001 -- retain every attempted case
            row.update(
                status="error", error_type=type(exc).__name__, error=str(exc)[:1024]
            )
        rows.append(row)
        atomic_write(root / "progress.json", canonical_json({"rows": rows}))
    if sources() != setup["sources"]:
        raise ValueError("source changed during comparison")
    body = {
        "schema": "space-env.task-contract-comparisons/1",
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
