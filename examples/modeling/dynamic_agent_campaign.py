"""Causal demand and interruption demonstrations on one coupled EO runtime."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from examples.modeling.campaign import recipe as base_recipe
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.campaign import AgentCampaignSession
from space_env.simulation.snapshot import atomic_write, canonical_json


def recipe(profile="arrival"):
    if profile not in {"arrival", "recovery"}:
        raise ValueError("unknown dynamic campaign profile")
    case, campaign = base_recipe("eo")
    case = copy.deepcopy(case)
    lifecycle = case["scenario"]["definition"]["lifecycle"]
    request = lifecycle["requests"].pop(1)
    request["release_sec"] = 2.0
    jobs = lifecycle["jobs"]
    lifecycle["jobs"] = []
    events = [
        {
            "id": "new-customer",
            "at_sec": 2.0,
            "type": "request_added",
            "payload": {"request": request, "jobs": jobs},
        },
        {
            "id": "urgent-customer",
            "at_sec": 2.5,
            "type": "request_priority_changed",
            "payload": {"request_id": request["id"], "priority": 9},
        },
    ]
    for action in case["task"]["actions"]:
        if action["id"] in {"cpu-input", "cpu", "final-radio"}:
            action["request_ref"] = request["id"]
    schedule = case["campaign_steps"]
    if profile == "recovery":
        # The original require_healthy means NO historical fault, not recovered
        # health. This fault-tolerant fixture explicitly permits fault history;
        # tests separately require cleared faults and a nonsafe final bus.
        campaign["phases"][1]["goal"]["limits"]["a"]["require_healthy"] = False
        events.extend(
            [
                {
                    "id": "cpu-loss",
                    "at_sec": 4.75,
                    "type": "spacecraft_fault",
                    "payload": {"actor": "a", "subsystem": "cpu"},
                },
                {
                    "id": "station-loss",
                    "at_sec": 4.75,
                    "type": "station_outage_added",
                    "payload": {"station": "Ground", "start_sec": 4.75, "end_sec": 7.0},
                },
                {
                    "id": "cpu-clear",
                    "at_sec": 5.5,
                    "type": "spacecraft_fault_cleared",
                    "payload": {"actor": "a", "subsystem": "cpu"},
                },
                {
                    "id": "station-clear",
                    "at_sec": 6.0,
                    "type": "station_recovered",
                    "payload": {"station": "Ground"},
                },
            ]
        )
        schedule = (
            schedule[:5]
            + [
                {"action_ids": ["cpu", "archive-radio"], "advance_sec": 2.0},
                {"action_ids": ["final-radio"], "advance_sec": 1.0},
                {"action_ids": ["new-radio"], "advance_sec": 1.0},
                {"action_ids": ["control"], "advance_sec": 1.0},
            ]
            + [{"action_ids": ["wait"], "advance_sec": 1.0} for _ in range(20)]
        )
    case["task"]["environment_events"] = events
    return {
        "scenario": case["scenario"],
        "task": case["task"],
        "campaign": campaign,
    }, schedule


def run(configuration, schedule, policy):
    episode = IndustrialEpisode(
        AgentCampaignSession(configuration),
        max_calls=128,
        policy_origin="scripted_privileged",
        task_text=(
            "Deliver all three customer products, including newly announced demand, then reorient. "
            "This is an author-scripted reference demonstration, not an LLM result."
        ),
    )
    for item in schedule:
        remaining = item["advance_sec"]
        first = True
        while remaining > 1e-9 and not episode.done:
            view = episode.observe()
            enabled = {
                row["id"] for row in view["native"]["catalogue"] if row["enabled"]
            }
            choices = (
                [a for a in item["action_ids"] if a in enabled]
                if first and policy == "reference"
                else ["wait"]
            )
            index = len(episode._events)
            result = episode.call(
                str(index),
                json.dumps(
                    {
                        "expected_revision": view["revision"],
                        "idempotency_key": str(index),
                        "action": {
                            "choices": {"@central": choices},
                            "advance_sec": remaining,
                        },
                    }
                ),
            )
            if "error" in result:
                raise RuntimeError("invalid scripted decision")
            remaining -= result["elapsed_sec"]
            first = False
        if episode.done:
            break
    while not episode.done:
        index = len(episode._events)
        episode.call(
            str(index),
            json.dumps(
                {
                    "expected_revision": episode.observe()["revision"],
                    "idempotency_key": str(index),
                    "action": {"choices": {"@central": ["wait"]}, "advance_sec": 1.0},
                }
            ),
        )
    return episode.record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["arrival", "recovery"], default="arrival")
    args = parser.parse_args()
    configuration, schedule = recipe(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for policy in ["wait", "reference"]:
        record = run(configuration, schedule, policy)
        verified = verify_episode(record)
        atomic_write(args.output / (policy + ".json"), canonical_json(record))
        export_training(record, args.output / (policy + "-rl.jsonl"))
        if record["metrics"]["success"] and all(
            e["response"].get("feedback", {}).get("accepted") for e in record["events"]
        ):
            export_training(record, args.output / (policy + "-sft.jsonl"), format="sft")
        rows.append(
            {
                "policy": policy,
                "metrics": record["metrics"],
                "verified": verified,
                "sha256": record["sha256"],
            }
        )
    report = {
        "profile": args.profile,
        "scope": "Short synthetic scripted continuity test; no model inference or hardware qualification.",
        "rows": rows,
    }
    atomic_write(args.output / "report.json", canonical_json(report))
    print(canonical_json(report))


if __name__ == "__main__":
    main()
