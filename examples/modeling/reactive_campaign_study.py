"""Public reactive control against fixed scripts and waiting on declared variants."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from examples.modeling.dynamic_agent_campaign import recipe
from examples.modeling.dynamic_agent_campaign import run as run_fixed
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import CampaignPolicySpec, campaign_policy
from space_env.simulation.snapshot import atomic_write, canonical_json


def cases():
    rows = []
    for name in [
        "arrival",
        "recovery",
        "delayed-arrival",
        "delayed-recovery",
        "mandatory-cancel",
        "permanent-cpu",
    ]:
        base = (
            "arrival"
            if name in {"arrival", "delayed-arrival", "mandatory-cancel"}
            else "recovery"
        )
        configuration, schedule = recipe(base)
        configuration = copy.deepcopy(configuration)
        configuration["task"]["max_decisions"] = 128
        # Same target, finite declared control occupancy. Unlike the original
        # 20-second template, 10 seconds can also start after a late recovery.
        for action in configuration["task"]["actions"]:
            if action["id"] == "control":
                action["commands"][0]["duration_sec"] = 10.0
        events = configuration["task"]["environment_events"]
        if name == "delayed-arrival":
            events[0]["at_sec"] = events[0]["payload"]["request"]["release_sec"] = 3.0
            events[1]["at_sec"] = 3.5
        elif name == "delayed-recovery":
            for event in events:
                if event["id"] == "cpu-clear":
                    event["at_sec"] = 7.0
                elif event["id"] == "station-clear":
                    event["at_sec"] = 7.5
                elif event["id"] == "station-loss":
                    event["payload"]["end_sec"] = 10.0
        elif name == "mandatory-cancel":
            events.append(
                {
                    "id": "cancel-required",
                    "at_sec": 3.0,
                    "type": "request_cancelled",
                    "payload": {"request_id": "customer-a"},
                }
            )
        elif name == "permanent-cpu":
            configuration["task"]["environment_events"] = [
                e for e in events if e["id"] != "cpu-clear"
            ]
        configuration["task"]["environment_events"].sort(key=lambda e: e["at_sec"])
        rows.append(
            {
                "id": name,
                "configuration": configuration,
                "fixed_schedule": schedule,
                "expected_reactive_success": name
                not in {"mandatory-cancel", "permanent-cpu"},
            }
        )
    return rows


def run_reactive(configuration, *, public_contract=None, max_calls=128, task_text=None):
    contract = CampaignPolicySpec.model_validate(
        public_contract
        or {"minimum_completed_requests": 3, "completion_action_ids": ["control"]}
    )
    episode = IndustrialEpisode(
        AgentCampaignSession(configuration),
        max_calls=max_calls,
        policy_origin="scripted_public",
        task_text=(
            (
                task_text
                or "Deliver customer products, then perform the declared completion action. "
            )
            + " Public service contract: "
            + canonical_json(contract.model_dump(mode="json"))
        ),
    )
    while not episode.done:
        observation = episode.observe()
        action = campaign_policy(observation, contract)
        index = len(episode._events)
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": observation["revision"],
                    "idempotency_key": str(index),
                    "action": action,
                }
            ),
        )
    return episode.record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases():
        directory = args.output / case["id"]
        directory.mkdir()
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        for policy in ["wait", "fixed_script", "reactive"]:
            record = (
                run_reactive(case["configuration"])
                if policy == "reactive"
                else run_fixed(
                    case["configuration"],
                    case["fixed_schedule"],
                    "wait" if policy == "wait" else "reference",
                )
            )
            verified = verify_episode(record)
            atomic_write(directory / (policy + ".json"), canonical_json(record))
            export_training(record, directory / (policy + "-rl.jsonl"))
            if record["metrics"]["success"] and all(
                e["response"].get("feedback", {}).get("accepted")
                for e in record["events"]
            ):
                export_training(
                    record, directory / (policy + "-sft.jsonl"), format="sft"
                )
            rows.append(
                {
                    "case": case["id"],
                    "policy": policy,
                    "metrics": record["metrics"],
                    "calls": len(record["events"]),
                    "verified": verified,
                    "sha256": record["sha256"],
                    "expected_reactive_success": case["expected_reactive_success"],
                }
            )
    report = {
        "scope": (
            "Six hand-authored short variants of one synthetic geometry, not blind holdouts, "
            "optimal-planner comparison, LLM inference or independent physical validation. "
            "Fixed scripts come from the original cases and are not retuned for shifted events."
        ),
        "rows": rows,
    }
    atomic_write(args.output / "report.json", canonical_json(report))
    print(canonical_json(report))


if __name__ == "__main__":
    main()
