"""Replay-verified campaign tool demonstrations; scripted, not model inference."""

import argparse
import json
from pathlib import Path

from examples.modeling.campaign import recipe
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.campaign import AgentCampaignSession
from space_env.simulation.snapshot import atomic_write, canonical_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mission", choices=["finite", "eo"], default="finite")
    args = parser.parse_args()
    case, campaign = recipe(args.mission)
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    schedule = case.get(
        "campaign_steps",
        [
            {"action_ids": ["burn"], "advance_sec": 1.0},
            {"action_ids": ["deliver"], "advance_sec": 2.0},
            {"action_ids": ["wait"], "advance_sec": 1.0},
        ],
    )
    for policy in ["wait", "reference"]:
        episode = IndustrialEpisode(
            AgentCampaignSession(
                {
                    "scenario": case["scenario"],
                    "task": case["task"],
                    "campaign": campaign,
                }
            ),
            policy_origin="scripted_privileged",
            max_calls=128,
            task_text="Execute the declared delivery and continuation mission. The reference demonstration uses privileged setup.",
        )
        while not episode.done:
            index = episode._session.revision
            item = (
                schedule[index]
                if index < len(schedule)
                else {"action_ids": ["wait"], "advance_sec": 1.0}
            )
            actions = item["action_ids"] if policy == "reference" else ["wait"]
            episode.call(
                str(index),
                json.dumps(
                    {
                        "expected_revision": index,
                        "idempotency_key": str(index),
                        "action": {
                            "choices": {"@central": actions},
                            "advance_sec": item["advance_sec"],
                        },
                    }
                ),
            )
        record = episode.record()
        verified = verify_episode(record)
        atomic_write(args.output / (policy + ".json"), canonical_json(record))
        export_training(record, args.output / (policy + "-rl.jsonl"))
        if record["metrics"]["success"]:
            export_training(record, args.output / (policy + "-sft.jsonl"), format="sft")
        report.append(
            {
                "policy": policy,
                "verified": verified,
                "metrics": record["metrics"],
                "sha256": record["sha256"],
            }
        )
    atomic_write(
        args.output / "report.json",
        canonical_json(
            {
                "scope": "Fixed synthetic scripted demonstrations; no LLM inference, training or independent physics validation.",
                "rows": report,
            }
        ),
    )
    print(canonical_json(report))


if __name__ == "__main__":
    main()
