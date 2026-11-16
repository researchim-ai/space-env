"""Recheck native episodes and the public-observation origin of reactive actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import CampaignConfiguration
from space_env.industrial.campaign_policy import campaign_policy


def verify_policy_trace(record, contract, *, policy=campaign_policy):
    """Decision provenance check; callers must separately verify_episode(record)."""
    if record["policy_origin"] != "scripted_public":
        raise ValueError("public policy provenance required")
    observation = json.loads(record["messages"][1]["content"])["observation"]
    for event in record["events"]:
        actual = json.loads(event["arguments"])
        if actual["action"] != policy(observation, contract):
            raise ValueError("decision differs from the declared public policy")
        if "observation" not in event["response"]:
            raise ValueError("public policy transition has no observation")
        observation = event["response"]["observation"]
    return {"public_policy_verified": True, "decisions": len(record["events"])}


def audit(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    expected_cases = {
        "arrival",
        "recovery",
        "delayed-arrival",
        "delayed-recovery",
        "mandatory-cancel",
        "permanent-cpu",
    }
    expected_policies = {"wait", "fixed_script", "reactive"}
    seen = set()
    counts = {"episodes": 0, "public_policy_traces": 0, "rl_rows": 0, "sft_rows": 0}
    contract = {"minimum_completed_requests": 3, "completion_action_ids": ["control"]}
    for row in report["rows"]:
        name, policy = row["case"], row["policy"]
        if (
            name not in expected_cases
            or policy not in expected_policies
            or (name, policy) in seen
        ):
            raise ValueError("invalid or duplicate study row")
        seen.add((name, policy))
        directory = root / name
        declared = json.loads((directory / "privileged-case.json").read_text())
        configuration = CampaignConfiguration.model_validate(
            declared["configuration"]
        ).model_dump(mode="json", by_alias=True)
        record = json.loads((directory / (policy + ".json")).read_text())
        if (
            declared["id"] != name
            or configuration != record["initial"]["configuration"]
        ):
            raise ValueError("episode differs from declared case")
        result = verify_episode(record)
        if (
            row["sha256"] != record["sha256"]
            or row["metrics"] != record["metrics"]
            or row["verified"] != result
        ):
            raise ValueError("study row differs from verified episode")
        if row["calls"] != len(record["events"]):
            raise ValueError("call count differs from episode")
        counts["episodes"] += 1
        if policy == "reactive":
            verify_policy_trace(record, contract)
            counts["public_policy_traces"] += 1
        for format in ["rl", "sft"]:
            path = directory / (policy + "-" + format + ".jsonl")
            if not path.exists():
                if format == "rl":
                    raise ValueError("missing RL row")
                continue
            lines = path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError("one episode per training row required")
            training = json.loads(lines[0])
            fields = {
                "schema",
                "format",
                "kind",
                "source_record_sha256",
                "messages",
                "policy_origin",
                "rewards",
                "return",
                "terminal",
                "call_budget_exhausted",
                "success",
            }
            rewards = [e["response"]["reward"] for e in record["events"]]
            if (
                set(training) != fields
                or training["schema"] != "space-env.industrial-training/1"
                or training["kind"] != record["initial"]["kind"]
                or training["rewards"] != rewards
                or training["return"] != sum(rewards)
                or training["terminal"] != record["metrics"]["terminal"]
                or training["call_budget_exhausted"] != record["call_budget_exhausted"]
            ):
                raise ValueError("training schema, rewards or terminal state differ")
            if (
                training["source_record_sha256"] != record["sha256"]
                or training["messages"] != record["messages"]
                or training["policy_origin"] != record["policy_origin"]
                or training["success"] != record["metrics"]["success"]
                or training["format"] != format
            ):
                raise ValueError("training provenance or messages differ")
            if format == "sft" and (
                not record["metrics"]["success"]
                or any(
                    "error" in e["response"]
                    or e["response"].get("feedback", {}).get("accepted") is False
                    for e in record["events"]
                )
            ):
                raise ValueError("invalid SFT success label")
            counts[format + "_rows"] += 1
    if seen != {(c, p) for c in expected_cases for p in expected_policies}:
        raise ValueError("incomplete comparison matrix")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
