"""Reproduce public routing decisions, native custody and replayed RL exports."""

import argparse
import shutil
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from examples.modeling.relay_campaign_study import run
from space_env.industrial.agent import export_training
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.routing_campaign import routing_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "executed-driver.py")
    rows = []
    profiles = [
        "none",
        "before_transfer",
        "mid_transfer",
        "after_custody_recover",
        "after_custody_permanent",
        "all_relays_down",
    ]
    for profile in profiles:
        case = routing_campaign(
            {"horizon_sec": 240, "release_sec": 60, "fault_profile": profile}
        )
        directory = args.output / profile
        directory.mkdir()
        record, native, ack = run(case)
        expected = profile not in {"after_custody_permanent", "all_relays_down"}
        assert record["metrics"]["success"] is expected
        assert not record["call_budget_exhausted"]
        trace = verify_policy_trace(record, case["public_contract"])
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        atomic_write(directory / "episode.json", canonical_json(record))
        atomic_write(directory / "native-network.json", canonical_json(native))
        exported = export_training(record, directory / "rl.jsonl")
        if profile == "mid_transfer":
            export_training(record, directory / "sft.jsonl", format="sft")
        rows.append(
            {
                "profile": profile,
                "expected_success": expected,
                "metrics": record["metrics"],
                "calls": len(record["events"]),
                "late_ack_sec": ack,
                "policy_trace": trace,
                "replay_export": exported,
                "episode_sha256": record["sha256"],
            }
        )
        atomic_write(args.output / "report.json", canonical_json({"rows": rows}))
        print(profile, expected, ack, flush=True)


if __name__ == "__main__":
    main()
