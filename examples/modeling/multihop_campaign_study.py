"""Native two-hop custody, loop-safe public policy and replayed training exports."""

import argparse
import shutil
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from examples.modeling.relay_campaign_study import run
from space_env.industrial.agent import export_training
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.multihop_campaign import multihop_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for source in [
        Path(__file__),
        Path(__file__).with_name("relay_campaign_study.py"),
        Path(__file__).with_name("audit_reactive_campaign.py"),
    ]:
        shutil.copyfile(source, args.output / source.name)
    rows = []
    profiles = [
        ("nominal", {}, True),
        ("recovery", {"fault_profile": "after_custody_recover"}, True),
        ("blocked-forward", {"forward_link_available": False}, False),
        ("small-buffer", {"sink_memory_bits": 250000}, False),
        ("delayed", {"route_delay_sec": 1}, True),
        ("expired", {"bundle_ttl_sec": 1}, False),
        ("lost-owner", {"fault_profile": "after_custody_permanent"}, False),
        ("two-hours", {"horizon_sec": 7200, "release_sec": None}, True),
    ]
    for name, options, success in profiles:
        case = multihop_campaign({"horizon_sec": 240, "release_sec": 60, **options})
        directory = args.output / name
        directory.mkdir()
        record, native, ack = run(case)
        assert record["metrics"]["success"] is success
        assert not record["call_budget_exhausted"]
        assert all(
            e["response"].get("feedback", {}).get("accepted") for e in record["events"]
        )
        network = record["metrics"]["network_accounting"]
        assert network["maximum_balance_error_bits"] < 1e-6
        assert network["wire_balance_error_bits"] < 1e-6
        assert not any(
            r["source"] == "s" and r["destination"] == "r"
            for r in network["custody_history"]
        )
        trace = verify_policy_trace(record, case["public_contract"])
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        atomic_write(directory / "episode.json", canonical_json(record))
        atomic_write(directory / "native-network.json", canonical_json(native))
        exported = export_training(record, directory / "rl.jsonl")
        if name == "two-hours":
            export_training(record, directory / "sft.jsonl", format="sft")
        rows.append(
            {
                "profile": name,
                "expected_success": success,
                "calls": len(record["events"]),
                "late_ack_sec": ack,
                "late_ack_delay_sec": None
                if ack is None
                else ack - case["setup"]["release_sec"],
                "metrics": record["metrics"],
                "policy_trace": trace,
                "replay_export": exported,
                "episode_sha256": record["sha256"],
            }
        )
        atomic_write(
            args.output / "report.json",
            canonical_json(
                {
                    "rows": rows,
                    "scope": "synthetic four-spacecraft campaign with two ISL hops for camera a, static product topology, native custody and current central feedback; scripted_public, not LLM training or flight calibration",
                }
            ),
        )
        print(name, success, len(record["events"]), ack, flush=True)


if __name__ == "__main__":
    main()
