"""Replayed five-spacecraft detours, link recovery and no-path negatives."""

import argparse
import shutil
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from examples.modeling.relay_campaign_study import run
from space_env.industrial.agent import export_training
from space_env.industrial.detour import detour_policy
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.detour_campaign import detour_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in [
        Path(__file__).name,
        "relay_campaign_study.py",
        "audit_reactive_campaign.py",
    ]:
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    profiles = [
        ("detour", {}, True),
        ("direct", {"forward_link_available": True}, True),
        ("blocked", {"backup_link_available": False}, False),
        (
            "link-recovery",
            {"forward_recovery_delay_sec": 10, "backup_link_available": False},
            True,
        ),
        ("radio-recovery", {"fault_profile": "after_custody_recover"}, True),
        ("expired", {"bundle_ttl_sec": 1}, False),
        ("lost-owner", {"fault_profile": "after_custody_permanent"}, False),
        ("delayed", {"route_delay_sec": 1}, True),
        ("two-hours", {"horizon_sec": 7200, "release_sec": None}, True),
    ]
    rows = []
    for name, options, expected in profiles:
        case = detour_campaign({"horizon_sec": 240, "release_sec": 60, **options})
        directory = args.output / name
        directory.mkdir()
        record, native, ack = run(case, policy=detour_policy)
        assert record["metrics"]["success"] is expected
        assert not record["call_budget_exhausted"]
        assert all(
            e["response"].get("feedback", {}).get("accepted") for e in record["events"]
        )
        network = record["metrics"]["network_accounting"]
        assert network["maximum_balance_error_bits"] < 1e-6
        assert network["wire_balance_error_bits"] < 1e-6
        for pid in {h["product_id"] for h in network["custody_history"]}:
            history = [h for h in network["custody_history"] if h["product_id"] == pid]
            holders = [history[0]["source"]] + [h["destination"] for h in history]
            assert len(holders) == len(set(holders))
        trace = verify_policy_trace(
            record, case["public_contract"], policy=detour_policy
        )
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        atomic_write(directory / "episode.json", canonical_json(record))
        atomic_write(directory / "native-network.json", canonical_json(native))
        exported = export_training(record, directory / "rl.jsonl")
        if name == "two-hours":
            export_training(record, directory / "sft.jsonl", format="sft")
        rows.append(
            {
                "profile": name,
                "expected_success": expected,
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
                    "scope": "synthetic five-spacecraft EO/CPU/ISL/ground campaign; current central RF/custody graph and bounded polling; no future RF/resource guarantee, no LLM training or flight calibration",
                }
            ),
        )
        print(name, expected, len(record["events"]), ack, flush=True)


if __name__ == "__main__":
    main()
