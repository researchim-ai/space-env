"""Delayed route-channel evidence; native central operational feedback is current."""

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
    for source in [
        Path(__file__),
        Path(__file__).with_name("relay_campaign_study.py"),
        Path(__file__).with_name("audit_reactive_campaign.py"),
    ]:
        shutil.copyfile(source, args.output / source.name)
    profiles = [
        ("instant-mid", {}, True),
        ("delayed-mid", {"route_delay_sec": 1}, True),
        (
            "delayed-before",
            {"route_delay_sec": 4, "fault_profile": "before_transfer"},
            True,
        ),
        ("stale-only", {"route_delay_sec": 1, "max_route_age_sec": 0.5}, False),
    ]
    rows = []
    for name, options, expected in profiles:
        case = routing_campaign(
            {
                "horizon_sec": 240,
                "release_sec": 60,
                "fault_profile": "mid_transfer",
                **options,
            }
        )
        directory = args.output / name
        directory.mkdir()
        record, native, ack = run(case)
        assert record["metrics"]["success"] is expected
        assert not record["call_budget_exhausted"]
        trace = verify_policy_trace(record, case["public_contract"])
        network = record["metrics"]["network_accounting"]
        assert network["maximum_balance_error_bits"] < 1e-6
        assert network["wire_balance_error_bits"] < 1e-6
        assert all(
            e["response"].get("feedback", {}).get("accepted") for e in record["events"]
        )
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        atomic_write(directory / "episode.json", canonical_json(record))
        atomic_write(directory / "native-network.json", canonical_json(native))
        exported = export_training(record, directory / "rl.jsonl")
        if name == "delayed-mid":
            export_training(record, directory / "sft.jsonl", format="sft")
        ages = [
            e["response"]["observation"].get("route_telemetry", {}).get("age_sec")
            for e in record["events"]
        ]
        ages = [a for a in ages if a is not None]
        if options.get("route_delay_sec"):
            assert min(ages) >= options["route_delay_sec"] - 1e-9
        rows.append(
            {
                "profile": name,
                "expected_success": expected,
                "calls": len(record["events"]),
                "late_ack_sec": ack,
                "metrics": record["metrics"],
                "policy_trace": trace,
                "minimum_sample_age_sec": min(ages) if ages else None,
                "maximum_sample_age_sec": max(ages) if ages else None,
                "replay_export": exported,
                "episode_sha256": record["sha256"],
            }
        )
        atomic_write(
            args.output / "report.json",
            canonical_json(
                {
                    "rows": rows,
                    "scope": "decision-boundary delayed route channel; other central observations current; synthetic scripted_public evidence, no LLM training",
                }
            ),
        )
        print(
            name, record["metrics"]["success"], len(record["events"]), ack, flush=True
        )


if __name__ == "__main__":
    main()
