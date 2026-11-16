"""Continuous native relay routes, shared ground pools and causal packet failures."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from examples.modeling.adaptive_campaign_study import accounting
from examples.modeling.audit_reactive_campaign import verify_policy_trace
from space_env.industrial.agent import IndustrialEpisode, export_training
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.relay_campaign import relay_campaign


def run(case, *, policy=campaign_policy):
    episode = IndustrialEpisode(
        AgentCampaignSession(case["configuration"]),
        max_calls=512,
        policy_origin="scripted_public",
        task_text=case["public_brief"]
        + " Public contract: "
        + canonical_json(case["public_contract"]),
    )
    while not episode.done:
        view = episode.observe()
        index = len(episode._events)
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": view["revision"],
                    "idempotency_key": str(index),
                    "action": policy(view, case["public_contract"]),
                }
            ),
        )
    record = episode.record()
    # Privileged evaluation AFTER execution; never passed into campaign_policy.
    report = episode._session._native._session._runtime._backend.report()
    native = {
        "dtn": report["dtn"],
        "radio": report["radio_network"],
        "receipts": report["receipts"],
        "products": report["products"],
        "radio_offers": [
            {
                "start_sec": i["start_sec"],
                "end_sec": i["end_sec"],
                "radio_offers": i["radio_offers"],
            }
            for i in report["lifecycle"]["resource_lanes"]["intervals"]
            if i.get("radio_offers")
        ],
    }
    completed = {p["id"]: p["completed_sec"] for p in report["products"]}
    times = [completed.get(p) for p in ["late-final-a", "late-final-b"]]
    late_ack = None if any(t is None for t in times) else max(times)
    return record, native, late_ack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    profiles = [
        ("one-terminal", {}),
        ("extra-terminal", {"ground_terminals": 2}),
        ("wide-pool", {"ground_terminals": 2, "ground_rate_bps": 500000.0}),
        ("ack-loss", {"ack_loss_probability": 0.5}),
        ("expired-bundles", {"bundle_ttl_sec": 0.001}),
        ("fine-network", {"network_sample_sec": 1.0}),
        ("two-hours", {"horizon_sec": 7200.0, "release_sec": None}),
    ]
    for name, options in profiles:
        parameters = {"horizon_sec": 240.0, "release_sec": 60.0, **options}
        case = relay_campaign(parameters)
        directory = args.output / name
        directory.mkdir()
        atomic_write(directory / "privileged-case.json", canonical_json(case))
        start = time.perf_counter()
        record, native, late_ack = run(case)
        elapsed = time.perf_counter() - start
        trace = verify_policy_trace(record, case["public_contract"])
        atomic_write(directory / "episode.json", canonical_json(record))
        atomic_write(directory / "native-network.json", canonical_json(native))
        replay_export = export_training(record, directory / "rl.jsonl")
        if (
            name == "two-hours"
            and record["metrics"]["success"]
            and all(
                e["response"].get("feedback", {}).get("accepted")
                for e in record["events"]
            )
        ):
            export_training(record, directory / "sft.jsonl", format="sft")
        row = {
            "id": name,
            "setup": case["setup"],
            "metrics": record["metrics"],
            "calls": len(record["events"]),
            "policy_trace": trace,
            "replay_export": replay_export,
            "sha256": record["sha256"],
            "late_ground_ack_sec": late_ack,
            "late_ground_ack_delay_sec": None
            if late_ack is None
            else late_ack - case["setup"]["release_sec"],
            "accounting": accounting(record),
            "execution_wall_sec": elapsed,
        }
        rows.append(row)
        print(
            name,
            row["calls"],
            record["metrics"]["success"],
            row["late_ground_ack_delay_sec"],
            flush=True,
        )
    report = {
        "scope": "Constructive synthetic four-spacecraft EO/CPU/S-band custody/X-band ground campaign; shared receiver pools; ideal lossy out-of-band ACK; not hardware/protocol certification or a general constellation optimizer.",
        "timing_scope": "Uncontrolled local host; execution_wall_sec excludes compiler and replay/export.",
        "rows": rows,
    }
    atomic_write(args.output / "report.json", canonical_json(report))


if __name__ == "__main__":
    main()
