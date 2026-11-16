"""Audit sealed episodes, public policy traces and recorded native network budgets.

Physical replay is performed by export_training during generation. This audit
reconciles the saved journals and data files; it is not an independent RF model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from space_env.simulation.snapshot import digest


def audit(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    records = {}
    counts = {"episodes": 0, "public_policy_traces": 0, "rl_rows": 0, "sft_rows": 0}
    expected = {
        "one-terminal",
        "extra-terminal",
        "wide-pool",
        "ack-loss",
        "expired-bundles",
        "fine-network",
        "two-hours",
    }
    for row in report["rows"]:
        name = row["id"]
        if name not in expected or name in records:
            raise ValueError("invalid study matrix")
        directory = root / name
        record = json.loads((directory / "episode.json").read_text())
        case = json.loads((directory / "privileged-case.json").read_text())
        native = json.loads((directory / "native-network.json").read_text())
        if (
            record["sha256"]
            != digest({k: v for k, v in record.items() if k != "sha256"})
            or row["sha256"] != record["sha256"]
        ):
            raise ValueError("episode seal mismatch")
        if row["metrics"] != record["metrics"] or not record["task_text"].startswith(
            case["public_brief"]
        ):
            raise ValueError("episode metrics or public brief mismatch")
        verify_policy_trace(record, case["public_contract"])
        counts["public_policy_traces"] += 1
        ops = case["configuration"]["scenario"]["definition"]["operations"]
        limits = {s["name"]: s["terminals"] for s in ops["stations"]}
        rates = ops["radio_network"]["receivers"]
        total = {s: 0.0 for s in limits}
        for interval in native["radio"]["intervals"]:
            dt = interval["end_sec"] - interval["start_sec"]
            if dt < 0:
                raise ValueError("negative radio interval")
            for station, limit in limits.items():
                grants = [
                    g for g in interval["grants"].values() if g["station"] == station
                ]
                cap = rates[station]["aggregate_rate_bps"]
                if (
                    len(grants) > limit
                    or math.fsum(g["rate_bps"] for g in grants) > cap + 1e-6
                ):
                    raise ValueError("terminal/rate capacity exceeded")
                bandwidth = rates[station].get("aggregate_bandwidth_hz")
                if (
                    bandwidth is not None
                    and math.fsum(g["bandwidth_hz"] for g in grants) > bandwidth + 1e-6
                ):
                    raise ValueError("aggregate bandwidth exceeded")
                bits = interval["station_bits"][station]
                if bits < 0 or bits > cap * dt + 1e-6:
                    raise ValueError("interval bit budget exceeded")
                total[station] += bits
        for station, bits in total.items():
            receipts = math.fsum(
                r["bits"] for r in native["receipts"] if r["station"] == station
            )
            if (
                abs(bits - receipts) > 1e-6
                or abs(bits - native["radio"]["station_transmitted_bits"][station])
                > 1e-6
            ):
                raise ValueError("ground bit counters disagree")
        dtn = native["dtn"]
        left = math.fsum(dtn["tx_serialized_bits"].values())
        right = math.fsum(
            [
                *dtn["rx_unique_bits"].values(),
                *dtn["rx_duplicate_bits"].values(),
                *dtn["rx_lost_fragment_bits"].values(),
                *dtn["wasted_partial_bits"].values(),
                *dtn["abandoned_inflight_bits"].values(),
                dtn["inflight_payload_bits"],
                dtn["partial_serialization_bits"],
            ]
        )
        if abs(left - right) > 1e-6:
            raise ValueError("DTN serialization counters disagree")
        if (
            dtn["custody_history"]
            != record["metrics"]["network_accounting"]["custody_history"]
        ):
            raise ValueError("custody journal differs")
        for format in ["rl", "sft"]:
            path = directory / (format + ".jsonl")
            if not path.exists():
                if format == "rl":
                    raise ValueError("missing RL row")
                continue
            data = json.loads(path.read_text())
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
                set(data) != fields
                or data["schema"] != "space-env.industrial-training/1"
                or data["format"] != format
                or data["policy_origin"] != record["policy_origin"]
                or data["kind"] != record["initial"]["kind"]
                or data["rewards"] != rewards
                or data["return"] != sum(rewards)
                or data["terminal"] != record["metrics"]["terminal"]
                or data["call_budget_exhausted"] != record["call_budget_exhausted"]
            ):
                raise ValueError("training schema, rewards or terminal label differ")
            if (
                data["messages"] != record["messages"]
                or data["source_record_sha256"] != record["sha256"]
                or data["success"] != record["metrics"]["success"]
            ):
                raise ValueError("training row differs")
            if format == "sft" and not record["metrics"]["success"]:
                raise ValueError("invalid SFT success label")
            counts[format + "_rows"] += 1
        records[name] = record
        counts["episodes"] += 1
    if set(records) != expected:
        raise ValueError("incomplete matrix")
    base = records["one-terminal"]["initial"]["configuration"]
    for name, rate in [("extra-terminal", 250000.0), ("wide-pool", 500000.0)]:
        modified = copy.deepcopy(base)
        ops = modified["scenario"]["definition"]["operations"]
        next(s for s in ops["stations"] if s["name"] == "Late-Ground")["terminals"] = 2
        ops["radio_network"]["receivers"]["Late-Ground"]["aggregate_rate_bps"] = rate
        if modified != records[name]["initial"]["configuration"]:
            raise ValueError("capacity comparison changed unrelated native settings")
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(audit(p.parse_args().directory), sort_keys=True))


if __name__ == "__main__":
    main()
