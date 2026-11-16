"""Audit detour study integrity and public policy; exports already replay natively."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from space_env.industrial.campaign import CampaignConfiguration
from space_env.industrial.detour import detour_policy
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    expected = {
        "detour": True,
        "direct": True,
        "blocked": False,
        "link-recovery": True,
        "radio-recovery": True,
        "expired": False,
        "lost-owner": False,
        "delayed": True,
        "two-hours": True,
    }
    require(
        len(report["rows"]) == len(expected)
        and {r["profile"] for r in report["rows"]} == set(expected),
        "profile coverage mismatch",
    )
    results = []
    identity = sources()
    for row in report["rows"]:
        directory = root / row["profile"]
        record = json.loads((directory / "episode.json").read_text())
        case = json.loads((directory / "privileged-case.json").read_text())
        native = json.loads((directory / "native-network.json").read_text())
        require(
            record["sha256"]
            == digest({k: v for k, v in record.items() if k != "sha256"}),
            "episode seal mismatch",
        )
        require(record["initial"]["sources"] == identity, "historical source mismatch")
        require(
            CampaignConfiguration.model_validate(case["configuration"]).model_dump(
                mode="json", by_alias=True
            )
            == record["initial"]["configuration"],
            "configuration mismatch",
        )
        require(
            record["metrics"] == row["metrics"]
            and record["sha256"] == row["episode_sha256"],
            "report mismatch",
        )
        require(
            record["metrics"]["success"] is expected[row["profile"]],
            "unexpected outcome",
        )
        require(not record["call_budget_exhausted"], "call budget exhausted")
        require(
            all(
                e["response"].get("feedback", {}).get("accepted")
                for e in record["events"]
            ),
            "rejected command",
        )
        verify_policy_trace(record, case["public_contract"], policy=detour_policy)
        dtn = native["dtn"]
        require(
            abs(dtn["wire_balance_error_bits"]) < 1e-6
            and all(abs(v) < 1e-6 for v in dtn["receiver_balance_error_bits"].values()),
            "bit imbalance",
        )
        for pid in {h["product_id"] for h in dtn["custody_history"]}:
            history = [h for h in dtn["custody_history"] if h["product_id"] == pid]
            holders = [history[0]["source"]] + [h["destination"] for h in history]
            require(len(holders) == len(set(holders)), "custody revisit")
            require(
                all(
                    a["destination"] == b["source"] and a["time_sec"] < b["time_sec"]
                    for a, b in pairwise(history)
                ),
                "custody continuity mismatch",
            )
            expiries = {
                t["expires_sec"]
                for t in dtn["transfers"].values()
                if t["product_id"] == pid
            }
            require(len(expiries) <= 1, "TTL renewed")
        if expected[row["profile"]]:
            path = [
                (h["source"], h["destination"])
                for h in dtn["custody_history"]
                if h["product_id"] == "late-final-a"
            ]
            wanted = (
                [("a", "r"), ("r", "s")]
                if row["profile"] in {"direct", "link-recovery"}
                else [("a", "r"), ("r", "t"), ("t", "s")]
            )
            require(path == wanted, "delivery route mismatch")
        bits = sum(
            s["transmitted_bits"]
            for s in row["metrics"]["network_accounting"]["stations"].values()
        )
        require(
            abs(bits - sum(p["delivered_bits"] for p in native["products"])) < 1e-6,
            "ground credit duplicated",
        )
        views = [json.loads(record["messages"][1]["content"])["observation"]]
        views.extend(e["response"]["observation"] for e in record["events"])
        for view in views:
            graph = view["detour_graph"]
            require(
                graph["measured_at"] == view["native"]["time_sec"],
                "stale current graph",
            )
            visible = {a["id"] for a in view["native"]["catalogue"]}
            require(
                all(e["action_id"] in visible for e in graph["links"]), "hidden link"
            )
        for fmt in ("rl", "sft"):
            file = directory / (fmt + ".jsonl")
            if not file.exists():
                require(fmt == "sft", "missing RL export")
                continue
            exported = json.loads(file.read_text())
            require(
                exported["source_record_sha256"] == record["sha256"]
                and exported["messages"] == record["messages"],
                "training source mismatch",
            )
            require(
                exported["policy_origin"] == "scripted_public",
                "policy provenance mismatch",
            )
            require(fmt != "sft" or record["metrics"]["success"], "failed SFT export")
        results.append(
            {"profile": row["profile"], "public_views": len(views), "passed": True}
        )
    return {
        "checks": results,
        "scope": "archive seal, public policy, current graph, unique custody paths, TTL and native bit/ground credit invariants; physical replay was performed by export_training",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    atomic_write(args.root / "audit.json", canonical_json(result))
    print(canonical_json(result))


if __name__ == "__main__":
    main()
