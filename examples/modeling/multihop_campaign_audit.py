"""Audit a multihop study archive; native replay is performed by its exporter."""

import argparse
import json
from pathlib import Path

from examples.modeling.audit_reactive_campaign import verify_policy_trace
from space_env.industrial.campaign import CampaignConfiguration
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    expected = {
        "nominal": True,
        "recovery": True,
        "blocked-forward": False,
        "small-buffer": False,
        "delayed": True,
        "expired": False,
        "lost-owner": False,
        "two-hours": True,
    }
    require(
        {r["profile"] for r in report["rows"]} == set(expected)
        and len(report["rows"]) == len(expected),
        "profile coverage mismatch",
    )
    results = []
    for row in report["rows"]:
        folder = root / row["profile"]
        record = json.loads((folder / "episode.json").read_text())
        case = json.loads((folder / "privileged-case.json").read_text())
        native = json.loads((folder / "native-network.json").read_text())
        require(
            record["sha256"]
            == digest({k: v for k, v in record.items() if k != "sha256"}),
            "episode seal mismatch",
        )
        require(record["initial"]["sources"] == sources(), "historical source mismatch")
        require(
            CampaignConfiguration.model_validate(case["configuration"]).model_dump(
                mode="json", by_alias=True
            )
            == record["initial"]["configuration"],
            "declared configuration mismatch",
        )
        require(
            record["metrics"] == row["metrics"]
            and row["episode_sha256"] == record["sha256"],
            "report mismatch",
        )
        require(
            record["metrics"]["success"] is expected[row["profile"]],
            "unexpected outcome",
        )
        require(not record["call_budget_exhausted"], "call budget exhausted")
        verify_policy_trace(record, case["public_contract"])
        dtn = native["dtn"]
        require(abs(dtn["wire_balance_error_bits"]) < 1e-6, "wire imbalance")
        require(
            all(abs(v) < 1e-6 for v in dtn["receiver_balance_error_bits"].values()),
            "receiver imbalance",
        )
        history = [
            r for r in dtn["custody_history"] if r["product_id"] == "late-final-a"
        ]
        require(not any(r["source"] == "s" for r in history), "return loop detected")
        if expected[row["profile"]]:
            require(
                [(r["source"], r["destination"]) for r in history]
                == [("a", "r"), ("r", "s")],
                "custody chain mismatch",
            )
            product = next(p for p in native["products"] if p["id"] == "late-final-a")
            require(
                history[0]["time_sec"]
                < history[1]["time_sec"]
                <= product["completed_sec"],
                "noncausal delivery",
            )
        transfers = [
            t for t in dtn["transfers"].values() if t["product_id"] == "late-final-a"
        ]
        require(len({t["expires_sec"] for t in transfers}) <= 1, "bundle TTL renewed")
        ground_bits = sum(
            s["transmitted_bits"]
            for s in row["metrics"]["network_accounting"]["stations"].values()
        )
        require(
            abs(ground_bits - sum(p["delivered_bits"] for p in native["products"]))
            < 1e-6,
            "ground credit duplicated",
        )
        views = [json.loads(record["messages"][1]["content"])["observation"]]
        views.extend(e["response"]["observation"] for e in record["events"])
        for view in views:
            visible = {a["id"] for a in view["native"]["catalogue"]}
            require(
                all(e["action_id"] in visible for e in view["relay_topology"]),
                "hidden topology action",
            )
        for fmt in ("rl", "sft"):
            path = folder / (fmt + ".jsonl")
            if not path.exists():
                require(fmt == "sft", "missing RL export")
                continue
            exported = json.loads(path.read_text())
            require(
                exported["source_record_sha256"] == record["sha256"]
                and exported["messages"] == record["messages"],
                "training source mismatch",
            )
            require(
                exported["policy_origin"] == "scripted_public",
                "policy provenance mismatch",
            )
            require(fmt != "sft" or record["metrics"]["success"], "failed SFT episode")
        results.append(
            {"profile": row["profile"], "public_views": len(views), "passed": True}
        )
    return {
        "checks": results,
        "scope": "archive seal, public policy/topology, TTL, causal custody, ground credit and bit balance; not independent physical validation or an additional native replay",
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
