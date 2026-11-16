"""Replay contract comparisons and export only train negatives as declared-task RL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_env.industrial.agent import export_training, verify_episode
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def export_negatives(corpus, comparisons, output):
    source, root, destination = Path(corpus), Path(comparisons).resolve(), Path(output)
    if destination.exists():
        raise ValueError("choose a new export directory")
    manifest = json.loads((source / "manifest.json").read_text())
    setup = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    for value in (manifest, report):
        if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("invalid source seal")
    if (
        manifest["sources"] != sources()
        or setup["sources"] != sources()
        or setup["corpus_manifest_sha256"] != manifest["sha256"]
        or report["manifest_sha256"] != digest(setup)
    ):
        raise ValueError("source or manifest mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    expected = {(case, "priority_only") for case in cases} | {
        ("four-priority", "public"),
        ("four-mandatory", "public"),
        ("four-mandatory", "priority_only"),
        ("four-infeasible", "public"),
        ("capacity", "public"),
        ("unsafe_fuel", "unsafe_fuel"),
        ("budget_valid", "wait"),
        ("budget_invalid", "invalid_budget"),
    }
    actual = [(r["case"], r["policy"]) for r in report["rows"]]
    if (
        len(cases) != len(manifest["cases"])
        or len(actual) != len(expected)
        or set(actual) != expected
        or any(r["status"] != "completed" for r in report["rows"])
    ):
        raise ValueError("incomplete or failed comparison matrix")
    training = []
    for row in report["rows"]:
        path = root / row["record"]
        if path.resolve().parent != root or path.name != row["record"]:
            raise ValueError("record path outside comparisons")
        record = json.loads(path.read_text())
        verify_episode(record)
        if record["sha256"] != row["sha256"] or record["metrics"] != row["metrics"]:
            raise ValueError("record identity mismatch")
        if record["split"] != "train":
            continue
        if row["case"] in cases:
            configuration = cases[row["case"]]["configuration"]
        elif row["case"] == "unsafe_fuel":
            configuration = setup["extras"]["safety"]
        elif row["case"] in {"budget_valid", "budget_invalid"}:
            configuration = setup["extras"]["budget"]
        else:
            raise ValueError("unexpected training comparison")
        if (
            record["initial"]["configuration"] != configuration
            or record["metrics"]["task_acceptance"]["accepted"]
        ):
            raise ValueError("comparison is not the declared training negative")
        training.append(record)
    destination.mkdir(parents=True)
    exports = [
        export_training(
            record,
            destination / f"negative-{index:04}.rl.jsonl",
            format="rl",
            objective="declared",
        )
        for index, record in enumerate(training)
    ]
    body = {
        "schema": "space-env.contract-negative-export/1",
        "comparisons_sha256": report["sha256"],
        "verified_records": len(report["rows"]),
        "exports": exports,
        "training_measured": False,
        "scope": "Train-only mandatory failures, native safety failure and budget-truncated records; held-out messages not exported.",
    }
    result = {**body, "sha256": digest(body)}
    atomic_write(destination / "report.json", canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_negatives(args.corpus, args.comparisons, args.output)
    print(
        canonical_json(
            {
                "verified_records": result["verified_records"],
                "exports": len(result["exports"]),
            }
        )
    )


if __name__ == "__main__":
    main()
