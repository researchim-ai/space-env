"""Verify a generated phasing corpus, then export train-only SFT/RL JSONL.

Public synthetic example. Validation/test messages never enter output datasets.
Existing records and output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from space_env.industrial.agent import export_training, training_success, verify_episode
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def export_corpus(corpus, output, *, objective="native"):
    root, destination = Path(corpus).resolve(), Path(output)
    if destination.exists():
        raise ValueError("choose a new export directory")
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    for value in (manifest, report):
        if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("invalid corpus seal")
    if (
        manifest["sources"] != sources()
        or report["manifest_sha256"] != manifest["sha256"]
    ):
        raise ValueError("source or manifest mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in cases):
        raise ValueError("invalid case identifier")
    if len(cases) != len(manifest["cases"]):
        raise ValueError("duplicate cases")
    expected = {(name, policy) for name in cases for policy in ("wait", "public")}
    actual = [(r["case"], r["policy"]) for r in report["rows"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("incomplete or duplicate attempt matrix")
    training = []
    for row in report["rows"]:
        if row["status"] != "completed":
            raise ValueError("retain failed attempts; repair corpus before export")
        path = root / row["record"]
        if path.resolve().parent != root or path.name != row["record"]:
            raise ValueError("record path outside corpus")
        record = json.loads(path.read_text())
        case = cases[row["case"]]
        if (
            record["sha256"] != row["sha256"]
            or record["initial"]["configuration"] != case["configuration"]
            or record["split"] != case["configuration"]["generation"]["split"]
            or record["metrics"] != row["metrics"]
        ):
            raise ValueError("record and case identity mismatch")
        verify_episode(record)
        if record["split"] == "train":
            training_success(record, objective=objective)
            training.append((row["case"] + "--" + row["policy"], record))
    destination.mkdir(parents=True)
    exports = []
    for name, record in training:
        exports.append(
            export_training(
                record,
                destination / (name + ".rl.jsonl"),
                format="rl",
                objective=objective,
            )
        )
        eligible = training_success(record, objective=objective) and all(
            "error" not in e["response"]
            and e["response"].get("feedback", {}).get("accepted") is not False
            for e in record["events"]
        )
        if eligible:
            exports.append(
                export_training(
                    record,
                    destination / (name + ".sft.jsonl"),
                    format="sft",
                    objective=objective,
                )
            )
    body = {
        "schema": "space-env.phasing-corpus-export/1",
        "corpus_manifest_sha256": manifest["sha256"],
        "corpus_report_sha256": report["sha256"],
        "verified_records": len(report["rows"]),
        "train_records": len(training),
        "exports": exports,
        "selection": "RL: all train records. SFT: successful error-free train records only. No development/test messages exported.",
        "objective": objective,
        "training_measured": False,
        "external_llm_measured": False,
    }
    result = {**body, "sha256": digest(body)}
    atomic_write(destination / "report.json", canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["native", "declared"], default="native")
    args = parser.parse_args()
    result = export_corpus(args.corpus, args.output, objective=args.objective)
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
