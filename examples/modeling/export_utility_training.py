"""Verified train-only inverse-priority/safety RL and tool-call preferences.

Preferences rank task utility, not full mission success. No model training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from space_env.industrial.agent import export_training, verify_episode
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def _preference_payload(high, low):
    # Inputs must already have passed verify_episode in export_utility_training.
    if (
        high["split"] != "train"
        or low["split"] != "train"
        or high["initial"]["configuration"] != low["initial"]["configuration"]
    ):
        raise ValueError("preferences require matching train scenarios")
    for record in (high, low):
        if (
            record["metrics"]["utility"]["violations"]
            or not record["metrics"]["terminal"]
            or record["call_budget_exhausted"]
        ):
            raise ValueError("preferences require safe complete episodes")
    chosen_return = sum(e["response"]["reward"] for e in high["events"])
    rejected_return = sum(e["response"]["reward"] for e in low["events"])
    if not chosen_return > rejected_return:
        raise ValueError("no strict task utility preference")
    left, right = high["messages"], low["messages"]
    index = next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b), None
    )
    if (
        index is None
        or index < 2
        or left[index]["role"] != "assistant"
        or right[index]["role"] != "assistant"
        or not left[index].get("tool_calls")
        or not right[index].get("tool_calls")
    ):
        raise ValueError("first divergence must be an assistant tool decision")
    return {
        "schema": "space-env.utility-preference/1",
        "prompt": left[:index],
        "chosen": [left[index]],
        "rejected": [right[index]],
        "outcomes": {
            "chosen_return": chosen_return,
            "rejected_return": rejected_return,
        },
        "source_records": {"chosen": high["sha256"], "rejected": low["sha256"]},
        "scope": "Train-only first divergent tool decision from matched verified episodes; preference by weighted task utility, not a certificate of full mission success.",
        "training_measured": False,
    }


def _read_record(root, row):
    path = root / row["record"]
    if path.resolve().parent != root.resolve() or path.name != row["record"]:
        raise ValueError("record path outside source directory")
    record = json.loads(path.read_text())
    verify_episode(record)
    if record["metrics"] != row["metrics"] or (
        "sha256" in row and record["sha256"] != row["sha256"]
    ):
        raise ValueError("record metrics or seal mismatch")
    return record


def export_utility_training(corpus, comparisons, output):
    root, other, destination = Path(corpus), Path(comparisons), Path(output)
    if destination.exists():
        raise ValueError("choose a new output directory")
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    setup = json.loads((other / "manifest.json").read_text())
    comparison = json.loads((other / "report.json").read_text())
    for value in (manifest, report, comparison):
        if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("invalid source seal")
    if (
        manifest["sources"] != sources()
        or setup["sources"] != sources()
        or report["manifest_sha256"] != manifest["sha256"]
        or comparison["manifest_sha256"] != digest(setup)
        or setup["competition_manifest_sha256"] != manifest["sha256"]
    ):
        raise ValueError("source or manifest identity mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in cases):
        raise ValueError("invalid case identifier")
    expected = {(c, p) for c in cases for p in ("wait", "public")}
    actual = [(r["case"], r["policy"]) for r in report["rows"]]
    expected_comparison = (
        {(c, "low_priority") for c in cases}
        | {("four", p) for p in ("wait", "public", "low_priority")}
        | {("unsafe_fuel", "unsafe_fuel")}
    )
    actual_comparison = [(r["case"], r["policy"]) for r in comparison["rows"]]
    if (
        len(cases) != len(manifest["cases"])
        or set(actual) != expected
        or len(actual) != len(expected)
        or set(actual_comparison) != expected_comparison
        or len(actual_comparison) != len(expected_comparison)
    ):
        raise ValueError("incomplete or duplicate attempt matrix")
    if any(r["status"] != "completed" for r in report["rows"] + comparison["rows"]):
        raise ValueError("failed attempts must be resolved before export")
    preferences = []
    rl = []
    for case in manifest["cases"]:
        if case["configuration"]["generation"]["split"] != "train":
            continue
        high = _read_record(
            root,
            next(
                r
                for r in report["rows"]
                if r["case"] == case["id"] and r["policy"] == "public"
            ),
        )
        low = _read_record(
            other,
            next(
                r
                for r in comparison["rows"]
                if r["case"] == case["id"] and r["policy"] == "low_priority"
            ),
        )
        if (
            high["initial"]["configuration"] != case["configuration"]
            or low["initial"]["configuration"] != case["configuration"]
        ):
            raise ValueError("case configuration mismatch")
        preferences.append(_preference_payload(high, low))
        rl.append((case["id"] + "--low_priority", low))
    unsafe = _read_record(
        other, next(r for r in comparison["rows"] if r["case"] == "unsafe_fuel")
    )
    if (
        unsafe["split"] != "train"
        or unsafe["initial"]["configuration"] != setup["safety"]
        or unsafe["metrics"]["utility"]["score"] != -2
        or not unsafe["metrics"]["utility"]["violations"]
    ):
        raise ValueError("invalid native safety-negative example")
    rl.append(("unsafe-fuel", unsafe))
    destination.mkdir(parents=True)
    exports = [
        export_training(record, destination / (name + ".rl.jsonl"), format="rl")
        for name, record in rl
    ]
    atomic_write(
        destination / "preferences.jsonl",
        "".join(canonical_json(row) + "\n" for row in preferences),
    )
    body = {
        "schema": "space-env.utility-training-export/1",
        "corpus_sha256": report["sha256"],
        "comparisons_sha256": comparison["sha256"],
        "rl_exports": exports,
        "preference_rows": len(preferences),
        "preferences": "preferences.jsonl",
        "training_measured": False,
        "scope": "Only verified train records and first divergent public tool decisions; no held-out messages or privileged scenario/checkpoints.",
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
    result = export_utility_training(args.corpus, args.comparisons, args.output)
    print(
        canonical_json(
            {
                "rl_exports": len(result["rl_exports"]),
                "preference_rows": result["preference_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
