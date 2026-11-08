"""Native-replayed paired evaluation; resample geometries, never decision ticks.

Intervals describe the declared public case population, not unseen missions or
physical model accuracy. No external competitor or learning claim is inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from space_env.llm.episode import seal, sources, verify_record
from space_env.llm.mission_suite import TASK, compile_suite, validate_suite
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.workspace import MissionWorkspace


def _interval(values, groups, samples, seed):
    """Equal case weights, paired cluster bootstrap, deterministic RNG."""
    keys = sorted(set(groups))
    if len(keys) < 2:
        return None
    clustered = {key: [] for key in keys}
    for value, key in zip(values, groups):
        clustered[key].append(value)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [v for key in rng.choices(keys, k=len(keys)) for v in clustered[key]]
        draws.append(sum(selected) / len(selected))
    draws.sort()
    return [
        draws[math.floor(0.025 * (samples - 1))],
        draws[math.ceil(0.975 * (samples - 1))],
    ]


def evaluate_pairs(suite, selected, rows, *, bootstrap_samples=2000, seed=0):
    """Every selected case must have baseline/candidate, including explicit errors.

    A completed row contains a full native LLMEpisode. An error row has only
    case, arm, status and error_type; it is unknown rather than a zero score.
    Pairing requires byte-identical initial checkpoints and dialogue budgets.
    """
    validate_suite(suite)
    source_identity = sources()
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be 100..10000")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32")
    cases = {c["id"]: c for c in suite["cases"]}
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= cases.keys()
    ):
        raise ValueError("Invalid selected cases")
    expected = {(c, arm) for c in selected for arm in ("baseline", "candidate")}
    if len(rows) != len(expected) or {(r["case"], r["arm"]) for r in rows} != expected:
        raise ValueError("Complete unique paired matrix required")
    indexed, clean, initial_worlds = {}, [], {}
    for row in rows:
        case = cases[row["case"]]
        item = {k: row[k] for k in ("case", "arm", "status")}
        if row["status"] == "error":
            if (
                set(row) != {"case", "arm", "status", "error_type"}
                or not isinstance(row["error_type"], str)
                or not row["error_type"]
            ):
                raise ValueError(
                    "Error rows require an explicit error_type and no metrics"
                )
            item.update(error_type=row["error_type"], value=None)
        elif row["status"] == "completed":
            if set(row) != {"case", "arm", "status", "record"}:
                raise ValueError("Completed rows require a native record")
            record = row["record"]
            verify_record(record)
            cfg = record["configuration"]
            checkpoint = record["initial_checkpoint"]
            actual = checkpoint["scenario"]["metadata"]["network_workspace"][
                "configuration"
            ]
            if (
                actual != case["configuration"]
                or cfg["group"] != case["group"]
                or cfg["split"] != case["split"]
                or cfg["task_text"] != TASK
            ):
                raise ValueError("Native case/configuration/split mismatch")
            method = checkpoint["scenario"]["metadata"]["network_workspace"][
                "planning_method"
            ]
            key = (case["id"], method)
            if key not in initial_worlds:
                workspace = MissionWorkspace.network(
                    case["configuration"], planning_method=method
                )
                try:
                    initial_worlds[key] = workspace.operator.session.checkpoint(
                        compact=True
                    )
                finally:
                    workspace.close()
            if checkpoint != initial_worlds[key]:
                raise ValueError(
                    "Native world/catalogue differs from frozen case compilation"
                )
            denominator = sum(r["priority"] for r in actual["mission"]["requests"])
            value = record["metrics"]["weighted_fulfilled"] / denominator
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid native normalized priority")
            item.update(
                value=value,
                success=record["metrics"]["success"],
                episode_sha256=record["sha256"],
                policy_origin=cfg["policy_origin"],
                failed_calls=record["metrics"]["failed_calls"],
                tool_calls=record["metrics"]["tool_calls"],
            )
        else:
            raise ValueError("Unknown row status")
        indexed[row["case"], row["arm"]] = row
        clean.append(item)
    for case in selected:
        pair = [indexed[case, a] for a in ("baseline", "candidate")]
        if all(r["status"] == "completed" for r in pair):
            left, right = (r["record"] for r in pair)
            if left["initial_checkpoint"] != right["initial_checkpoint"] or any(
                left["configuration"][k] != right["configuration"][k]
                for k in ("task_text", "max_calls", "max_dialogue_bytes")
            ):
                raise ValueError(
                    "Unmatched initial world, catalogue or information/call budget"
                )
    clean.sort(key=lambda r: (r["case"], r["arm"]))
    by_key = {(r["case"], r["arm"]): r for r in clean}
    aggregates = []
    for split in ("overall", "train", "development", "test"):
        ids = [c for c in selected if split == "overall" or cases[c]["split"] == split]
        if not ids:
            continue
        deltas, groups = [], []
        lower, upper = 0.0, 0.0
        for case in ids:
            before, after = (
                by_key[case, a]["value"] for a in ("baseline", "candidate")
            )
            lower += (after if after is not None else 0) - (
                before if before is not None else 1
            )
            upper += (after if after is not None else 1) - (
                before if before is not None else 0
            )
            if before is not None and after is not None:
                deltas.append(after - before)
                groups.append(cases[case]["geometry_sha256"])
        complete = len(deltas) == len(ids)
        aggregates.append(
            {
                "split": split,
                "expected_pairs": len(ids),
                "observed_pairs": len(deltas),
                "geometry_groups": len({cases[c]["geometry_sha256"] for c in ids}),
                "improved": sum(d > 1e-12 for d in deltas),
                "tied": sum(abs(d) <= 1e-12 for d in deltas),
                "regressed": sum(d < -1e-12 for d in deltas),
                "unknown_pairs": len(ids) - len(deltas),
                "delta_lower": lower / len(ids),
                "delta_upper": upper / len(ids),
                "mean_delta": sum(deltas) / len(ids) if complete else None,
                "cluster_bootstrap_95": _interval(
                    deltas, groups, bootstrap_samples, seed
                )
                if complete
                else None,
            }
        )
    if sources() != source_identity:
        raise ValueError("Sources changed during paired evaluation")
    return seal(
        {
            "schema": "space-env.paired-native-evaluation/1",
            "suite_sha256": suite["sha256"],
            "selected": list(selected),
            "rows": clean,
            "aggregates": aggregates,
            "model_sources": source_identity,
            "bootstrap": {
                "unit": "geometry",
                "samples": bootstrap_samples,
                "seed": seed,
            },
            "competitor_superiority_established": False,
            "llm_training_gain_established": False,
            "scope": "native matched catalogue/information/tool budgets; public regression geometries; paired cluster percentile bootstrap conditional on declared population, not blind/OOD or repeated model-seed uncertainty; token, latency and model identity require provider bundle audit",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for row in manifest["rows"]:
        row = dict(row)
        if row["status"] == "completed":
            row["record"] = json.loads(
                (args.manifest.parent / row.pop("episode")).read_text()
            )
        rows.append(row)
    report = evaluate_pairs(compile_suite(), manifest["selected"], rows)
    if args.output.exists():
        raise ValueError("Output already exists")
    atomic_write(args.output, canonical_json(report))
    print(
        canonical_json({"report": str(args.output), "aggregates": report["aggregates"]})
    )


if __name__ == "__main__":
    main()
