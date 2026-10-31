"""Portable bounded calendar benchmarks with declared split/lineage checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from space_env.data_generators.reference_operations import _runtime
from space_env.reference.optimal import SCOPE, solve_small_downlinks
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def evaluate_calendar_suite(cases: list[dict], *, max_states: int = 65536, total_states: int = 262144) -> dict:
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("suite requires 1-32 cases")
    if isinstance(total_states, bool) or not isinstance(total_states, int) or not 1 <= total_states <= 2097152:
        raise ValueError("total_states must be an integer in [1,2097152]")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or not 1 <= max_states <= 65536:
        raise ValueError("max_states must be an integer in [1,65536]")
    ids, lineages, hashes = set(), {}, {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "lineage", "split", "spec"}:
            raise ValueError("each case requires id,lineage,split,spec")
        if any(not isinstance(case[k], str) or not case[k].strip() for k in ("id", "lineage")):
            raise ValueError("nonempty case ID and lineage required")
        if case["id"] in ids or case["split"] not in {"train", "test"}:
            raise ValueError("duplicate case ID or invalid split")
        ids.add(case["id"])
        sha = digest(case["spec"])
        for mapping, key in ((lineages, case["lineage"]), (hashes, sha)):
            if key in mapping and mapping[key] != case["split"]:
                raise ValueError("scenario hash or declared lineage crosses train/test splits")
            mapping[key] = case["split"]
    remaining, rows = total_states, []
    for case in cases:
        if remaining == 0:
            raise ValueError("suite state budget exhausted")
        solution = solve_small_downlinks(case["spec"], max_states=min(max_states, remaining))
        remaining -= solution["states_explored"]
        rows.append({**copy.deepcopy(case), "solution": solution})
    summary = {}
    for split in ("train", "test"):
        selected = [r["solution"] for r in rows if r["split"] == split]
        optimum = sum(int(r["report"]["on_time_bits"]) for r in selected)
        baseline = sum(int(r["baseline"]["report"]["on_time_bits"]) for r in selected)
        summary[split] = {"cases": len(selected), "optimal_bits": optimum, "baseline_bits": baseline,
                          "regret_bits": optimum-baseline,
                          "relative_regret": (optimum-baseline)/optimum if optimum else 0.}
    body = {"schema": "space-env.reference-calendar-benchmark/1", "scope": SCOPE, "runtime": _runtime(),
            "limits": {"max_states": max_states, "total_states": total_states},
            "states_explored": total_states-remaining, "cases": rows, "summary": summary,
            "expert_demonstrations": False}
    return {**body, "sha256": digest(body)}


def save_calendar_benchmark(path: str | Path, report: dict):
    atomic_write(path, canonical_json(report))


def verify_calendar_benchmark(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text())
    body = {k: v for k, v in report.items() if k != "sha256"}
    if report.get("schema") != "space-env.reference-calendar-benchmark/1" or report.get("sha256") != digest(body):
        raise ValueError("invalid or corrupt calendar benchmark")
    if report["runtime"] != _runtime():
        return {"ok": False, "reason": "runtime_mismatch"}
    cases = [{k: r[k] for k in ("id", "lineage", "split", "spec")} for r in report["cases"]]
    actual = evaluate_calendar_suite(cases, **report["limits"])
    return {"ok": actual == report, "reason": "verified" if actual == report else "result_mismatch"}
