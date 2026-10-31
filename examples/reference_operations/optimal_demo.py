"""A provable throughput counterexample, matching-class regret and portable suite."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from demo import build_spec

from space_env.reference.benchmark import (
    evaluate_calendar_suite,
    save_calendar_benchmark,
    verify_calendar_benchmark,
)
from space_env.reference.optimal import solve_small_downlinks


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec["outages"] = []
    for v in spec["vehicles"].values():
        v.update(base_power_w=0, tx_power_w=0)
    spec["products"] = [
        {"id": "urgent", "actor": "a", "size_bits": 1e6, "deadline_sec": 4, "priority": 10},
        {"id": "early", "actor": "b", "size_bits": 2e6, "deadline_sec": 2},
        {"id": "later", "actor": "b", "size_bits": 2e6, "deadline_sec": 4}]
    result = solve_small_downlinks(spec)
    for name, value in (("spec.json", spec), ("plan.json", result["plan"]), ("oracle.json", result)):
        (directory / name).write_text(json.dumps(value, allow_nan=False))
    other = copy.deepcopy(spec)
    other["products"][0]["deadline_sec"] = 5
    # Both variants share a family and stay in one split; no held-out claim.
    cases = [{"id": f"deadline-{i}", "lineage": "synthetic-calendar-counterexample", "split": "test", "spec": case}
             for i, case in enumerate((spec, other))]
    benchmark = evaluate_calendar_suite(cases)
    path = directory / "benchmark.json"
    save_calendar_benchmark(path, benchmark)
    verified = verify_calendar_benchmark(path)
    if result["regret_bits"] != 1e6 or not verified["ok"]:
        raise RuntimeError("calendar benchmark failed")
    print(json.dumps({"scope": result["scope"], "optimum_bits": result["report"]["on_time_bits"],
                      "baseline_bits": result["baseline"]["report"]["on_time_bits"],
                      "project_baseline_comparable": result["project_baseline"]["comparable"],
                      "regret_bits": result["regret_bits"], "benchmark_summary": benchmark["summary"],
                      "verification": verified}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
