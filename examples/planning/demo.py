"""A baseline controls the EO/CPU/ISL/ground mission; no scripted action plan."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.planning import PlanningRun
from space_env.tasks import export_dataset


def build(mode="nominal"):
    path = Path(__file__).resolve().parents[1] / "agent_task/demo.py"
    spec = importlib.util.spec_from_file_location("agent_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario, task, _ = module.build(mode)
    scenario["lineage"] = "B07-eo-" + mode
    task.update(decision_sec=2.0, max_decisions=64)
    if mode == "ttl":
        # No alternative long-TTL template can bypass this negative case.
        for action in task["actions"]:
            for command in action.get("commands", []):
                if (
                    command.get("product_id") == "raw-after"
                    and command["kind"] == "relay"
                ):
                    command["relay"]["ttl_sec"] = 0.4
    return scenario, task


def main(directory, mode="nominal", policy="route_aware_rolling", export=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scenario, task = build(mode)
    config = {"name": policy, "max_nodes": 512, "seed": 19}
    for name, value in [("scenario", scenario), ("task", task), ("policy", config)]:
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    run = PlanningRun(scenario, task, config)
    run.run(3)
    run.save(directory / "prefix.json")
    run = PlanningRun.load(directory / "prefix.json")
    metrics = run.run()
    record = run.save(directory / "run.json")
    assert PlanningRun.from_checkpoint(record).checkpoint() == record
    episode = run.session.save(directory / "episode.json")
    for format in ["rl", "sft"] if export else []:
        export_dataset(episode, directory / f"{format}.jsonl", format=format)
    report = run.session._runtime._backend.report()  # Trusted evaluator only.
    for ledger in [report["inventory_ledger"], report["lifecycle"]["ledger"]]:
        assert all(abs(v["balance_error_bits"]) < 1e-6 for v in ledger.values())
    assert abs(report["dtn"]["wire_balance_error_bits"]) < 1e-6
    for ledger in report["spacecraft_bus"]["ledgers"].values():
        assert abs(ledger["battery_balance_error_wh"]) < 1e-9
        assert abs(ledger["thermal_balance_error_j"]) < 1e-6
    if policy == "route_aware_rolling":
        assert metrics["fulfilled"] == (1 if mode == "ttl" else 2), metrics
    result = {
        "mode": mode,
        "policy": policy,
        "fulfilled": metrics["fulfilled"],
        "total": metrics["total"],
        "rejections": metrics["rejections"],
        "search_nodes": metrics["search_nodes"],
        "decisions": metrics["decisions"],
        "replay": True,
        "conservation": True,
        "qualification": "synthetic causal baseline, not an expert policy",
    }
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        sys.argv[2] if len(sys.argv) > 2 else "nominal",
        sys.argv[3] if len(sys.argv) > 3 else "route_aware_rolling",
        "--export" in sys.argv,
    )
