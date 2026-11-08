"""Explicit local ablation on the frozen demand suite; no CI integration.

python examples/network/compare_optimizer.py --output /tmp/optimizer-comparison
"""

import argparse

from space_env.api.network_workspace import collect_network_forecast
from space_env.planning.network_benchmark import NetworkSuite
from space_env.planning.network_demand_benchmark import demand_suite, workload
from space_env.planning.network_optimal import optimize_network, verify_exact_schedule
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.network_evidence import (
    run_scale_case,
    seal,
    source_manifest,
    validate_row,
)


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    suite = demand_suite()
    rows = []
    for case in NetworkSuite.model_validate(suite).cases:
        view = collect_network_forecast(case.configuration)
        atomic_write(
            args.output / (case.id + "--forecast.json"), canonical_json(view) + "\n"
        )
        ablation = []
        for pruning in (False, True):
            plan = optimize_network(view, max_nodes=50000, max_hops=0, pruning=pruning)
            verify_exact_schedule(view, plan)
            atomic_write(
                args.output / (case.id + f"--pruning-{pruning}.json"),
                canonical_json(plan) + "\n",
            )
            ablation.append(
                {
                    k: plan[k]
                    for k in (
                        "pruning",
                        "objective_score",
                        "upper_bound_score",
                        "optimal",
                        "search_nodes",
                        "pruned_states",
                        "candidates",
                    )
                }
            )
        work = workload(case, "exact_small", view)
        native = run_scale_case(work, timeout_sec=120)
        validate_row(native, work)
        rows.append(
            {
                "case": case.id,
                "split": case.split,
                "forecast_sha256": view["sha256"],
                "ablation": ablation,
                "native": native,
            }
        )
        print(case.id, native["status"], flush=True)
    report = seal(
        {
            "schema": "space-env.optimizer-ablation/1",
            "suite": suite,
            "suite_sha256": digest(suite),
            "model_sources": source_manifest(),
            "rows": rows,
            "scope": "same forecasts, 50000 node budgets; optimum only in conservative discrete model; fresh native exact executions",
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report) + "\n")
    return 0 if all(r["native"]["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
