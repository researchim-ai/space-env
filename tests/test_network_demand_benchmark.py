import copy

import pytest

from space_env.planning import network_demand_benchmark as benchmark
from space_env.planning.network_benchmark import NetworkSuite, network_suite


def test_new_suite_is_frozen_disjoint_and_demand_distributions_vary(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Outcome/forecast filtering in suite factory")

    monkeypatch.setattr(benchmark, "collect_network_forecast", forbidden)
    previous = network_suite()
    suite = benchmark.demand_suite()
    assert suite == benchmark.demand_suite()
    assert previous == network_suite()
    assert len(suite["cases"]) == 12
    assert {c["geometry_sha256"] for c in suite["cases"]}.isdisjoint(
        c["geometry_sha256"] for c in previous["cases"]
    )
    assert {len(c["configuration"]["mission"]["requests"]) for c in suite["cases"]} == {
        3,
        5,
    }
    assert all(c["configuration"]["max_hops"] == 0 for c in suite["cases"])
    assert {c["configuration"]["mission"]["memory_mbit"] for c in suite["cases"]} == {
        4.0,
        8.0,
    }
    assert {
        c["configuration"]["mission"]["stations"][0]["terminals"]
        for c in suite["cases"]
    } == {1, 2}
    NetworkSuite.model_validate(suite)


def comparison_rows():
    suite = benchmark.demand_suite()
    rows = []
    for c in suite["cases"]:
        total = len(c["configuration"]["mission"]["requests"])
        priority_total = sum(
            r["priority"] for r in c["configuration"]["mission"]["requests"]
        )
        for method in benchmark.METHODS:
            row = {
                "status": "completed",
                "total": total,
                "weighted_fulfilled": 11,
                "priority_total": priority_total,
            }
            if method == "exact_small":
                row["optimizer"] = {
                    "upper_bound_score": 12 * (total + 1) + 2,
                    "optimal": True,
                }
                row["weighted_fulfilled"] = 12
            rows.append(
                {
                    "case": c["id"],
                    "group": c["group"],
                    "split": c["split"],
                    "method": method,
                    "forecast_sha256": "f" * 64,
                    "result": row,
                }
            )
    return suite, rows


def test_summary_keeps_native_scores_separate_from_forecast_bounds():
    suite, rows = comparison_rows()
    result = benchmark.summarize(suite, rows)
    assert len(result["paired"]) == 48
    baseline = next(r for r in result["paired"] if r["method"] == "priority")
    assert baseline["native_weighted_fulfilled"] == 11
    assert baseline["forecast_value_upper_bound"] == 12
    assert baseline["native_gap_to_forecast_bound"] == 1
    rows[0]["result"] = {
        "status": "timeout",
        "total": rows[0]["result"]["total"],
        "native_fulfilled": None,
    }
    result = benchmark.summarize(suite, rows)
    aggregate = next(
        a
        for a in result["aggregates"]
        if a["split"] == "train" and a["method"] == "priority"
    )
    assert aggregate["rows"] == 4 and aggregate["completed_rows"] == 3
    assert aggregate["mean_value_lower_bound"] < aggregate["mean_value_upper_bound"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "information", "split", "denominator"]
)
def test_comparison_admission_preserves_full_equal_information_matrix(fault):
    suite, rows = comparison_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "information":
        rows[0]["forecast_sha256"] = "0" * 64
    elif fault == "split":
        rows[0]["split"] = "test"
    else:
        rows[0]["result"]["total"] = 1
    with pytest.raises(ValueError):
        benchmark.summarize(suite, rows)
