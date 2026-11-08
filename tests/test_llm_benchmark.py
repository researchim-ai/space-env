"""Frozen comparison accounting; scripted policies never become LLM scores."""

import copy

import pytest

from space_env.llm.benchmark import POLICIES, summarize
from space_env.planning.network_demand_benchmark import demand_suite


def rows():
    suite = demand_suite()
    result = [
        {
            "case": c["id"],
            "group": c["group"],
            "split": c["split"],
            "policy": p,
            "priority_total": sum(
                r["priority"] for r in c["configuration"]["mission"]["requests"]
            ),
            "status": "error",
            "metrics": None,
        }
        for c in suite["cases"]
        for p in POLICIES
    ]
    return suite, result


def test_failed_episodes_are_unknown_and_preserved_in_every_denominator():
    suite, data = rows()
    report = summarize(suite, data)
    assert all(
        r["value_lower"] == 0
        and r["value_upper"] == 1
        and r["completed"] == 0
        and r["rows"] == 4
        for r in report["aggregates"]
    )
    assert report["competitor_superiority_established"] is False
    assert report["llm_training_gain_established"] is False
    assert all(
        r["privileged"] == (r["policy"] == "reference_schedule")
        and not r["external_llm_measured"]
        for r in report["aggregates"]
    )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "split", "group", "denominator", "value"]
)
def test_comparison_rejects_incomplete_matrix_and_mislabelled_groups(fault):
    suite, data = rows()
    if fault == "missing":
        data.pop()
    elif fault == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif fault == "split":
        data[0]["split"] = "test"
    elif fault == "group":
        data[0]["group"] = "renamed"
    elif fault == "denominator":
        data[0]["priority_total"] = 1
    else:
        data[0].update(status="completed", metrics={"weighted_fulfilled": -1})
    with pytest.raises(ValueError):
        summarize(suite, data)
