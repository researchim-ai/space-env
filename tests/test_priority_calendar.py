"""Weighted DP vs independent integer-time exhaustive scheduling and native replay."""

import copy

import numpy as np
import pytest

from space_env.reference.priority_calendar import solve_priority_calendar
from space_env.validation.tradeoff_study import calendar_suite


@pytest.fixture(scope="module")
def cases():
    return calendar_suite()


def test_explicit_greedy_gaps_and_negative_cases(cases):
    for case in cases:
        result = solve_priority_calendar(case["spec"])
        expected = {
            "bulk_vs_pair": (12, 11),
            "release_and_gap": (24, 17),
            "infeasible": (0, 0),
        }[case["family"]]
        assert (result["value"], result["baseline"]["value"]) == expected
        assert result["optimality_proven_in_scope"]
        assert not result["full_mission_optimality"]
        delivered = {
            p["id"]
            for p in result["report"]["products"]
            if p["status"] == "delivered_on_time"
        }
        assert delivered == set(result["selected_products"])
        assert result["value"] == sum(
            p["priority"] for p in case["spec"]["products"] if p["id"] in delivered
        )


def exhaustive(products, horizon, outage):
    # Separate discrete start-time enumeration, not the production slot/DP code.
    best = 0

    def visit(clock, remaining, value):
        nonlocal best
        best = max(best, value)
        for i in remaining:
            p = products[i]
            duration = int(p["size_bits"] / 100000)
            for start in range(
                max(clock, int(p["ready_sec"])),
                min(horizon, int(p["deadline_sec"])) - duration + 1,
            ):
                end = start + duration
                if max(start, outage[0]) < min(end, outage[1]):
                    continue
                visit(end, remaining - {i}, value + p["priority"])

    visit(0, set(range(len(products))), 0)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_weighted_optimum_against_exhaustive_start_times(cases, seed):
    spec = copy.deepcopy(cases[0]["spec"])
    spec["horizon_sec"] = 12.0
    for actor in spec["vehicles"].values():
        actor["rate_bps"] = 100000.0
    rng = np.random.default_rng(seed)
    spec["products"] = [
        {
            "id": str(i),
            "actor": "a" if i % 2 else "b",
            "size_bits": 100000 * int(rng.integers(1, 5)),
            "priority": int(rng.integers(0, 13)),
            "ready_sec": float(rng.integers(0, 4)),
            "deadline_sec": 12.0,
        }
        for i in range(3)
    ]
    spec["outages"] = [{"station": "Ground", "start_sec": 3.0, "end_sec": 5.0}]
    result = solve_priority_calendar(spec)
    assert result["value"] == exhaustive(spec["products"], 12, (3, 5))


@pytest.mark.parametrize(
    "fault", ["power", "solar", "quota", "processing", "terminal", "radio", "dtn"]
)
def test_outside_scope_never_receives_certificate(cases, fault):
    spec = copy.deepcopy(cases[0]["spec"])
    if fault == "power":
        spec["vehicles"]["a"]["tx_power_w"] = 1.0
    elif fault == "solar":
        spec["vehicles"]["a"]["solar_area_m2"] = 1.0
    elif fault == "quota":
        spec["vehicles"]["a"]["max_tx_sec"] = 1.0
    elif fault == "processing":
        spec["products"][0]["processing_sec"] = 1.0
    elif fault == "terminal":
        spec["stations"][0]["terminals"] = 2
    else:
        spec["radio_network" if fault == "radio" else "dtn_network"] = {}
    with pytest.raises(ValueError):
        solve_priority_calendar(spec)


@pytest.mark.parametrize("budget", [True, 0, 1.0, 65537])
def test_invalid_budget(cases, budget):
    with pytest.raises(ValueError):
        solve_priority_calendar(cases[0]["spec"], max_states=budget)


def test_budget_exhaustion_cannot_claim_optimum(cases):
    with pytest.raises(ValueError, match="no optimality certificate"):
        solve_priority_calendar(cases[0]["spec"], max_states=1)


def test_input_is_owned_and_repeatable(cases):
    spec = copy.deepcopy(cases[0]["spec"])
    before = copy.deepcopy(spec)
    assert solve_priority_calendar(spec) == solve_priority_calendar(spec)
    assert spec == before
