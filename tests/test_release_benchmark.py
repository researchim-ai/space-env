"""Fixed suites, shared information, bounded search and fail-closed promotion."""

import copy
from functools import lru_cache

import pytest

from space_env.planning.chain import collect_chain_forecast, schedule_chains
from space_env.planning.release_benchmark import (
    POLICIES,
    Suite,
    select_schedule,
    teacher_report,
)
from space_env.planning.release_cases import release_suite
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import digest
from tests.test_chain_planning import fixture


@lru_cache(maxsize=1)
def _view():
    value = fixture()
    return collect_chain_forecast(
        MissionRuntime(value["scenario"], access=AccessPolicy(forecast=True)),
        value["scenario"],
        value["task"],
    )


def chain_view():
    return copy.deepcopy(_view())


def test_full_suite_fixed_and_disjoint():
    spec = Suite.model_validate(release_suite())
    assert len(spec.cases) == 32
    assert len({c.family for c in spec.cases}) == 8
    assert spec.model_dump(mode="json", by_alias=True) == Suite.model_validate(
        release_suite()
    ).model_dump(mode="json", by_alias=True)
    groups = {}
    for c in spec.cases:
        groups.setdefault(c.group, set()).add(c.split)
    assert all(len(x) == 1 for x in groups.values())


def test_smoke_is_explicit_subset():
    spec = Suite.model_validate(release_suite(smoke=True))
    assert len(spec.cases) == 5 and "smoke" in spec.id
    assert sum(c.negative for c in spec.cases) == 1


@pytest.mark.parametrize(
    "fault", ["geometry", "group", "clone", "zero_floor", "case_id", "extra"]
)
def test_suite_rejects_leakage_and_unsupported_intent(fault):
    s = release_suite(smoke=True)
    if fault == "geometry":
        s["cases"][0]["geometry_sha256"] = "0" * 64
    elif fault == "group":
        s["cases"][1]["group"] = s["cases"][0]["group"]
    elif fault == "clone":
        s["cases"][1] = copy.deepcopy(s["cases"][0])
        s["cases"][1]["id"] = "clone"
    elif fault == "zero_floor":
        s["cases"][0]["goals"]["min_fulfilled"] = 0
    elif fault == "case_id":
        s["cases"][0]["id"] = "../bad"
    else:
        s["budget"]["unknown"] = True
    with pytest.raises(ValueError):
        Suite.model_validate(s)


def rows(spec):
    out = []
    for c in spec["cases"]:
        n = len(c["scenario"]["definition"]["lifecycle"]["requests"])
        for p in POLICIES:
            out.append(
                {
                    "case": c["id"],
                    "policy": p,
                    "error": None,
                    "forecast_sha256": digest(c["scenario"]),
                    "native_replay_verified": True,
                    "resource_gate_passed": True,
                    "metrics": {
                        "total": n,
                        "fulfilled": 0 if c["negative"] else n,
                        "completion_fraction": 0.0 if c["negative"] else 1.0,
                        "rejections": 0,
                        "mission_goal_passed": not c["negative"],
                    },
                }
            )
    return out


def test_perfect_small_sample_does_not_prove_superiority():
    s = release_suite(smoke=True)
    report = teacher_report(s, rows(s))
    assert not report["passed_declared_gates"]
    assert any(f["reason"] == "uncertainty_or_margin_gate" for f in report["failures"])
    assert not report["expert_teacher_qualified"]


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "nan", "denominator", "forecast", "negative", "error"],
)
def test_promotion_retains_failures_and_rejects_corruption(fault):
    s = release_suite(smoke=True)
    r = rows(s)
    if fault == "missing":
        r.pop()
    elif fault == "duplicate":
        r[-1] = r[0]
    elif fault == "nan":
        r[0]["metrics"]["completion_fraction"] = float("nan")
    elif fault == "denominator":
        r[0]["metrics"]["total"] = 999
    elif fault == "forecast":
        r[0]["forecast_sha256"] = "0" * 64
    elif fault == "negative":
        row = next(
            x
            for x in r
            if x["case"] == next(c["id"] for c in s["cases"] if c["negative"])
        )
        row["metrics"].update(
            fulfilled=24, completion_fraction=1.0, mission_goal_passed=True
        )
        assert not teacher_report(s, r)["passed_declared_gates"]
        return
    else:
        r[6]["error"] = "worker_timeout"
        r[6]["total"] = 24
        r[6]["native_replay_verified"] = False
        assert not teacher_report(s, r)["passed_declared_gates"]
        return
    with pytest.raises(ValueError):
        teacher_report(s, r)


def test_default_scheduler_legacy_bytes_and_explicit_order():
    v = chain_view()
    default = schedule_chains(v)
    assert "request_order" not in default
    order = [c["id"] for c in v["chains"]]
    ordered = schedule_chains(v, order=order)
    assert ordered["request_order"] == order
    assert v == chain_view()
    with pytest.raises(ValueError):
        schedule_chains(v, order=order + [order[0]])


@pytest.mark.parametrize("policy", POLICIES)
def test_search_same_view_and_total_budget(policy):
    v = chain_view()
    result = select_schedule(v, policy, {"max_search_nodes": 10000, "max_orders": 4})
    assert result["forecast_sha256"] == v["sha256"]
    assert result["total_search_nodes"] <= 10000
    assert len(result["attempts"]) <= 4
    assert v == chain_view()


def test_unknown_policy_and_exhausted_budget_fail_closed():
    with pytest.raises(ValueError):
        select_schedule(chain_view(), "oracle")
    with pytest.raises(ValueError):
        select_schedule(chain_view(), POLICIES[2], {"max_search_nodes": 1})


def test_native_runner_and_resealed_score_tamper(tmp_path):

    from space_env.planning.release_benchmark import run_suite, verify_suite
    from space_env.simulation.snapshot import atomic_write, canonical_json

    s = release_suite(smoke=True)
    s["cases"] = s["cases"][:1]
    out = tmp_path / "benchmark"
    report = run_suite(s, out)
    assert len(report["rows"]) == 3
    assert all(r["native_replay_verified"] and not r["error"] for r in report["rows"])
    assert len({r["forecast_sha256"] for r in report["rows"]}) == 1
    assert verify_suite(out)["integrity_passed"]
    assert not report["teacher"]["passed_declared_gates"]
    report["rows"][0]["metrics"]["fulfilled"] += 1
    report["sha256"] = digest({k: v for k, v in report.items() if k != "sha256"})
    atomic_write(out / "report.json", canonical_json(report))
    with pytest.raises(ValueError):
        verify_suite(out)
