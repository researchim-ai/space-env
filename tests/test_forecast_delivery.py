"""Forecasts are explicit permissions and values, not native rollout oracles."""

import copy
import json
from unittest.mock import patch

import pytest

from space_env.eval.statistics import assign_split
from space_env.planning.forecast_benchmark import (
    ForecastBenchmarkSpec,
    forecast_teacher_report,
    run_forecast_benchmark,
    verify_forecast_benchmark,
)
from space_env.planning.forecast_delivery import (
    DeliveryForecast,
    collect_delivery_forecast,
    forecast_mission_plan,
    schedule_delivery,
)
from space_env.planning.mission_bundle import publish_plan_bundle
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime, reference_scenario
from space_env.simulation.snapshot import canonical_json
from tests.test_reference_operations import spec as native_fixture


def case(variant=0, negative=False, split="holdout"):
    d = copy.deepcopy(native_fixture.__wrapped__())
    d["horizon_sec"] = 30.0
    d["vehicles"]["a"]["rate_bps"] += float(variant) * 10000
    d["products"][0].update(destination="Ground", deadline_sec=3.0)
    d["products"][1].update(destination="Ground", size_bits=1e6, deadline_sec=1.0)
    if negative:
        d["outages"] = [{"station": "Ground", "start_sec": 0.0, "end_sec": 30.0}]
    for i in range(100):
        lineage = f"forecast-contention-{variant}-{negative}-{i}"
        if assign_split(lineage, "") == split:
            break
    scenario = reference_scenario(
        d, id=f"forecast-{variant}-{negative}", lineage=lineage
    )
    task = {
        "id": "forecast-contention",
        "runtime_audit": "compact",
        "policy_origin": "scripted",
        "roles": {a: {"actors": [a]} for a in ("a", "b")},
        "decision_sec": 1.0,
        "max_decision_sec": 1.0,
        "max_decisions": 31,
        "max_actions": 3,
        "max_joint_actions": 2,
        "actions": [
            {
                "id": "send-" + p["id"],
                "role": p["actor"],
                "kind": "submit",
                "commands": [
                    {
                        "command_id": "send-" + p["id"],
                        "actor_id": p["actor"],
                        "kind": "downlink",
                        "duration_sec": 1.0,
                        "product_id": p["id"],
                        "destination_id": "Ground",
                    }
                ],
            }
            for p in d["products"]
        ],
    }
    return {
        "id": f"case-{variant}-{negative}",
        "family": "fixed-contention",
        "scenario": scenario,
        "task": task,
        "split": split,
        "negative": negative,
        "min_fulfilled": 1,
    }


def view_for(c):
    r = MissionRuntime(c["scenario"], access=AccessPolicy(forecast=True))
    return collect_delivery_forecast(r, c["scenario"], c["task"])


def test_forecast_requires_explicit_authorization_and_initial_binding():
    c = case()
    r = MissionRuntime(c["scenario"])
    with pytest.raises(PermissionError):
        collect_delivery_forecast(r, c["scenario"], c["task"])
    r = MissionRuntime(c["scenario"], access=AccessPolicy(forecast=True))
    r.advance(1.0, expected_revision=r.revision, idempotency_key="move")
    with pytest.raises(ValueError, match="initial"):
        collect_delivery_forecast(r, c["scenario"], c["task"])


def test_bridge_and_scheduler_never_use_truth_admission_or_rollout():
    c = case()
    r = MissionRuntime(c["scenario"], access=AccessPolicy(forecast=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("private/rollout oracle used")

    with (
        patch.object(r, "observe", forbidden),
        patch.object(r, "report", forbidden),
        patch.object(r._backend, "validate_command", forbidden),
        patch.object(r, "advance", forbidden),
    ):
        v = collect_delivery_forecast(r, c["scenario"], c["task"])
        with patch("space_env.tasks.TaskSession.step", forbidden):
            result = forecast_mission_plan(c["scenario"], c["task"], v)
    assert result["plan"]["origin"] == "declared"
    assert result["plan"]["expected_native_sha256"] is None
    assert result["plan"]["source_planning_sha256"] is None
    assert result["schedule"]["predicted_fulfilled"] == 2


def test_deadline_scheduler_reserves_terminal_for_urgent_product(tmp_path):
    c = case()
    proposal = forecast_mission_plan(
        c["scenario"], c["task"], view_for(c), goals={"min_fulfilled": 2}
    )
    assert proposal["schedule"]["steps"][0]["action_ids"] == ["send-b"]
    assert proposal["schedule"]["steps"][1]["action_ids"] == ["send-a"]
    x = PlanExecution(proposal["plan"])
    assert x.run()["complete"]
    bundle = publish_plan_bundle(proposal["plan"], x.checkpoint(), tmp_path / "dataset")
    assert bundle["mission_goal_passed"]
    cert = json.loads((tmp_path / "dataset/certificate.json").read_text())
    assert cert["fulfilled"] == 2


def test_view_and_result_ownership_and_determinism():
    c = case()
    v = view_for(c)
    before = v.payload
    v.view()["products"].clear()
    a = schedule_delivery(v)
    a["steps"][0]["action_ids"].clear()
    assert v.payload == before and schedule_delivery(v)["steps"][0]["action_ids"] == [
        "send-b"
    ]
    assert schedule_delivery(v) == schedule_delivery(v)


def test_terminal_capacity_and_actor_exclusion():
    c = case()
    c["scenario"]["definition"]["stations"][0]["terminals"] = 2
    result = forecast_mission_plan(c["scenario"], c["task"], view_for(c))
    assert result["schedule"]["steps"][0]["action_ids"] == ["send-a", "send-b"]
    assert all(len(s["action_ids"]) <= 2 for s in result["schedule"]["steps"])


def test_outage_empty_pairs_and_full_denominator():
    c = case(negative=True)
    v = view_for(c)
    result = schedule_delivery(v)
    assert len(v.view()["pairs"]) == 2 and all(
        not p["windows"] for p in v.view()["pairs"]
    )
    assert result["total"] == 2 and result["predicted_fulfilled"] == 0
    assert len(result["unserved_reasons"]) == 2
    assert all(s["action_ids"] == ["wait"] for s in result["steps"])


def test_energy_budget_does_not_credit_solar_prediction():
    c = case()
    v = view_for(c).view()
    v["vehicles"]["b"].update(battery_wh=0.01, tx_power_w=10000.0)
    result = schedule_delivery(DeliveryForecast(canonical_json(v)))
    assert all("send-b" not in s["action_ids"] for s in result["steps"])


def test_quota_and_ready_deadline_are_not_relaxed():
    c = case()
    v = view_for(c).view()
    v["vehicles"]["a"]["max_tx_sec"] = 0.5
    v["products"][1]["ready_sec"] = 0.5
    result = schedule_delivery(DeliveryForecast(canonical_json(v)))
    assert result["predicted_fulfilled"] == 0 and result["total"] == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v["pairs"].pop(),
        lambda v: v["products"].append(copy.deepcopy(v["products"][0])),
        lambda v: v["vehicles"]["a"].__setitem__("seed", 42),
        lambda v: v["pairs"][0]["binding"].__setitem__("revision", 1),
        lambda v: v["pairs"][0]["windows"].__setitem__(
            0, {"start_sec": True, "end_sec": 30.0}
        ),
        lambda v: v["stations"].__setitem__("Ground", True),
        lambda v: v.__setitem__("decision_sec", 0.0),
    ],
)
def test_corrupt_forecast_view_rejected(mutate):
    v = view_for(case()).view()
    mutate(v)
    with pytest.raises(ValueError):
        DeliveryForecast(canonical_json(v))


def test_budget_duplicate_json_and_wrong_task_binding():
    c = case()
    v = view_for(c)
    with pytest.raises(ValueError, match="budget"):
        schedule_delivery(v, {"max_slot_products": 1})
    with pytest.raises(ValueError, match="window-check"):
        schedule_delivery(v, {"max_window_checks": 1})
    with pytest.raises(ValueError, match="duplicate"):
        DeliveryForecast('{"schema":1,"schema":2}')
    t = copy.deepcopy(c["task"])
    t["id"] = "other"
    with pytest.raises(ValueError, match="binding"):
        forecast_mission_plan(c["scenario"], t, v)


@pytest.mark.parametrize("kind", ["thermal", "processing", "duration", "rf"])
def test_unsupported_physics_never_silently_downgraded(kind):
    c = case()
    s = c["scenario"]
    t = c["task"]
    if kind == "thermal":
        s["definition"]["vehicles"]["a"]["thermal"] = {}
    if kind == "processing":
        s["definition"]["products"][0]["processing_sec"] = 1.0
    if kind == "duration":
        t["actions"][0]["commands"][0]["duration_sec"] = 2.0
    if kind == "rf":
        s["definition"]["radio_network"] = {}
    with pytest.raises(ValueError):
        forecast_mission_plan(s, t, view_for(case()))


def benchmark_spec():
    return {
        "id": "frozen-forecast",
        "cases": [case(0), case(1), case(2, negative=True)],
        "teacher": {"min_completion_fraction": 1.0},
    }


def fake_rows(spec):
    rows = []
    for c in spec["cases"]:
        for p in ("forecast_delivery", "priority", "earliest_deadline"):
            count = 0 if c["negative"] else 2 if p != "priority" else 1
            rows.append(
                {
                    "case": c["id"],
                    "policy": p,
                    "error": None,
                    "native_replay_verified": True,
                    "fulfilled": count,
                    "total": 2,
                    "completion_fraction": count / 2,
                    "rejections": 0,
                    "predicted_fulfilled": count,
                    "mission_goal_passed": not c["negative"],
                }
            )
    return rows


def test_teacher_requires_complete_paired_holdouts_and_honest_margin():
    spec = benchmark_spec()
    rows = fake_rows(spec)
    assert forecast_teacher_report(spec, rows)["eligible_for_sft_selection"]
    assert not forecast_teacher_report(spec, rows)["expert_teacher_qualified"]
    strict = copy.deepcopy(spec)
    strict["teacher"]["min_margin"] = 0.1
    assert not forecast_teacher_report(strict, rows)["eligible_for_sft_selection"]
    with pytest.raises(ValueError, match="every paired row"):
        forecast_teacher_report(spec, rows[:-1])
    rows[0]["error"] = {"message": "failed execution"}
    assert not forecast_teacher_report(spec, rows)["eligible_for_sft_selection"]


def test_teacher_refuses_duplicate_inputs_and_false_split():
    spec = benchmark_spec()
    spec["cases"][1] = copy.deepcopy(spec["cases"][0])
    spec["cases"][1]["id"] = "renamed"
    with pytest.raises(ValueError):
        ForecastBenchmarkSpec.model_validate(spec)
    spec = benchmark_spec()
    spec["cases"][0]["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        ForecastBenchmarkSpec.model_validate(spec)


def test_native_benchmark_and_offline_audit(tmp_path):
    result = run_forecast_benchmark(benchmark_spec(), tmp_path / "benchmark")
    assert result["execution_integrity_passed"], result["rows"]
    assert result["teacher"]["eligible_for_sft_selection"], result["teacher"]
    assert verify_forecast_benchmark(tmp_path / "benchmark")["integrity_passed"]
    assert verify_forecast_benchmark(tmp_path / "benchmark", native=True)[
        "native_execution_replayed"
    ]
    (tmp_path / "benchmark/extra").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory"):
        verify_forecast_benchmark(tmp_path / "benchmark")


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "negative", "optimistic", "replay", "holdouts"]
)
def test_teacher_denies_incomplete_or_unrealized_evidence(failure):
    spec = benchmark_spec()
    rows = fake_rows(spec)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    elif failure == "negative":
        rows[-3]["mission_goal_passed"] = True
    elif failure == "optimistic":
        rows[0]["predicted_fulfilled"] = 3
    elif failure == "replay":
        rows[1]["native_replay_verified"] = False
    else:
        spec["teacher"]["min_cases_per_family"] = 3
    if failure in {"missing", "duplicate"}:
        with pytest.raises(ValueError, match="every paired row"):
            forecast_teacher_report(spec, rows)
    else:
        assert not forecast_teacher_report(spec, rows)["eligible_for_sft_selection"]


def test_forecast_cli_proposal_is_not_native_certificate(tmp_path, capsys):
    from space_env.cli import main

    c = case()
    for name in ("scenario", "task"):
        (tmp_path / f"{name}.json").write_text(canonical_json(c[name]))
    args = [
        "forecast-delivery-plan",
        str(tmp_path / "scenario.json"),
        str(tmp_path / "task.json"),
        str(tmp_path / "proposal"),
    ]
    assert main(args) is None
    result = json.loads(capsys.readouterr().out)
    assert result["predicted_fulfilled"] == 2
    assert result["native_feasibility_validated"] is False
    assert not (tmp_path / "proposal/certificate.json").exists()
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_audit_rejects_resealed_false_schedule(tmp_path):
    from space_env.simulation.snapshot import digest
    from space_env.simulation.streaming import file_digest

    root = tmp_path / "benchmark"
    run_forecast_benchmark(benchmark_spec(), root)
    path = root / "000/forecast_delivery/schedule.json"
    schedule = json.loads(path.read_text())
    schedule["predicted_fulfilled"] = 0
    schedule["sha256"] = digest({k: v for k, v in schedule.items() if k != "sha256"})
    path.write_text(canonical_json(schedule))
    result = json.loads((root / "result.json").read_text())
    result["files_sha256"][path.relative_to(root).as_posix()] = file_digest(path)
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    (root / "result.json").write_text(canonical_json(result))
    with pytest.raises(ValueError, match="schedule recomputation"):
        verify_forecast_benchmark(root)
