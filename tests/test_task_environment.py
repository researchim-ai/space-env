"""Exogenous events cannot be undone by rejected controls or leak before arrival."""

import copy
import json

import pytest

from examples.modeling.campaign import recipe as campaign_recipe
from examples.modeling.dynamic_agent_campaign import recipe, run
from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.session import restore
from space_env.tasks import TaskSession
from space_env.tasks.schema import TaskSpec


def decision(session, actions=("wait",), duration=3.0, key=None):
    return session.step(
        {"choices": {"@central": list(actions)}, "advance_sec": duration},
        expected_revision=session.revision,
        idempotency_key=key or str(session.revision),
    )


def test_arrival_interrupt_is_causal_and_guess_is_not_a_tool():
    config, _ = recipe()
    session = AgentCampaignSession(config)
    initial = session.observe()
    assert initial["environment_events"] == []
    assert all(
        row.get("request_ref") != "customer-a" for row in initial["native"]["catalogue"]
    )
    assert "customer-a" not in json.dumps(initial)
    cp = session.checkpoint()
    with pytest.raises(ValueError, match="public catalogue"):
        decision(session, ["cpu-input"])
    assert session.checkpoint() == cp
    result = decision(session)
    assert result["elapsed_sec"] == 2.0
    view = session.observe()
    assert view["environment_events"][0]["id"] == "new-customer"
    assert "urgent-customer" not in json.dumps(view)
    assert any(
        row.get("request_ref") == "customer-a" for row in view["native"]["catalogue"]
    )
    restored = restore(session.checkpoint())
    assert decision(session) == decision(restored)
    assert session.observe()["native"]["time_sec"] == 2.5
    assert all(e["accepted"] for e in session.observe()["environment_events"])


def test_rejected_command_does_not_suppress_event():
    config, _ = recipe()
    session = AgentCampaignSession(config)
    result = decision(session, ["archive-image", "archive-image"])
    assert result["feedback"]["accepted"] is False
    assert result["elapsed_sec"] == 2.0
    assert session.observe()["environment_events"][0]["accepted"]
    assert session._native._session._runtime.request_authority("customer-a") is not None
    cp = session.checkpoint()
    assert restore(cp).checkpoint() == cp
    retry = session.step(
        {
            "choices": {"@central": ["archive-image", "archive-image"]},
            "advance_sec": 3.0,
        },
        expected_revision=0,
        idempotency_key="0",
    )
    assert retry == result
    assert session.checkpoint() == cp


def test_same_time_order_rejection_once_and_zero_time_events():
    case, campaign = campaign_recipe("eo")
    case["task"]["environment_events"] = [
        {
            "id": "fault",
            "at_sec": 0.0,
            "type": "spacecraft_fault",
            "payload": {"actor": "a", "subsystem": "cpu"},
        },
        {
            "id": "clear",
            "at_sec": 1.0,
            "type": "spacecraft_fault_cleared",
            "payload": {"actor": "a", "subsystem": "cpu"},
        },
        {
            "id": "duplicate-clear",
            "at_sec": 1.0,
            "type": "spacecraft_fault_cleared",
            "payload": {"actor": "a", "subsystem": "cpu"},
        },
    ]
    session = AgentCampaignSession(
        {"scenario": case["scenario"], "task": case["task"], "campaign": campaign}
    )
    assert session.observe()["environment_events"][0]["time_sec"] == 0.0
    result = decision(session)
    assert result["elapsed_sec"] == 1.0
    outcomes = session.observe()["environment_events"]
    assert [e["accepted"] for e in outcomes] == [True, True, False]
    decision(session, duration=0.5)
    assert session.observe()["environment_events"] == outcomes
    assert restore(session.checkpoint()).checkpoint() == session.checkpoint()


@pytest.mark.parametrize("profile", ["arrival", "recovery"])
def test_full_dynamic_mission_and_negative_baseline(profile):
    config, schedule = recipe(profile)
    result = run(config, schedule, "reference")
    assert result["metrics"]["success"]
    assert not result["metrics"]["resource_reset_between_phases"]
    assert result["metrics"]["environment_events_rejected"] == 0
    assert all(result["metrics"]["current_bus_health"].values())
    assert verify_episode(result)["success"]
    failure = run(config, schedule, "wait")
    assert failure["metrics"]["failed"] and not failure["metrics"]["success"]


def test_schedule_validation_and_legacy_omission():
    config, _ = recipe()
    task = copy.deepcopy(config["task"])
    bad_cases = []
    bad_cases.append([task["environment_events"][1], task["environment_events"][0]])
    bad_cases.append([task["environment_events"][0], task["environment_events"][0]])
    bad_cases.append(
        [
            {
                "id": "bad",
                "at_sec": float("nan"),
                "type": "spacecraft_recover",
                "payload": {"actor": "a"},
            }
        ]
    )
    bad_cases.append(
        [
            {
                "id": "bad",
                "at_sec": 1.0,
                "type": "spacecraft_fault",
                "payload": {"actor": "a", "subsystem": "imaginary"},
            }
        ]
    )
    bad_cases.append(
        [
            {
                "id": "bad",
                "at_sec": 1.0,
                "type": "spacecraft_recover",
                "payload": {"actor": "a", "extra": True},
            }
        ]
    )
    for events in bad_cases:
        with pytest.raises(ValueError):
            TaskSpec.model_validate({**task, "environment_events": events})
    task.pop("environment_events")
    assert "environment_events" not in TaskSpec.model_validate(task).model_dump(
        mode="json"
    )
    for event in [
        {
            "id": "late",
            "at_sec": 30.0,
            "type": "spacecraft_recover",
            "payload": {"actor": "a"},
        },
        {
            "id": "unknown",
            "at_sec": 1.0,
            "type": "spacecraft_recover",
            "payload": {"actor": "absent"},
        },
        {
            "id": "unknown",
            "at_sec": 1.0,
            "type": "station_recovered",
            "payload": {"station": "absent"},
        },
    ]:
        with pytest.raises(ValueError):
            AgentCampaignSession(
                {**config, "task": {**config["task"], "environment_events": [event]}}
            )


def test_native_task_replays_both_checkpoint_formats():
    config, _ = recipe()
    session = TaskSession(config["scenario"], config["task"])
    response = session.step(
        {"@central": ["wait"]},
        expected_revision=0,
        idempotency_key="one",
        advance_sec=3.0,
    )
    assert response["elapsed_sec"] == 2.0
    assert response["environment_events"][0]["accepted"]
    for compact in [False, True]:
        cp = session.checkpoint(compact=compact)
        assert TaskSession.from_checkpoint(cp).checkpoint(compact=compact) == cp


def test_partial_cpu_work_and_inventory_survive_fault_and_resume():
    config, _ = recipe("recovery")
    session = AgentCampaignSession(config)
    decision(session, ["archive-image"], 1.0)
    decision(session, duration=1.0)
    decision(session, ["cpu-input"], 2.0)  # priority interrupts at 2.5
    decision(session, duration=1.5)
    decision(session, ["new-image", "cpu", "archive-radio"], 2.0)
    native = session._native._session._runtime
    report = native._backend.report()
    job = copy.deepcopy(report["lifecycle"]["jobs"]["features-a"])
    assert native.time_sec == 4.75
    assert 0 < job["work_sec"] < 1.5
    assert job["remaining_sec"] > 0
    before_energy = report["energy_ledger"]["a"]["consumed_wh"]
    restored = restore(session.checkpoint())
    assert decision(session, duration=2.0) == decision(restored, duration=2.0)
    after = session._native._session._runtime._backend.report()
    assert after["lifecycle"]["jobs"]["features-a"]["work_sec"] == job["work_sec"]
    assert (
        after["lifecycle"]["jobs"]["features-a"]["remaining_sec"]
        == job["remaining_sec"]
    )
    assert after["energy_ledger"]["a"]["consumed_wh"] > before_energy
    assert "raw-a" in after["lifecycle"]["artifacts"]
    decision(session, duration=0.5)  # station recovery at 6
    decision(session, ["cpu", "archive-radio"], 2.0)
    final = session._native._session._runtime._backend.report()
    assert final["lifecycle"]["jobs"]["features-a"]["status"] == "completed"
    assert final["lifecycle"]["jobs"]["features-a"]["work_sec"] == pytest.approx(1.5)
    assert (
        max(abs(v["balance_error_wh"]) for v in final["energy_ledger"].values()) < 1e-10
    )


def test_raw_only_arrival_and_cancel_preserves_other_deliveries():
    case, _campaign = campaign_recipe("eo")
    lifecycle = case["scenario"]["definition"]["lifecycle"]
    request = lifecycle["requests"].pop(2)
    request["release_sec"] = 1.0
    # Direct raw products legitimately have no processing jobs.
    case["task"]["environment_events"] = [
        {
            "id": "raw-arrival",
            "at_sec": 1.0,
            "type": "request_added",
            "payload": {"request": request, "jobs": []},
        },
        {
            "id": "cancel-raw",
            "at_sec": 3.0,
            "type": "request_cancelled",
            "payload": {"request_id": request["id"]},
        },
    ]
    session = TaskSession(case["scenario"], case["task"])
    session.step(
        {"@central": ["archive-image"]},
        expected_revision=0,
        idempotency_key="image",
        advance_sec=2.0,
    )
    session.step(
        {"@central": ["wait"]},
        expected_revision=1,
        idempotency_key="finish",
        advance_sec=1.0,
    )
    before = copy.deepcopy(
        session._runtime._backend.report()["lifecycle"]["artifacts"]["archive"]
    )
    result = session.step(
        {"@central": ["wait"]},
        expected_revision=2,
        idempotency_key="cancel",
        advance_sec=5.0,
    )
    assert result["elapsed_sec"] == 1.0
    report = session._runtime._backend.report()
    assert report["lifecycle"]["requests"]["new-request"]["status"] == "cancelled"
    assert report["lifecycle"]["artifacts"]["archive"] == before
    assert TaskSession.from_checkpoint(session.checkpoint(compact=True)).checkpoint(
        compact=True
    ) == session.checkpoint(compact=True)
