"""Two active spacecraft over one continuous coupled campaign and clock."""

import copy
import json

import pytest

from examples.modeling.reactive_campaign_study import run_reactive
from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.industrial.session import restore
from space_env.tasks.long_campaign import LongCampaignSpec, long_campaign


@pytest.fixture(scope="module")
def case():
    return long_campaign({"horizon_sec": 240.0, "release_sec": 60.0})


def test_two_wave_native_success_and_public_provenance(case):
    record = run_reactive(
        case["configuration"], public_contract=case["public_contract"], max_calls=512
    )
    assert record["metrics"]["success"] and record["metrics"]["time_sec"] == 240.0
    assert len(record["metrics"]["completed_phases"]) == 3
    assert record["metrics"]["environment_events_rejected"] == 0
    assert all(record["metrics"]["current_bus_health"].values())
    assert verify_episode(record)["success"]
    actors = {
        c["actor_id"]
        for e in record["events"]
        for row in e["response"]["observation"]["native"]["catalogue"]
        for c in row.get("commands", [])
        if c["kind"] == "imaging"
    }
    assert actors == {"a", "b"}
    assert len(record["events"]) < 60
    assert (
        max(
            abs(v["balance_error_wh"])
            for v in record["metrics"]["resource_accounting"]["energy"].values()
        )
        < 1e-8
    )


def test_midcoast_replay_and_late_request_privacy(case):
    session = AgentCampaignSession(case["configuration"])
    while session.observe()["native"]["time_sec"] < 20.0:
        action = campaign_policy(session.observe(), case["public_contract"])
        session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    view = session.observe()
    assert "late-b" not in json.dumps(view)
    cp = session.checkpoint()
    restored = restore(cp)
    action = campaign_policy(view, case["public_contract"])
    assert action["advance_sec"] > 30.0
    response = session.step(
        action, expected_revision=session.revision, idempotency_key="late"
    )
    assert response == restored.step(
        action, expected_revision=restored.revision, idempotency_key="late"
    )
    assert session.observe()["native"]["time_sec"] == 60.0
    assert session.observe()["environment_events"][-1]["accepted"]
    assert "late-b" in json.dumps(session.observe())
    assert session.metrics()["resource_accounting"]["energy"]["a"]["consumed_wh"] > 0.1


def test_low_battery_does_not_reset_between_phases():
    case = long_campaign({"horizon_sec": 600.0, "release_sec": 60.0, "battery_wh": 1.0})
    record = run_reactive(
        case["configuration"], public_contract=case["public_contract"], max_calls=512
    )
    assert record["metrics"]["failed"] and not record["metrics"]["success"]
    assert record["metrics"]["completed_phases"][:2] == [
        "initial-delivery",
        "late-delivery",
    ]
    assert not record["metrics"]["resource_reset_between_phases"]
    assert not all(record["metrics"]["current_bus_health"].values())


def test_convergence_profile_does_not_change_target():
    balanced = long_campaign({"horizon_sec": 240.0, "release_sec": 60.0})
    fine = long_campaign(
        {
            "horizon_sec": 240.0,
            "release_sec": 60.0,
            "sample_sec": 1.0,
            "max_step_sec": 5.0,
            "execution_step_sec": 0.5,
        }
    )
    left = copy.deepcopy(balanced["configuration"])
    right = fine["configuration"]
    for orbit in left["scenario"]["definition"]["orbits"].values():
        orbit["integration"] = {"sample_sec": 1.0, "max_step_sec": 5.0}
    left["scenario"]["definition"]["lifecycle"]["execution_step_sec"] = 0.5
    assert left == right


def test_bounds_and_explicit_short_release():
    for value in [
        {"horizon_sec": 100.0},
        {"horizon_sec": 300000.0},
        {"release_sec": 120.0, "horizon_sec": 120.0},
        {"battery_wh": float("nan")},
        {"sample_sec": 61.0},
    ]:
        with pytest.raises(ValueError):
            LongCampaignSpec.model_validate(value)
    with pytest.raises(ValueError, match="earlier release"):
        long_campaign({"horizon_sec": 240.0})
