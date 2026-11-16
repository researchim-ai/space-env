"""Reactive control uses public state, including negative outcomes and restarts."""

import copy

import pytest

from examples.modeling.reactive_campaign_study import cases, run_reactive
from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import CampaignPolicySpec, campaign_policy
from space_env.industrial.session import restore
from space_env.tasks.observation import STATES

CONTRACT = {"minimum_completed_requests": 3, "completion_action_ids": ["finish"]}


def observation():
    return {
        "schema": "space-env.agent-campaign-observation/1",
        "revision": 0,
        "native": {"entities": [], "relations": [], "catalogue": []},
        "control": {"max_decision_sec": 30.0},
        "environment_events": [],
    }


def row(name, kind, request=None, enabled=True):
    return {
        "id": name,
        "kind": "submit",
        "enabled": enabled,
        "command_kinds": [kind],
        "commands": [{"kind": kind, "request_id": request}],
    }


def test_stage_priority_deadline_and_opaque_names():
    view = observation()
    view["native"]["entities"] = [
        {
            "index": 0,
            "kind": "request",
            "id": "low",
            "values": {
                "priority": 1,
                "deadline_sec": 5.0,
                "status_code": STATES["released"],
            },
        },
        {
            "index": 1,
            "kind": "request",
            "id": "high",
            "values": {
                "priority": 9,
                "deadline_sec": 9.0,
                "status_code": STATES["released"],
            },
        },
    ]
    rows = [row("aaa", "imaging", "low"), row("zzz", "imaging", "high")]
    view["native"]["catalogue"] = rows
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["zzz"]
    view["native"]["entities"][0]["values"]["priority"] = 9
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["aaa"]
    rows.append(row("wire", "downlink"))
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["wire"]
    rows[-1]["enabled"] = False
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["aaa"]


def test_public_completion_contract_not_hidden_phase_state():
    view = observation()
    view["native"]["catalogue"] = [row("finish", "slew")]
    view["campaign"] = {"success": True, "completed_phases": ["delivery"]}
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["wait"]
    view["native"]["entities"] = [
        {
            "index": i,
            "id": str(i),
            "kind": "request",
            "values": {"status_code": STATES["completed"]},
        }
        for i in range(3)
    ]
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["finish"]
    before = copy.deepcopy(view)
    campaign_policy(view, CONTRACT)
    assert view == before
    view["control"]["max_decision_sec"] = 0.1
    assert campaign_policy(view, CONTRACT)["advance_sec"] == 0.1


def test_relation_based_delivery_priority():
    view = observation()
    view["native"]["entities"] = [
        {"index": 0, "id": "p", "kind": "product", "values": {}},
        {"index": 1, "id": "q", "kind": "request", "values": {"priority": 7}},
    ]
    view["native"]["relations"] = [{"kind": "fulfills", "source": 0, "destination": 1}]
    first, second = row("aaa", "downlink"), row("zzz", "downlink")
    second["commands"][0]["product_id"] = "p"
    view["native"]["catalogue"] = [first, second]
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["zzz"]


def test_configuration_rejects_invalid_contract():
    for spec in [
        {"minimum_completed_requests": -1},
        {"minimum_completed_requests": True},
        {"minimum_completed_requests": 0, "poll_sec": float("nan")},
        {"minimum_completed_requests": 0, "completion_action_ids": ["wait"]},
        {"minimum_completed_requests": 0, "completion_action_ids": ["x", "x"]},
    ]:
        with pytest.raises(ValueError):
            CampaignPolicySpec.model_validate(spec)
    with pytest.raises(ValueError):
        campaign_policy({"schema": "unknown"}, CONTRACT)


@pytest.mark.parametrize(
    "name",
    [
        "arrival",
        "recovery",
        "delayed-arrival",
        "delayed-recovery",
        "mandatory-cancel",
        "permanent-cpu",
    ],
)
def test_native_outcomes_are_not_relabelled(name):
    case = next(c for c in cases() if c["id"] == name)
    record = run_reactive(case["configuration"])
    assert record["metrics"]["success"] == case["expected_reactive_success"]
    assert record["policy_origin"] == "scripted_public"
    assert not record["call_budget_exhausted"]
    assert all(
        e["response"].get("feedback", {}).get("accepted") for e in record["events"]
    )
    if name == "delayed-recovery":
        assert verify_episode(record)["success"]


def test_identical_observations_under_changed_future_and_midfault_replay():
    case = next(c for c in cases() if c["id"] == "recovery")
    config = copy.deepcopy(case["configuration"])
    other = copy.deepcopy(config)
    next(e for e in other["task"]["environment_events"] if e["id"] == "cpu-clear")[
        "at_sec"
    ] = 5.75
    left, right = AgentCampaignSession(config), AgentCampaignSession(other)
    public = {"minimum_completed_requests": 3, "completion_action_ids": ["control"]}
    while left.observe()["native"]["time_sec"] < 4.75:
        assert left.observe() == right.observe()
        action = campaign_policy(left.observe(), public)
        assert action == campaign_policy(right.observe(), public)
        for s in [left, right]:
            s.step(
                action, expected_revision=s.revision, idempotency_key=str(s.revision)
            )
    restored = restore(left.checkpoint())
    action = campaign_policy(left.observe(), public)
    assert action == campaign_policy(restored.observe(), public)
    assert left.step(
        action, expected_revision=left.revision, idempotency_key="next"
    ) == restored.step(
        action, expected_revision=restored.revision, idempotency_key="next"
    )


def test_cancelled_work_and_compound_templates_are_not_dispatched():
    view = observation()
    view["native"]["entities"] = [
        {
            "index": 0,
            "kind": "request",
            "id": "cancelled",
            "values": {"priority": 100, "status_code": STATES["cancelled"]},
        }
    ]
    cancelled = row("cancelled-image", "imaging", "cancelled")
    compound = row("compound", "downlink")
    compound["commands"].append({"kind": "downlink"})
    view["native"]["catalogue"] = [cancelled, compound]
    assert campaign_policy(view, CONTRACT)["choices"]["@central"] == ["wait"]
