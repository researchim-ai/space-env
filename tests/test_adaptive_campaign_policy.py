"""Adaptive waits use declared current timers, not private event lookahead."""

import copy

import pytest

from examples.modeling.reactive_campaign_study import cases, run_reactive
from space_env.industrial.campaign_policy import CampaignPolicySpec, campaign_policy
from tests.test_campaign_policy import observation, row

CONTRACT = {
    "minimum_completed_requests": 3,
    "completion_action_ids": ["finish"],
    "adaptive_wait": True,
}


def test_current_boundaries_and_clipping():
    view = observation()
    view["native"]["time_sec"] = 5.0
    assert campaign_policy(view, CONTRACT)["advance_sec"] == 1.0
    view["native"]["entities"] = [
        {"index": 0, "id": "c", "kind": "command", "values": {"deadline_sec": 5.4}},
        {"index": 1, "id": "p", "kind": "product", "values": {"ready_sec": 5.7}},
        {"index": 2, "id": "r", "kind": "request", "values": {"deadline_sec": 5.3}},
    ]
    assert campaign_policy(view, CONTRACT)["advance_sec"] == pytest.approx(0.3)
    view["native"]["entities"][2]["values"]["deadline_sec"] = 4.0
    assert campaign_policy(view, CONTRACT)["advance_sec"] == pytest.approx(0.4)
    view["control"]["max_decision_sec"] = 0.1
    assert campaign_policy(view, CONTRACT)["advance_sec"] == 0.1


def test_new_commands_keep_short_interval_and_default_is_unchanged():
    view = observation()
    view["native"]["time_sec"] = 0.0
    view["native"]["catalogue"] = [row("acquire", "imaging")]
    before = copy.deepcopy(view)
    action = campaign_policy(view, CONTRACT)
    assert action["choices"]["@central"] == ["acquire"]
    assert action["advance_sec"] == 0.25
    assert view == before
    view["native"]["catalogue"] = []
    assert (
        campaign_policy(view, {**CONTRACT, "adaptive_wait": False})["advance_sec"]
        == 0.25
    )
    view["future_events"] = [{"at_sec": 0.001, "type": "fault"}]
    assert campaign_policy(view, CONTRACT)["advance_sec"] == 1.0


def test_adaptive_configuration_validation():
    for value in [
        {"adaptive_wait": "true"},
        {"adaptive_wait": True, "max_wait_sec": 0.1},
        {"max_wait_sec": float("inf")},
    ]:
        with pytest.raises(ValueError):
            CampaignPolicySpec.model_validate(
                {"minimum_completed_requests": 1, **value}
            )


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
def test_native_adaptive_outcomes_and_call_bound(name):
    case = next(c for c in cases() if c["id"] == name)
    record = run_reactive(
        case["configuration"],
        public_contract={**CONTRACT, "completion_action_ids": ["control"]},
    )
    assert record["metrics"]["success"] == case["expected_reactive_success"]
    assert not record["call_budget_exhausted"]
    assert len(record["events"]) < 50
    assert all(
        e["response"].get("feedback", {}).get("accepted") for e in record["events"]
    )
