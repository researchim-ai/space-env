"""Unknown events still interrupt waits; resource diagnostics stay trainer-only."""

import copy
import json

import pytest

from examples.modeling.reactive_campaign_study import cases
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.industrial.session import restore


def step(session, action):
    return session.step(
        action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def test_unannounced_event_interrupt_and_resource_diagnostics():
    config = copy.deepcopy(cases()[0]["configuration"])
    other = copy.deepcopy(config)
    other["task"]["environment_events"].append(
        {
            "id": "unannounced-b-fault",
            "at_sec": 0.6,
            "type": "spacecraft_fault",
            "payload": {"actor": "b", "subsystem": "cpu"},
        }
    )
    other["task"]["environment_events"].sort(key=lambda e: e["at_sec"])
    left, right = AgentCampaignSession(config), AgentCampaignSession(other)
    contract = {
        "minimum_completed_requests": 3,
        "completion_action_ids": ["control"],
        "adaptive_wait": True,
        "max_wait_sec": 5.0,
    }
    assert left.observe() == right.observe()
    for session in [left, right]:
        step(session, campaign_policy(session.observe(), contract))
    assert left.observe() == right.observe()
    action = campaign_policy(left.observe(), contract)
    assert action == campaign_policy(right.observe(), contract)
    assert action["choices"]["@central"] == ["wait"] and action["advance_sec"] > 0.35
    response = step(right, action)
    assert response["elapsed_sec"] == pytest.approx(0.35)
    assert right.observe()["environment_events"][-1]["id"] == "unannounced-b-fault"
    assert right.observe()["native"]["time_sec"] == 0.6
    restored = restore(right.checkpoint())
    next_action = campaign_policy(right.observe(), contract)
    assert next_action == campaign_policy(restored.observe(), contract)
    assert step(right, next_action) == step(restored, next_action)
    metrics = right.metrics()
    assert "resource_accounting" not in json.dumps(right.observe())
    assert (
        max(
            abs(e["balance_error_wh"])
            for e in metrics["resource_accounting"]["energy"].values()
        )
        < 1e-8
    )
    metrics["resource_accounting"]["energy"]["a"]["consumed_wh"] = -100.0
    assert right.metrics()["resource_accounting"]["energy"]["a"]["consumed_wh"] > 0.0
