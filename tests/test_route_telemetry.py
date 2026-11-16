"""Causal delayed route channel; current native control remains explicit."""

import copy

import pytest

from examples.modeling.relay_campaign_study import run
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.industrial.route_telemetry import RouteTelemetry, RouteTelemetrySpec
from space_env.industrial.session import restore
from space_env.tasks.routing_campaign import RoutingCampaignSpec, routing_campaign
from tests.test_campaign_policy import observation, row


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 3601])
def test_delay_validation(value):
    with pytest.raises(ValueError):
        RouteTelemetrySpec(delay_sec=value)
    with pytest.raises(ValueError):
        RoutingCampaignSpec(route_delay_sec=value)


def test_delayed_snapshot_is_causal_pure_and_detached():
    channel = RouteTelemetry({"delay_sec": 2})
    routes = [{"action_id": "route", "available": True, "measured_at": 0}]
    channel.capture(0, routes)
    routes[0]["available"] = False
    assert channel.observation(1, {"route"})[0] == []
    channel.capture(1, [{"action_id": "route", "available": False, "measured_at": 1}])
    before = copy.deepcopy(channel.samples)
    old, meta = channel.observation(2, {"route"})
    assert old[0]["available"] and old[0]["age_sec"] == 2
    assert old[0]["delivered_at"] == 2 and meta["next_delivery_sec"] == 3
    old[0]["available"] = False
    assert channel.observation(2, {"route"})[0][0]["available"]
    assert channel.samples == before
    assert not channel.observation(3, {"route"})[0][0]["available"]
    assert channel.observation(3, set())[0] == []


def test_queue_retains_latest_delivered_and_in_transit_only():
    channel = RouteTelemetry({"delay_sec": 2})
    for i in range(20):
        channel.capture(i, [])
        assert len(channel.samples) <= 3
    assert [s["measured_at"] for s in channel.samples] == [17, 18, 19]
    channel.capture(19, [{"action_id": "replacement"}])
    assert len(channel.samples) == 3
    with pytest.raises(ValueError, match="monotonic"):
        channel.capture(18, [])


def test_age_filter_and_public_delivery_wait():
    view = observation()
    view["native"].update(time_sec=10, catalogue=[row("hop", "relay", "job")])
    view["native"]["entities"] = [
        {
            "kind": "request",
            "id": "job",
            "index": 0,
            "values": {"status_code": 0, "deadline_sec": 50},
        }
    ]
    view["route_telemetry"] = {"next_delivery_sec": 10.5}
    view["relay_routes"] = [{"action_id": "hop", "available": True, "age_sec": 2}]
    contract = {
        "minimum_completed_requests": 1,
        "adaptive_routing": True,
        "adaptive_wait": True,
        "max_wait_sec": 30,
        "max_route_age_sec": 1,
    }
    action = campaign_policy(view, contract)
    assert action["choices"]["@central"] == ["wait"]
    assert action["advance_sec"] == 0.5
    view["relay_routes"][0]["age_sec"] = 1
    assert campaign_policy(view, contract)["choices"]["@central"] == ["hop"]


def test_pending_channel_survives_checkpoint_without_observe_side_effects():
    case = routing_campaign(
        {"horizon_sec": 240, "release_sec": 60, "route_delay_sec": 1}
    )
    session = AgentCampaignSession(case["configuration"])
    for _ in range(30):
        view = session.observe()
        session.step(
            campaign_policy(view, case["public_contract"]),
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
        if view["native"]["time_sec"] > 64:
            break
    checkpoint = session.checkpoint()
    assert session.observe() == session.observe()
    assert session.checkpoint() == checkpoint
    other = restore(checkpoint)
    assert other.checkpoint() == checkpoint
    for _ in range(3):
        action = campaign_policy(session.observe(), case["public_contract"])
        key = str(session.revision)
        assert session.step(
            action, expected_revision=session.revision, idempotency_key=key
        ) == other.step(action, expected_revision=other.revision, idempotency_key=key)


@pytest.mark.parametrize(
    "options,success",
    [
        ({"route_delay_sec": 1, "fault_profile": "mid_transfer"}, True),
        ({"route_delay_sec": 1, "max_route_age_sec": 0.5}, False),
    ],
)
def test_native_delayed_routes_and_stale_only_negative(options, success):
    case = routing_campaign({"horizon_sec": 240, "release_sec": 60, **options})
    record, _, _ = run(case)
    assert record["metrics"]["success"] is success
    assert not record["call_budget_exhausted"]
    network = record["metrics"]["network_accounting"]
    assert network["maximum_balance_error_bits"] < 1e-6
    assert network["wire_balance_error_bits"] < 1e-6
    if success:
        assert network["stations"]["Late-Ground"]["transmitted_bits"] == 1000000
    else:
        assert network["custody_history"] == []
        assert network["completed_transfers"] == 0
        assert (
            network["stations"].get("Late-Ground", {}).get("transmitted_bits", 0) == 0
        )
