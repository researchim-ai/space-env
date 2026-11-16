"""Public route selection and native custody under causal radio failures."""

import json

import pytest

from examples.modeling.relay_campaign_study import run
from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.tasks.routing_campaign import routing_campaign
from tests.test_campaign_policy import observation, row


def test_routing_opt_in_fails_closed_and_ranks_current_capacity():
    view = observation()
    view["native"]["catalogue"] = [row("a", "relay"), row("b", "relay")]
    contract = {"minimum_completed_requests": 1, "adaptive_routing": True}
    assert campaign_policy(view, contract)["choices"]["@central"] == ["wait"]
    view["relay_routes"] = [
        {
            "action_id": "a",
            "available": False,
            "rate_bps": 100,
            "receiver_free_bits": 100,
        },
        {"action_id": "b", "available": True, "rate_bps": 1, "receiver_free_bits": 1},
    ]
    assert campaign_policy(view, contract)["choices"]["@central"] == ["b"]
    view["relay_routes"][0]["available"] = True
    assert campaign_policy(view, contract)["choices"]["@central"] == ["a"]
    view["relay_routes"][1].update(rate_bps=100, receiver_free_bits=200)
    assert campaign_policy(view, contract)["choices"]["@central"] == ["b"]
    contract["adaptive_routing"] = False
    assert campaign_policy(view, contract)["choices"]["@central"] == ["a"]


def test_receiver_memory_prevents_infeasible_whole_product_route():
    case = routing_campaign({"horizon_sec": 240, "release_sec": 60})
    case["configuration"]["scenario"]["definition"]["operations"]["vehicles"]["r"][
        "memory_bits"
    ] = 250000
    record, _, _ = run(case)
    assert record["metrics"]["success"]
    custody = record["metrics"]["network_accounting"]["custody_history"]
    assert {r["destination"] for r in custody} == {"s"}


@pytest.mark.parametrize(
    "profile,success,destination",
    [
        ("none", True, "r"),
        ("before_transfer", True, "s"),
        ("mid_transfer", True, "s"),
        ("after_custody_recover", True, "r"),
        ("after_custody_permanent", False, "r"),
        ("all_relays_down", False, None),
    ],
)
def test_native_fault_routes_and_nonrecoverable_custody(profile, success, destination):
    case = routing_campaign(
        {"horizon_sec": 240, "release_sec": 60, "fault_profile": profile}
    )
    session = AgentCampaignSession(case["configuration"])
    initial = session.observe()
    assert initial["relay_routes"] == []
    assert "late-final-a" not in json.dumps(initial)
    assert "radio-recovery-r" not in json.dumps(initial)
    record, native, ack = run(case)
    assert record["metrics"]["success"] is success
    assert (ack is not None) is success
    assert not record["call_budget_exhausted"]
    assert all(
        e["response"].get("feedback", {}).get("accepted") for e in record["events"]
    )
    network = record["metrics"]["network_accounting"]
    assert network["maximum_balance_error_bits"] < 1e-6
    assert network["wire_balance_error_bits"] < 1e-6
    custody = [
        r for r in network["custody_history"] if r["product_id"] == "late-final-a"
    ]
    assert [r["destination"] for r in custody] == (
        [] if destination is None else [destination]
    )
    station = network["stations"].get("Late-Ground", {})
    assert station.get("terminal_excess", 0) == 0
    assert station.get("rate_excess_bps", 0) == 0
    if profile == "mid_transfer":
        transfers = [
            t
            for t in native["dtn"]["transfers"].values()
            if t["product_id"] == "late-final-a"
        ]
        assert {t["status"] for t in transfers} == {"cancelled", "completed"}
        assert len({t["expires_sec"] for t in transfers}) == 1
        assert native["dtn"]["wasted_partial_bits"]["a"] > 0
        assert verify_episode(record)["success"]
    if profile == "after_custody_permanent":
        product = next(p for p in native["products"] if p["id"] == "late-final-a")
        assert product["completed_sec"] is None
        assert custody[0]["destination"] == "r"
