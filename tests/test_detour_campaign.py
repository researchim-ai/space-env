"""Longer RF detours with current public custody history and immutable TTL."""

import copy
import json

import pytest

from examples.modeling.relay_campaign_study import run
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.detour import detour_policy
from space_env.industrial.session import restore
from space_env.tasks.detour_campaign import DetourCampaignSpec, detour_campaign
from tests.test_campaign_policy import observation, row


def policy_view():
    view = observation()
    view["native"]["time_sec"] = 10
    view["native"]["catalogue"] = [
        row("direct", "relay"),
        row("detour", "relay"),
        row("return", "relay"),
    ]
    view["relay_topology"] = [{"kind": "downlink", "actor_id": "s", "product_id": "p"}]
    view["relay_routes"] = [
        {"action_id": name, "available": True}
        for name in ["direct", "detour", "return"]
    ]
    view["detour_graph"] = {
        "measured_at": 10,
        "custody_paths": {"p": ["a", "r"]},
        "links": [
            {
                "action_id": name,
                "source": source,
                "destination": destination,
                "product_id": "p",
                "rf_available": ready,
            }
            for name, source, destination, ready in [
                ("direct", "r", "s", False),
                ("detour", "r", "t", True),
                ("finish", "t", "s", True),
                ("return", "r", "a", True),
            ]
        ],
    }
    return view


CONTRACT = {"minimum_completed_requests": 1, "adaptive_routing": True}


def test_future_failure_does_not_change_initial_public_graph():
    common = {"horizon_sec": 240, "release_sec": 60}
    first = AgentCampaignSession(detour_campaign(common)["configuration"])
    second = AgentCampaignSession(
        detour_campaign({**common, "fault_profile": "after_custody_permanent"})[
            "configuration"
        ]
    )
    assert first.observe() == second.observe()
    checkpoint = first.checkpoint()
    assert first.observe() == first.observe()
    assert first.checkpoint() == checkpoint


def test_longer_path_shortest_available_and_no_revisit_with_stale_route_rows():
    view = policy_view()
    before = copy.deepcopy(view)
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["detour"]
    assert view == before
    view["detour_graph"]["links"][0]["rf_available"] = True
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["direct"]
    view["detour_graph"]["custody_paths"]["p"] = ["a", "s", "r"]
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["wait"]


@pytest.mark.parametrize("missing", ["detour_graph", "relay_topology", "relay_routes"])
def test_missing_information_fails_closed(missing):
    view = policy_view()
    del view[missing]
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["wait"]


def test_stale_graph_and_invalid_contract_rejected():
    view = policy_view()
    view["detour_graph"]["measured_at"] = 9
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["wait"]
    with pytest.raises(ValueError, match="static progress"):
        detour_policy(view, {**CONTRACT, "require_route_progress": True})
    with pytest.raises(ValueError):
        DetourCampaignSpec(backup_link_available="no")
    with pytest.raises(ValueError):
        detour_campaign([])
    with pytest.raises(ValueError, match="initially unavailable"):
        DetourCampaignSpec(forward_link_available=True, forward_recovery_delay_sec=10)


def test_receiving_actor_does_not_start_conflicting_ground_transmission():
    view = policy_view()
    downlink = row("ground", "downlink")
    downlink["commands"][0]["actor_id"] = "s"
    view["native"]["catalogue"].append(downlink)
    view["detour_graph"]["receivers"] = {"s": {"receiving": True}}
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["detour"]
    view["detour_graph"]["receivers"]["s"]["receiving"] = False
    assert detour_policy(view, CONTRACT)["choices"]["@central"] == ["ground"]


@pytest.mark.parametrize(
    "options,success,route",
    [
        ({}, True, [("a", "r"), ("r", "t"), ("t", "s")]),
        ({"forward_link_available": True}, True, [("a", "r"), ("r", "s")]),
        ({"backup_link_available": False}, False, []),
        (
            {"fault_profile": "after_custody_recover"},
            True,
            [("a", "r"), ("r", "t"), ("t", "s")],
        ),
        ({"fault_profile": "after_custody_permanent"}, False, [("a", "r")]),
        ({"bundle_ttl_sec": 1}, False, [("a", "r")]),
        (
            {"forward_recovery_delay_sec": 10, "backup_link_available": False},
            True,
            [("a", "r"), ("r", "s")],
        ),
    ],
)
def test_native_detours_custody_and_ttl(options, success, route):
    case = detour_campaign({"horizon_sec": 240, "release_sec": 60, **options})
    record, native, ack = run(case, policy=detour_policy)
    assert record["metrics"]["success"] is success
    assert (ack is not None) is success
    assert not record["call_budget_exhausted"]
    assert all(
        e["response"].get("feedback", {}).get("accepted") for e in record["events"]
    )
    network = record["metrics"]["network_accounting"]
    assert network["maximum_balance_error_bits"] < 1e-6
    assert network["wire_balance_error_bits"] < 1e-6
    path = [h for h in network["custody_history"] if h["product_id"] == "late-final-a"]
    assert [(h["source"], h["destination"]) for h in path] == route
    holders = ([path[0]["source"]] + [h["destination"] for h in path]) if path else []
    assert len(holders) == len(set(holders))
    transfers = [
        t
        for t in native["dtn"]["transfers"].values()
        if t["product_id"] == "late-final-a"
    ]
    assert len({t["expires_sec"] for t in transfers}) <= 1


def test_unreleased_graph_privacy_and_checkpoint_after_intermediate_custody():
    case = detour_campaign({"horizon_sec": 240, "release_sec": 60})
    session = AgentCampaignSession(case["configuration"])
    assert "late-final" not in json.dumps(session.observe())
    for _ in range(100):
        paths = session.observe()["detour_graph"]["custody_paths"]
        if paths.get("late-final-a") == ["a", "r", "t"]:
            break
        session.step(
            detour_policy(session.observe(), case["public_contract"]),
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    else:
        pytest.fail("backup relay never took custody")
    cp = session.checkpoint()
    other = restore(cp)
    for _ in range(4):
        action = detour_policy(session.observe(), case["public_contract"])
        key = str(session.revision)
        assert session.step(
            action, expected_revision=session.revision, idempotency_key=key
        ) == other.step(action, expected_revision=other.revision, idempotency_key=key)
    assert other.checkpoint() == session.checkpoint()
