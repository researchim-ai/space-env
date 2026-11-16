"""Product-specific graph progress and native multi-hop custody continuity."""

import pytest

from examples.modeling.relay_campaign_study import run
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import CampaignPolicySpec, campaign_policy
from space_env.industrial.route_progress import route_progress
from space_env.industrial.session import restore
from space_env.tasks.multihop_campaign import multihop_campaign
from tests.test_campaign_policy import observation, row


def edge(name, source, destination, product="p", kind="relay"):
    return {
        "id": name,
        "commands": [
            {
                "kind": kind,
                "actor_id": source,
                "destination_id": destination,
                "product_id": product,
            }
        ],
    }


def test_progress_is_product_specific_and_rejects_cycles_and_dead_ends():
    graph = [
        edge("ar", "a", "r"),
        edge("rs", "r", "s"),
        edge("sr", "s", "r"),
        edge("sx", "s", "x"),
        edge("xx", "x", "x"),
        edge("ground", "s", "Ground", kind="downlink"),
        edge("other", "r", "s", product="other"),
    ]
    result = route_progress(graph)
    assert result["ar"] == {
        "source_hops_to_ground_sender": 2,
        "destination_hops_to_ground_sender": 1,
        "makes_progress": True,
    }
    assert result["rs"]["makes_progress"]
    assert not result["sr"]["makes_progress"]
    assert not result["sx"]["makes_progress"] and not result["xx"]["makes_progress"]
    assert result["other"]["source_hops_to_ground_sender"] is None
    assert route_progress(list(reversed(graph))) == result


def test_progress_gate_is_explicit_and_fails_closed():
    with pytest.raises(ValueError, match="adaptive routing"):
        CampaignPolicySpec(minimum_completed_requests=1, require_route_progress=True)
    view = observation()
    view["native"]["catalogue"] = [row("back", "relay")]
    view["relay_routes"] = [{"action_id": "back", "available": True}]
    contract = {
        "minimum_completed_requests": 1,
        "adaptive_routing": True,
        "require_route_progress": True,
    }
    assert campaign_policy(view, contract)["choices"]["@central"] == ["wait"]
    view["relay_routes"][0]["makes_progress"] = True
    assert campaign_policy(view, contract)["choices"]["@central"] == ["back"]


@pytest.mark.parametrize(
    "options,success",
    [
        ({}, True),
        ({"fault_profile": "after_custody_recover"}, True),
        ({"forward_link_available": False}, False),
        ({"sink_memory_bits": 250000}, False),
        ({"route_delay_sec": 1}, True),
        ({"bundle_ttl_sec": 1}, False),
        ({"fault_profile": "after_custody_permanent"}, False),
    ],
)
def test_native_two_hop_delivery_expiry_and_resource_conservation(options, success):
    case = multihop_campaign({"horizon_sec": 240, "release_sec": 60, **options})
    record, native, ack = run(case)
    assert record["metrics"]["success"] is success
    assert (ack is not None) is success
    assert not record["call_budget_exhausted"]
    network = record["metrics"]["network_accounting"]
    assert network["maximum_balance_error_bits"] < 1e-6
    assert network["wire_balance_error_bits"] < 1e-6
    history = [
        r for r in network["custody_history"] if r["product_id"] == "late-final-a"
    ]
    assert not any(r["source"] == "s" and r["destination"] == "r" for r in history)
    if success:
        assert [(r["source"], r["destination"]) for r in history] == [
            ("a", "r"),
            ("r", "s"),
        ]
        assert history[0]["time_sec"] < history[1]["time_sec"]
        transfers = [
            t
            for t in native["dtn"]["transfers"].values()
            if t["product_id"] == "late-final-a"
        ]
        assert len({t["expires_sec"] for t in transfers}) == 1
        assert network["stations"]["Late-Ground"]["transmitted_bits"] == 1000000


def test_checkpoint_between_hops_and_physical_ground_band_constraint():
    case = multihop_campaign({"horizon_sec": 240, "release_sec": 60})
    session = AgentCampaignSession(case["configuration"])
    mission = session._native._session._runtime._backend.mission
    assert mission.radio_network.budget(mission, "r", "Late-Ground")["rate_bps"] == 0
    assert (
        mission.dtn.budget(mission, {"source": "s", "destination": "r"})["rate_bps"] > 0
    )
    assert not any(
        e["product_id"].startswith("late-final")
        for e in session.observe()["relay_topology"]
    )
    for _ in range(100):
        history = session.metrics()["network_accounting"]["custody_history"]
        if any(
            r["product_id"] == "late-final-a" and r["destination"] == "r"
            for r in history
        ):
            break
        session.step(
            campaign_policy(session.observe(), case["public_contract"]),
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    else:
        pytest.fail("first custody transition never occurred")
    checkpoint = session.checkpoint()
    current_mission = session._native._session._runtime._backend.mission
    assert current_mission.products["late-final-a"].actor == "r"
    other = restore(checkpoint)
    for _ in range(4):
        action = campaign_policy(session.observe(), case["public_contract"])
        key = str(session.revision)
        assert session.step(
            action, expected_revision=session.revision, idempotency_key=key
        ) == other.step(action, expected_revision=other.revision, idempotency_key=key)
    assert session.checkpoint() == other.checkpoint()
