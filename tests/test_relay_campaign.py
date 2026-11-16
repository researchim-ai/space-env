"""Native EO/CPU/custody/RF integration, contention and checkpoint continuity."""

import json

import pytest

from examples.modeling.reactive_campaign_study import run_reactive
from space_env.industrial.agent import verify_episode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.campaign_policy import campaign_policy
from space_env.industrial.session import restore
from space_env.tasks.relay_campaign import RelayCampaignSpec, relay_campaign
from tests.test_campaign_policy import observation, row


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, True),
        ({"ack_loss_probability": 0.5}, True),
        ({"ground_terminals": 2, "ground_rate_bps": 500000.0}, True),
        ({"bundle_ttl_sec": 0.001}, False),
    ],
)
def test_native_routes_capacity_and_bit_conservation(options, expected):
    case = relay_campaign({"horizon_sec": 240.0, "release_sec": 60.0, **options})
    record = run_reactive(
        case["configuration"],
        public_contract=case["public_contract"],
        task_text=case["public_brief"],
        max_calls=512,
    )
    assert record["metrics"]["success"] == expected
    assert not record["call_budget_exhausted"]
    assert all(
        e["response"].get("feedback", {}).get("accepted") for e in record["events"]
    )
    network = record["metrics"]["network_accounting"]
    assert network["maximum_balance_error_bits"] < 1e-6
    ground = network["stations"]["Late-Ground"]
    assert ground["terminal_excess"] == 0 and ground["rate_excess_bps"] == 0
    assert ground["interval_capacity_excess_bits"] < 1e-6
    if expected:
        assert {
            (r["source"], r["destination"]) for r in network["custody_history"]
        } == {("a", "r"), ("b", "s")}
        assert ground["transmitted_bits"] == 1000000.0
        if options.get("ground_terminals") == 2:
            assert ground["maximum_active_grants"] == 2
            assert ground["maximum_granted_rate_bps"] == 500000.0
        else:
            assert ground["contended_offer_sec"] > 0
    else:
        assert network["expired_transfers"] == 2
        assert ground["transmitted_bits"] == 0.0
    if options.get("ack_loss_probability"):
        assert network["retry_events"] > 0 and network["duplicate_received_bits"] > 0
        assert verify_episode(record)["success"]


def test_midtransfer_replay_and_future_catalogue_privacy():
    case = relay_campaign({"horizon_sec": 240.0, "release_sec": 60.0})
    session = AgentCampaignSession(case["configuration"])
    assert "late-a" not in json.dumps(session.observe()) and "late-b" not in json.dumps(
        session.observe()
    )
    while not any(
        e["kind"] == "transfer" for e in session.observe()["native"]["entities"]
    ):
        assert not session._done
        action = campaign_policy(session.observe(), case["public_contract"])
        session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    cp = session.checkpoint()
    restored = restore(cp)
    for _ in range(4):
        action = campaign_policy(session.observe(), case["public_contract"])
        assert action == campaign_policy(restored.observe(), case["public_contract"])
        assert session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        ) == restored.step(
            action,
            expected_revision=restored.revision,
            idempotency_key=str(restored.revision),
        )
    assert "network_accounting" not in json.dumps(session.observe())
    assert restored.checkpoint() == session.checkpoint()


def test_band_mismatch_enforces_relay_route():
    case = relay_campaign({"horizon_sec": 240.0, "release_sec": 60.0})
    session = AgentCampaignSession(case["configuration"])
    mission = session._native._session._runtime._backend.mission
    assert mission.radio_network.budget(mission, "a", "Late-Ground")["rate_bps"] == 0.0
    assert mission.radio_network.budget(mission, "b", "Late-Ground")["rate_bps"] == 0.0
    for actor in ["r", "s"]:
        assert (
            mission.radio_network.config.transmitters[actor].frequency_hz
            == 8200000000.0
        )


def test_pointing_is_opt_in_and_uses_current_error():
    view = observation()
    view["native"]["time_sec"] = 0.0
    point = row("point", "slew", "q")
    point["request_ref"] = "q"
    point["commands"][0]["slew"] = {"frame": "GCRS", "target_id": "site"}
    point["pointing"] = [{"boresight_error_deg": 8.0, "payload_tolerance_deg": 5.0}]
    view["native"]["catalogue"] = [point]
    contract = {
        "minimum_completed_requests": 1,
        "adaptive_wait": True,
        "request_pointing": True,
    }
    assert campaign_policy(view, contract)["choices"]["@central"] == ["point"]
    assert campaign_policy(view, {**contract, "request_pointing": False})["choices"][
        "@central"
    ] == ["wait"]
    point["pointing"][0]["boresight_error_deg"] = 1.0
    assert campaign_policy(view, contract)["choices"]["@central"] == ["wait"]
    point["enabled"] = False
    point["reason"] = "current_settling"
    assert campaign_policy(view, contract)["advance_sec"] == 0.25


def test_relay_profile_rejects_invalid_parameters():
    for value in [
        {"ack_loss_probability": 1.1},
        {"bundle_ttl_sec": 0.0},
        {"ground_terminals": True},
        {"network_sample_sec": 6.0},
        {"ground_rate_bps": 1.0},
    ]:
        with pytest.raises(ValueError):
            RelayCampaignSpec.model_validate(value)
