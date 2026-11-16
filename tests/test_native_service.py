import json

import pytest

from space_env.industrial import NativeServiceSession
from space_env.industrial.native_service import (
    native_service_policy,
    native_service_recipe,
)
from space_env.runtime import RuntimeContractError


def step(session, action):
    return session.step(
        action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def run(session):
    while not session._done:
        step(session, native_service_policy(session.observe()))
    return session


def test_nominal_real_custody_ground_ack_and_native_resources():
    session = run(NativeServiceSession(native_service_recipe()))
    metrics = session.metrics()
    assert metrics["success"] and metrics["delivered_bits"] == 700000
    assert metrics["maximum_balance_error_bits"] < 1e-6
    report = session._runtime._backend.report()
    assert report["dtn"]["custody_history"]
    assert any(r["consumed_wh"] > 0 for r in report["energy_ledger"].values())
    assert session._runtime.time_sec == 60.0
    assert (
        NativeServiceSession.from_checkpoint(
            json.loads(json.dumps(session.checkpoint()))
        ).metrics()
        == metrics
    )


def test_new_stream_and_chunks_not_preloaded_or_leaked():
    s = NativeServiceSession(native_service_recipe())
    assert s.metrics()["released_chunks"] == 1
    assert "urgent" not in json.dumps(s.observe())
    future = next(pid for at, fid, pid, b in s._schedule if fid == "urgent")
    assert future not in json.dumps(s.observe())
    step(s, {})
    step(s, {})
    assert future in json.dumps(s.observe()) and "urgent" in json.dumps(s.observe())


def test_true_band_mismatch_does_not_deliver_at_a_supplied_capacity():
    s = NativeServiceSession(native_service_recipe())
    actor = "camera-00"
    pid = s._runtime.inventory(actor)["products"][0]["id"]
    assert all(
        r["rate_bps"] == 0 for r in s.observe()["ground_links"] if r["actor"] == actor
    )
    step(
        s,
        {
            "commands": [
                {
                    "actor_id": actor,
                    "kind": "downlink",
                    "duration_sec": 2.0,
                    "product_id": pid,
                    "destination_id": "Ground",
                }
            ]
        },
    )
    assert s.metrics()["delivered_bits"] == 0 and s._runtime.time_sec == 2.0


def test_memory_admission_rejection_kept_in_denominator():
    s = NativeServiceSession(native_service_recipe("low_memory"))
    step(s, {})
    step(s, {})
    assert s.metrics()["rejected_chunks"] >= 1
    assert s.metrics()["total_flows"] == 2
    assert s.metrics()["maximum_balance_error_bits"] < 1e-6


def test_native_prefix_keeps_mass_energy_and_absolute_clock():
    cfg = native_service_recipe("prefix")
    s = NativeServiceSession(cfg)
    assert s._runtime.checkpoint(compact=True) == cfg["prefix_checkpoint"]
    assert s.observe()["time_sec"] == 4.0 and not s.metrics()["released_chunks"]
    orbit = s._runtime.observe("camera-00")["actors"]["camera-00"]["orbit"]
    assert orbit["propellant_kg"] < 10.0
    step(s, {})
    assert s.observe()["time_sec"] == 6.0 and s.metrics()["service_started_sec"] == 4.0
    assert NativeServiceSession.from_checkpoint(s.checkpoint()).observe() == s.observe()


def test_cancellation_drains_inflight_data_before_discard():
    cfg = native_service_recipe("cancel_inflight")
    cfg["decision_sec"] = 0.1
    # Ten thousand decisions would exceed the configured bound; shorten the
    # declared mission itself, preserving the physical native source profile.
    cfg["decision_sec"] = 0.5
    s = NativeServiceSession(cfg)
    pid = next(iter(s._chunks))
    step(
        s,
        {
            "commands": [
                {
                    "actor_id": "camera-00",
                    "kind": "relay",
                    "duration_sec": 2.0,
                    "destination_id": "relay-00",
                    "product_id": pid,
                    "relay": {"ttl_sec": 20.0},
                }
            ]
        },
    )
    mission = s._runtime._backend.mission
    assert s._flow["routine"]["cancelled"]
    assert mission.dtn.pinned(pid) and mission.remaining[pid] > 0
    assert any(r["cancel_requested"] for r in mission.dtn.transfers.values())
    for _ in range(4):
        step(s, {})
    assert not mission.dtn.pinned(pid) or not s._runtime._backend.mission.dtn.pinned(
        pid
    )
    assert s.metrics()["maximum_balance_error_bits"] < 1e-6


def test_joint_command_rejection_does_not_suppress_external_event():
    cfg = native_service_recipe("receiver_failure")
    s = NativeServiceSession(cfg)
    step(s, {})
    response = step(
        s,
        {
            "commands": [
                {
                    "actor_id": "camera-00",
                    "kind": "downlink",
                    "duration_sec": 2.0,
                    "destination_id": "Ground",
                    "product_id": "unknown",
                }
            ]
        },
    )
    assert not response["feedback"]["accepted"]
    assert (
        "radio"
        in s._runtime.observe("relay-00")["actors"]["relay-00"]["spacecraft_bus"][
            "faults"
        ]
    )


def test_completed_flow_priority_and_cancel_do_not_rewrite_past_value():
    cfg = native_service_recipe()
    cfg["streams"] = cfg["streams"][:1]
    cfg["streams"][0].update(size_bits=100000, deadline_sec=40.0)
    cfg["events"] = [
        {
            "at_sec": 40.0,
            "type": "flow_priority_changed",
            "payload": {"flow_id": "routine", "priority": 999},
        },
        {"at_sec": 42.0, "type": "flow_cancelled", "payload": {"flow_id": "routine"}},
    ]
    s = run(NativeServiceSession(cfg))
    assert (
        s.metrics()["completed_priority"] == 2 and s.metrics()["cancelled_flows"] == 0
    )


@pytest.mark.parametrize(
    "action",
    [
        {"cancel_actors": ["unknown"]},
        {"commands": [{"actor_id": "unknown", "kind": "wait", "duration_sec": 2.0}]},
        {"inject": {"type": "spacecraft_fault"}},
    ],
)
def test_invalid_decision_is_atomic(action):
    s = NativeServiceSession(native_service_recipe())
    before = s.checkpoint()
    with pytest.raises(ValueError):
        step(s, action)
    assert s.checkpoint() == before


def test_observation_is_detached_and_no_shadow_resource_state():
    s = NativeServiceSession(native_service_recipe())
    before = s.checkpoint()
    obs = s.observe()
    obs["actors"]["camera-00"]["soc"] = 0
    obs["flows"][0]["priority"] = 999
    assert s.checkpoint() == before
    assert not hasattr(s, "_nodes_state")


def test_incompatible_prefix_rejected():
    cfg = native_service_recipe("prefix")
    cfg["prefix_checkpoint"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeContractError)):
        NativeServiceSession(cfg)


def test_new_service_over_old_service_prefix_uses_fresh_command_keys():
    previous = NativeServiceSession(native_service_recipe())
    step(previous, native_service_policy(previous.observe()))
    inherited = previous._runtime.checkpoint(compact=True)
    resumed = NativeServiceSession(
        {
            "prefix_checkpoint": inherited,
            "streams": [
                {
                    "id": "followup",
                    "source": "camera-01",
                    "destination": "Ground",
                    "size_bits": 100000,
                    "release_sec": 2.0,
                    "deadline_sec": 40.0,
                    "priority": 3,
                }
            ],
        }
    )
    before = {r["command"]["command_id"] for r in resumed._runtime.command_status()}
    response = step(resumed, native_service_policy(resumed.observe()))
    assert (
        response["feedback"]["accepted"] and resumed.metrics()["native_rejections"] == 0
    )
    after = {r["command"]["command_id"] for r in resumed._runtime.command_status()}
    assert len(after) > len(before)
    assert (
        NativeServiceSession.from_checkpoint(resumed.checkpoint()).checkpoint()
        == resumed.checkpoint()
    )


def test_new_service_events_do_not_reuse_prefix_event_keys():
    cfg = native_service_recipe()
    cfg["events"] = [
        {
            "at_sec": 1.0,
            "type": "station_outage_added",
            "payload": {"station": "Ground", "start_sec": 1.0, "end_sec": 8.0},
        }
    ]
    previous = NativeServiceSession(cfg)
    step(previous, {})
    resumed = NativeServiceSession(
        {
            "prefix_checkpoint": previous._runtime.checkpoint(compact=True),
            "streams": [
                {
                    "id": "followup",
                    "source": "camera-01",
                    "destination": "Ground",
                    "size_bits": 100000,
                    "release_sec": 2.0,
                    "deadline_sec": 40.0,
                }
            ],
            "events": [
                {
                    "at_sec": 3.0,
                    "type": "station_recovered",
                    "payload": {"station": "Ground"},
                }
            ],
        }
    )
    assert resumed._runtime.station_observation("Ground")["outage"]
    step(resumed, {})
    assert not resumed._runtime.station_observation("Ground")["outage"]
    assert (
        NativeServiceSession.from_checkpoint(resumed.checkpoint()).checkpoint()
        == resumed.checkpoint()
    )


def test_caller_owned_model_cannot_change_future_stream_events():
    from space_env.industrial.native_service import NativeServiceSpec

    cfg = NativeServiceSpec.model_validate(native_service_recipe("cancel_inflight"))
    session = NativeServiceSession(cfg)
    cfg.events[0].payload["flow_id"] = "unknown-after-construction"
    step(session, {})
    assert session._flow["routine"]["cancelled"]
    assert session._configuration["events"][0]["payload"]["flow_id"] == "routine"
