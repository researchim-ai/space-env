import json

import pytest

from space_env.industrial import OperationsSession


def step(session, action):
    return session.step(
        action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def fault(at=0.0, subsystem="radio"):
    return {
        "at_sec": at,
        "event": {
            "type": "spacecraft_fault",
            "payload": {"actor": "camera-00", "subsystem": subsystem},
        },
    }


def test_future_event_not_in_public_view_and_driver_is_not_an_action():
    session = OperationsSession({"events": [fault(10.0)]})
    view = session.observe()
    assert "@driver" not in json.dumps(view)
    assert "spacecraft_fault" not in json.dumps(view)
    for role, obs in view["roles"].items():
        assert all(
            r["id"] == "wait" or session._public_actions[r["id"]]["role"] == role
            for r in obs["catalogue"]
        )
    hidden = session._event_actions[0]
    before = session.checkpoint()
    with pytest.raises(ValueError, match="privileged"):
        step(session, {"choices": {"camera-00": [hidden]}})
    assert session.checkpoint() == before


def test_diagnostic_and_delayed_reset_use_native_time():
    session = OperationsSession({"events": [fault()]})
    assert session.observe()["time_sec"] == 2.0
    response = step(session, {"kind": "diagnose", "actor": "camera-00"})
    assert response["elapsed_sec"] == 2.0
    assert response["observation"]["diagnostics"]["camera-00"]["reported_faults"] == [
        "radio"
    ]
    response = step(
        session, {"kind": "reset", "actor": "camera-00", "subsystem": "radio"}
    )
    assert response["elapsed_sec"] == 8.0
    assert response["feedback"]["accepted"]
    bus = session._native._runtime.observe("camera-00")["actors"]["camera-00"][
        "spacecraft_bus"
    ]
    assert not bus["faults"]
    assert (
        OperationsSession.from_checkpoint(session.checkpoint()).checkpoint()
        == session.checkpoint()
    )


def test_permanent_fault_cannot_be_magically_reset():
    session = OperationsSession(
        {"events": [fault()], "permanent_faults": [["camera-00", "radio"]]}
    )
    response = step(
        session, {"kind": "reset", "actor": "camera-00", "subsystem": "radio"}
    )
    assert response["feedback"]["code"] == "reset_ineffective"
    assert (
        "radio"
        in session._native._runtime.observe("camera-00")["actors"]["camera-00"][
            "spacecraft_bus"
        ]["faults"]
    )


def test_actual_request_addition_is_hidden_until_native_event():
    session = OperationsSession({"deferred_request_at_sec": 4.0})
    event = next(e for _, e in session._stimuli if e["type"] == "request_added")
    rid = event["payload"]["request"]["id"]
    assert rid not in json.dumps(session.observe())
    assert session.metrics()["total_requests"] == 3
    step(session, {})
    step(session, {})
    assert rid in json.dumps(session.observe())
    assert session.metrics()["total_requests"] == 4
    assert session.metrics()["environment_rejections"] == 0


def test_native_rejection_does_not_suppress_environment():
    session = OperationsSession({"events": [fault(2.0)]})
    # The first action is individually infeasible but remains a legal actor ID.
    row = next(
        r
        for r in session.observe()["roles"]["camera-00"]["catalogue"]
        if r.get("reason") == "product_not_local"
    )
    response = step(session, {"choices": {"camera-00": [row["id"]]}})
    assert not response["feedback"]["accepted"]
    assert session._environment_log[0]["accepted"]
    assert session.observe()["time_sec"] == 4.0


@pytest.mark.parametrize(
    "config",
    [
        {"events": [fault(3.0)]},
        {"diagnostic_sec": 3.0},
        {"permanent_faults": [["unknown", "cpu"]]},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        OperationsSession(config)
