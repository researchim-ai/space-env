import copy

import pytest

from space_env.industrial import ServiceSession
from space_env.industrial.service import service_policy


def configuration():
    return {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "g", "kind": "ground"}],
        "links": [
            {"id": "ab", "source": "a", "destination": "b", "bits_per_slot": 100},
            {"id": "bg", "source": "b", "destination": "g", "bits_per_slot": 100},
        ],
        "flows": [
            {
                "id": "f",
                "source": "a",
                "destination": "g",
                "size_bits": 100,
                "deadline_step": 4,
            }
        ],
        "horizon_steps": 4,
    }


def step(session, action):
    return session.step(
        action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def transfer(link="ab", flow="f", bits=100):
    return {"link_id": link, "flow_id": flow, "bits": bits}


def test_two_hop_custody_and_ground_receipt():
    s = ServiceSession(configuration())
    step(s, {"transfers": [transfer()]})
    assert s.metrics()["delivered_bits"] == 0
    step(s, {"transfers": [transfer("bg")]})
    while not s._done:
        step(s, {})
    assert s.metrics()["success"] and s.metrics()["queued_bits"] == 0
    assert ServiceSession.from_checkpoint(s.checkpoint()).checkpoint() == s.checkpoint()


def test_same_slot_forwarding_and_shared_terminals_rejected_atomically():
    s = ServiceSession(configuration())
    before = s.checkpoint()
    with pytest.raises(ValueError):
        step(s, {"transfers": [transfer(), transfer("bg")]})
    assert s.checkpoint() == before


def test_new_flow_and_future_outage_not_visible():
    cfg = configuration()
    cfg["flows"][0]["release_step"] = 2
    cfg["events"] = [{"at_step": 2, "kind": "node_fault", "target": "b"}]
    s = ServiceSession(cfg)
    assert not s.observe()["flows"] and "events" not in s.observe()
    step(s, {})
    step(s, {})
    assert len(s.observe()["flows"]) == 1 and s._nodes_state["b"]["fault"]


def test_deadline_completion_at_boundary_counts():
    cfg = configuration()
    cfg["flows"][0]["deadline_step"] = 2
    s = ServiceSession(cfg)
    step(s, {"transfers": [transfer()]})
    step(s, {"transfers": [transfer("bg")]})
    assert s._flows["f"]["status"] == "completed"


def test_partial_delivery_expiry_conserves_bits():
    cfg = configuration()
    cfg["flows"][0]["deadline_step"] = 2
    s = ServiceSession(cfg)
    step(s, {"transfers": [transfer(bits=50)]})
    step(s, {"transfers": [transfer("bg", bits=50)]})
    assert s.metrics()["delivered_bits"] == 50 and s.metrics()["discarded_bits"] == 50
    assert s.metrics()["expired_flows"] == 1


def test_queue_overflow_at_release_is_retained():
    cfg = configuration()
    cfg["nodes"][0]["queue_bits"] = 50
    s = ServiceSession(cfg)
    assert s.metrics()["rejected_flows"] == 1 and s.metrics()["total_flows"] == 1
    assert s.metrics()["discarded_bits"] == 100


def test_cancel_preserves_delivered_part_and_denominator():
    cfg = configuration()
    cfg["events"] = [{"at_step": 2, "kind": "flow_cancelled", "target": "f"}]
    s = ServiceSession(cfg)
    step(s, {"transfers": [transfer(bits=50)]})
    step(s, {"transfers": [transfer("bg", bits=50)]})
    assert s.metrics()["cancelled_flows"] == 1
    assert s.metrics()["delivered_bits"] == 50 and s.metrics()["discarded_bits"] == 50


def test_reserve_requires_commissioning_and_fault_clear():
    cfg = configuration()
    cfg["nodes"][1].update(reserve=True, commissioning_steps=2)
    cfg["events"] = [
        {"at_step": 2, "kind": "node_fault", "target": "b"},
        {"at_step": 3, "kind": "node_recovered", "target": "b"},
    ]
    s = ServiceSession(cfg)
    step(s, {"activate": ["b"]})
    assert not s._available("b")
    step(s, {})
    assert not s._available("b")
    step(s, {})
    assert s._available("b")


def test_handover_cost_and_no_silent_peer_switch():
    cfg = configuration()
    cfg["links"].append(
        {"id": "ag", "source": "a", "destination": "g", "bits_per_slot": 100}
    )
    s = ServiceSession(cfg)
    step(s, {"transfers": [transfer(bits=25)]})
    with pytest.raises(ValueError, match="handover"):
        step(s, {"transfers": [transfer("ag", bits=75)]})
    step(s, {"handovers": [{"node_id": "a", "destination": "g"}]})
    step(s, {"transfers": [transfer("ag", bits=75)]})
    assert s.metrics()["delivered_bits"] == 75


@pytest.mark.parametrize("field,value", [("bits", True), ("bits", 0), ("bits", 101)])
def test_bad_allocations_are_atomic(field, value):
    s = ServiceSession(configuration())
    before = s.checkpoint()
    allocation = transfer()
    allocation[field] = value
    with pytest.raises(ValueError):
        step(s, {"transfers": [allocation]})
    assert s.checkpoint() == before


def test_energy_and_receiver_capacity_gate_whole_batch():
    for field, value in [("energy_j", 1.0), ("queue_bits", 50)]:
        cfg = configuration()
        cfg["nodes"][1][field] = value
        s = ServiceSession(cfg)
        before = s.checkpoint()
        with pytest.raises(ValueError):
            step(s, {"transfers": [transfer()]})
        assert s.checkpoint() == before


def test_public_scheduler_delivers_without_future_inputs():
    s = ServiceSession(configuration())
    while not s._done:
        step(s, service_policy(s.observe()))
    assert s.metrics()["success"]


def test_observation_is_detached():
    s = ServiceSession(configuration())
    before = copy.deepcopy(s.observe())
    view = s.observe()
    view["flows"][0]["queues"]["a"] = 9999
    assert s.observe() == before
