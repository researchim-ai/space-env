"""Ground-resource boundaries must constrain actual receipts, not just labels."""
import pytest

from space_env.simulation.commands import MissionCommand
from tests.test_execution_integrity import world


@pytest.mark.parametrize("steps", [(10,), (2, 4, 4), (1,) * 10])
def test_booking_inside_step_limits_physical_delivery(steps):
    w = world()
    w.use_book({"accepted": [{"vehicle": "a", "station": "Shared", "start_sec": 2, "end_sec": 6}]})
    for dt in steps:
        w.step(dt, {"a": "comms"})
    assert w.engines["a"].delivered_bits == pytest.approx(4e6)
    assert all(row["start_sec"] >= 2 and row["end_sec"] <= 6 for row in w.engines["a"].receipts)


@pytest.mark.parametrize("steps", [(10,), (2, 4, 4), (1,) * 10])
def test_outage_inside_step_excludes_closed_interval(steps):
    w = world()
    w.set_outages([("Shared", 2, 6)])
    for dt in steps:
        w.step(dt, {"a": "comms"})
    assert w.engines["a"].delivered_bits == pytest.approx(6e6)
    assert all(row["end_sec"] <= 2 or row["start_sec"] >= 6 for row in w.engines["a"].receipts)


@pytest.mark.parametrize("start,end", [(3, 2), (-1, 2), (0, float('nan')), (0, float('inf')), (2, 2)])
def test_bad_resource_intervals_are_rejected_atomically(start, end):
    w = world()
    w.set_outages([("Shared", 1, 2)])
    with pytest.raises(ValueError):
        w.set_outages([("Shared", start, end)])
    assert w.outages == [("Shared", 1., 2.)]
    w.use_book({"accepted": [{"vehicle": "a", "station": "Shared", "start_sec": 1, "end_sec": 2}]})
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.use_book({"accepted": [{"vehicle": "a", "station": "Shared", "start_sec": start, "end_sec": end}]})
    assert w.snapshot()["book"] == before["book"]


def test_resource_boundary_failure_restores_entire_external_step(monkeypatch):
    w = world()
    w.set_outages([("Shared", 2, 6)])
    orders = w.accept([MissionCommand("pass", "a", "comms", duration_sec=10)])
    original = w.engines["a"].step

    def injected(dt, command=None):
        if w.now_sec() >= 2:
            raise RuntimeError("resource boundary")
        return original(dt, command)

    monkeypatch.setattr(w.engines["a"], "step", injected)
    with pytest.raises(RuntimeError, match="resource boundary"):
        w.step(10, orders)
    assert w.now_sec() == 0
    assert not w.engines["a"].receipts
    assert w.commands.get("pass")["status"] == "accepted"


def test_timed_operation_and_outage_share_boundaries_and_json_replay():
    import json

    from space_env.simulation.snapshot import decode, encode

    w = world()
    w.set_outages([("Shared", 2, 4)])
    orders = w.accept([MissionCommand("pass", "a", "comms", duration_sec=6)])
    snapshot = decode(json.loads(json.dumps(encode(w.snapshot()))))
    w.step(10, orders)
    w.settle()
    expected = list(w.engines["a"].receipts)
    assert w.engines["a"].delivered_bits == pytest.approx(4e6)
    assert w.commands.get("pass")["status"] == "completed"
    assert all(row["end_sec"] <= 2 or 4 <= row["start_sec"] < row["end_sec"] <= 6 for row in expected)
    w.restore(snapshot)
    w.step(10, orders)
    w.settle()
    assert list(w.engines["a"].receipts) == expected
