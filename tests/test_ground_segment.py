"""Наземный сегмент: терминалы, частота, срок продукта и подтверждение сдачи."""
import pytest

from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.world import MissionWorld
from tests.test_dynamics import _engine_at

_EPOCH = Time("2024-03-01T00:00:00Z")


def _overhead(name: str, *, terminals: int = 1, band_hz: float = 0.0):
    station = GroundStation(
        name="Equator",
        lat_deg=0.0,
        lon_deg=0.0,
        min_elevation_deg=5.0,
        terminals=terminals,
        band_hz=band_hz,
    )
    engine = _engine_at(
        ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), _EPOCH),
        _EPOCH,
        scenario=ScenarioConfig(ground_stations=[station]),
    )
    engine.spacecraft.config.name = name
    engine.accept_bits(2.0e6, origin="imaging", request_id=name)
    return engine


def test_one_terminal_keeps_a_single_downlink_and_two_do_not():
    single = MissionWorld(
        {"a": _overhead("a"), "b": _overhead("b")},
        epoch="2024-03-01T00:00:00Z",
        horizon_sec=30.0,
    )
    single.step(5.0, {"a": "comms", "b": "comms"})
    assert single.engines["a"].delivered_bits > 0.0
    assert single.engines["b"].delivered_bits == 0.0
    assert single.last_rejected == [{"actor_id": "b", "kind": "comms", "reason": "станция занята"}]

    both = MissionWorld(
        {"a": _overhead("a", terminals=2), "b": _overhead("b", terminals=2)},
        epoch="2024-03-01T00:00:00Z",
        horizon_sec=30.0,
    )
    both.step(5.0, {"a": "comms", "b": "comms"})
    assert both.engines["a"].delivered_bits > 0.0
    assert both.engines["b"].delivered_bits > 0.0
    assert both.last_rejected == []


def test_a_foreign_band_does_not_open_the_station():
    engine = _overhead("a", band_hz=8.0e9)
    engine.step(5.0, "comms")
    assert engine.last_command_ok is False
    assert engine.delivered_bits == 0.0
    assert engine.get_state_snapshot().in_contact is False


def test_ttl_drops_the_product_before_the_pass_and_ack_is_after_the_bits():
    expired = _overhead("a")
    expired.spacecraft.config.comms.ttl_sec = 3.0
    expired.step(5.0, "comms")
    assert expired.delivered_bits == 0.0
    assert expired.get_state_snapshot().lost_bits == pytest.approx(2.0e6)
    assert expired._products[0]["delivered_to"] == "expired"

    acked = _overhead("a")
    acked.spacecraft.config.comms.ack_sec = 4.0
    acked.step(5.0, "comms")
    receipt = acked.receipts[0]
    assert receipt["end_sec"] >= receipt["start_sec"]
    assert receipt["time_sec"] == pytest.approx(receipt["end_sec"] + 4.0)
    onboard = acked._products[0]["onboard_bits"]
    slips = len(acked.receipts)
    acked._mark_delivered(2.0e6, "Equator", 0.0, 1.0)
    assert len(acked.receipts) == slips
    assert acked._products[0]["onboard_bits"] == pytest.approx(onboard)
