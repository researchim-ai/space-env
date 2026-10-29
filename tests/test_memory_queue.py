"""Очередь памяти: вытеснение, обработка и доля сжатия."""
import numpy as np
import pytest

from space_env.core.orbit import Orbit
from space_env.core.power import step_battery_soc
from space_env.core.spacecraft import PayloadConfig, PowerConfig, Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")


def _balance(engine: SimulationEngine) -> None:
    snap = engine.get_state_snapshot()
    assert snap.initial_bits + snap.produced_bits + snap.received_bits == pytest.approx(
        snap.stored_bits + snap.delivered_bits + snap.relayed_out_bits + snap.lost_bits
    )


def _engine(**payload) -> SimulationEngine:
    fields = {"image_bits": 8.0, "memory_bits": 10.0}
    fields.update(payload)
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, EPOCH)
    craft = Spacecraft(
        config=SpacecraftConfig(
            payload=PayloadConfig(**fields),
            power=PowerConfig(solar_panel_area_m2=0.0, battery_capacity_Wh=10.0, consumption_nominal_W=0.0),
        )
    )
    craft.reset_state()
    return SimulationEngine(orbit, craft, scenario=ScenarioConfig(ground_stations=[]))


def test_reject_still_discards_the_new_overflow():
    engine = _engine()
    engine._store_capture("frame-a", 0.0, 1.0)
    engine._store_capture("frame-b", 0.0, 1.0)
    snap = engine.get_state_snapshot()
    assert snap.produced_bits == pytest.approx(16.0)
    assert snap.stored_bits == pytest.approx(10.0)
    assert snap.lost_bits == pytest.approx(6.0)
    assert engine._products[0]["onboard_bits"] == pytest.approx(8.0)
    assert engine._products[1]["quality"] == "partial"
    _balance(engine)


def test_drop_oldest_removes_a_whole_earlier_product():
    engine = _engine(memory_policy="drop_oldest")
    engine._store_capture("frame-a", 0.0, 1.0)
    engine._store_capture("frame-b", 0.0, 1.0)
    snap = engine.get_state_snapshot()
    assert engine._products[0]["delivered_to"] == "evicted"
    assert engine._products[0]["onboard_bits"] == pytest.approx(0.0)
    assert engine._products[1]["onboard_bits"] == pytest.approx(8.0)
    assert engine._products[1]["quality"] == "full"
    assert snap.stored_bits == pytest.approx(8.0)
    assert snap.lost_bits == pytest.approx(8.0)
    assert snap.produced_bits == pytest.approx(16.0)
    _balance(engine)


def test_compression_keeps_only_the_configured_fraction():
    plain = _engine()
    plain._store_capture("frame", 0.0, 1.0)
    packed = _engine(compression_ratio=0.5, memory_bits=0.0)
    packed._store_capture("frame", 0.0, 1.0)
    assert plain.get_state_snapshot().produced_bits == pytest.approx(8.0)
    assert packed.get_state_snapshot().produced_bits == pytest.approx(4.0)
    assert packed.get_state_snapshot().stored_bits == pytest.approx(4.0)
    _balance(packed)


def test_the_processor_holds_bits_until_they_are_finished_and_draws_power():
    engine = _engine(memory_bits=0.0, process_rate_bps=1.0, process_power_w=360.0)
    engine._store_capture("frame", 0.0, 1.0)
    assert engine._products[0]["available_time_sec"] == pytest.approx(1.0e30)
    assert engine._move_downlink(10.0, 10.0, "gs", 0.0, 10.0) == 0.0
    held = engine.capture()
    quiet = _engine(memory_bits=0.0, process_rate_bps=1.0, process_power_w=0.0)
    quiet._store_capture("frame", 0.0, 1.0)
    engine.step(4.0, "wait")
    assert engine._products[0]["available_time_sec"] == pytest.approx(1.0e30)
    assert len(engine._process_queue) == 1
    engine.restore(held)
    assert engine._products[0]["available_time_sec"] == pytest.approx(1.0e30)
    engine.step(8.0, "wait")
    assert engine._products[0]["available_time_sec"] == pytest.approx(8.0)
    assert engine._process_queue == []
    moved = engine._move_downlink(10.0, 10.0, "gs", 8.0, 18.0)
    assert moved == pytest.approx(8.0)
    expected = step_battery_soc(1.0, 10.0, 360.0, 0.0, 8.0)
    assert engine.spacecraft.state.power.battery_soc == pytest.approx(expected, abs=1e-9)
    quiet.step(8.0, "wait")
    assert quiet.spacecraft.state.power.battery_soc == pytest.approx(1.0)
    _balance(engine)
    assert np.isfinite(engine.get_state_snapshot().stored_bits)
