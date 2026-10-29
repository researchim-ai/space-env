"""Подготовка антенны: видимость станции не отдаёт биты до конца setup."""
import numpy as np
import pytest

from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from tests.test_dynamics import _engine_at


def _overhead(setup_sec: float):
    epoch = Time("2024-03-01T00:00:00Z")
    engine = _engine_at(
        ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    engine.spacecraft.config.comms.setup_sec = setup_sec
    engine.accept_bits(2.0e6, origin="imaging", request_id="frame")
    return engine


def test_zero_setup_still_sends_during_the_contact():
    engine = _overhead(0.0)
    engine.step(5.0, "comms")
    assert engine.get_state_snapshot().delivered_bits > 0.0


def test_setup_holds_the_bits_until_the_antenna_is_ready():
    engine = _overhead(20.0)
    engine.step(5.0, "comms")
    assert engine.get_state_snapshot().delivered_bits == 0.0
    assert engine._setup_until == pytest.approx(20.0)
    engine.step(10.0, "wait")
    assert engine.get_state_snapshot().delivered_bits == 0.0
    held = engine.capture()
    engine.step(10.0, "wait")
    assert engine.get_state_snapshot().delivered_bits > 0.0
    engine.restore(held)
    assert engine.get_state_snapshot().delivered_bits == 0.0
    assert engine._setup_until == pytest.approx(20.0)
    assert np.isfinite(engine.time_sec)
