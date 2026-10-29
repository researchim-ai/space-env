"""Режимы борта: подготовка съёмки, занятый передатчик и пустая батарея."""
import numpy as np

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PayloadConfig, Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.events import EventKind, PlannedEvent
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine

EPOCH = Time("2024-01-01T00:00:00Z")


def _imaging_engine(*, prep_sec: float = 0.0, exclusive: bool = False, soc: float = 1.0, planned=None) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    axis = np.array([0.0, 0.0, 1.0])
    velocity = np.cross(radius, axis)
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(
        config=SpacecraftConfig(
            payload=PayloadConfig(prep_sec=prep_sec, exclusive_with_downlink=exclusive),
        )
    )
    craft.reset_state()
    craft.state.power.battery_soc = soc
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[], planned_events=list(planned or [])),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine._now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    return engine


def test_imaging_starts_at_once_when_prep_is_zero():
    engine = _imaging_engine()
    snap, _events = engine.step(10.0, "imaging")
    assert engine.last_command_ok is True
    assert snap.imaging_count == 1


def test_prep_holds_the_shutter_until_it_elapses():
    engine = _imaging_engine(prep_sec=15.0)
    first, _events = engine.step(10.0, "imaging")
    assert engine.last_command_ok is True
    assert first.imaging_count == 0
    second, _events = engine.step(10.0, "imaging")
    assert second.imaging_count == 1


def test_an_armed_radio_blocks_imaging_when_the_payload_is_exclusive():
    engine = _imaging_engine(exclusive=True)
    engine._downlink_armed = True
    snap, _events = engine.step(10.0, "imaging")
    assert engine.last_command_ok is False
    assert snap.imaging_count == 0


def test_a_dead_battery_does_not_start_imaging_or_the_radio():
    imaging = _imaging_engine(soc=0.0)
    snap, _events = imaging.step(10.0, "imaging")
    assert imaging.last_command_ok is False
    assert snap.imaging_count == 0
    radio = _imaging_engine(soc=0.0)
    radio._in_contact = True
    radio.step(10.0, "comms")
    assert radio.last_command_ok is False
    assert radio._downlink_armed is False
    live = _imaging_engine()
    live._in_contact = True
    live.step(10.0, "comms")
    assert live.last_command_ok is True


def test_exclusive_payload_refuses_comms_over_a_planned_exposure():
    planned = [PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=0.0, duration_sec=20.0)]
    engine = _imaging_engine(exclusive=True, planned=planned)
    engine._in_contact = True
    engine.step(10.0, "comms")
    assert engine.last_command_ok is False
    assert engine._downlink_armed is False
