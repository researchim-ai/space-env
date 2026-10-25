"""Физика тени, термики, импульса, связи и ограничений."""
import numpy as np
import pytest

from space_env.core.constants import AU, MU_EARTH
from space_env.core.frames import ecef_to_eci, elevation_deg, geodetic_to_ecef
from space_env.core.packets import decode_space_packet_header, encode_space_packet
from space_env.core.access import subsatellite_latlon
from space_env.core.sun import sun_position_m
from space_env.mission.scenarios import GroundTarget
from space_env.core.thermal import step_temperature_c
from space_env.core.time import Time
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import PowerConfig, PropulsionConfig, Spacecraft, SpacecraftConfig, SubsystemHealth
from space_env.mission import Constraints, EventKind, PlannedEvent, ScenarioConfig
from space_env.mission.events import EventKind as Kind
from space_env.simulation import SimulationEngine


def _engine_at(r_m: np.ndarray, epoch: Time, **kwargs) -> SimulationEngine:
    speed = np.sqrt(MU_EARTH / np.linalg.norm(r_m))
    sun = r_m / np.linalg.norm(r_m)
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, sun))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    vdir = np.cross(sun, axis)
    vdir = vdir / np.linalg.norm(vdir)
    orbit = Orbit.from_vectors(r_m, vdir * speed, epoch)
    craft = Spacecraft(
        config=SpacecraftConfig(
            power=PowerConfig(
                solar_panel_area_m2=2.0,
                battery_capacity_Wh=100.0,
                consumption_nominal_W=50.0,
                consumption_peak_W=80.0,
            ),
            propulsion=PropulsionConfig(delta_v_total_m_s=100.0, mass_kg=50.0),
        )
    )
    craft.reset_state()
    return SimulationEngine(orbit, craft, **kwargs)


def test_sun_distance_and_season():
    january = sun_position_m(Time("2024-01-01T00:00:00Z"))
    july = sun_position_m(Time("2024-07-01T00:00:00Z"))
    assert np.linalg.norm(january) == pytest.approx(AU, rel=0.03)
    assert np.linalg.norm(july) == pytest.approx(AU, rel=0.03)
    assert float(np.dot(january, july)) < 0.0


def test_temperature_responds_to_flux():
    hot = step_temperature_c(
        20.0, 60.0, q_sun_w=2000.0, q_diss_w=0.0, q_heater_w=0.0,
        radiating_area_m2=0.1, emissivity=0.8, heat_capacity_j_k=5000.0,
    )
    cold = step_temperature_c(
        20.0, 60.0, q_sun_w=0.0, q_diss_w=0.0, q_heater_w=0.0,
        radiating_area_m2=2.0, emissivity=0.9, heat_capacity_j_k=1000.0,
    )
    assert hot > 20.0
    assert cold < 20.0


def test_zenith_pass_has_high_elevation():
    epoch = Time("2024-03-01T00:00:00Z")
    r_ecef = geodetic_to_ecef(0.0, 0.0, 400_000.0)
    r_eci = ecef_to_eci(r_ecef, epoch)
    elevation = elevation_deg(r_eci, 0.0, 0.0, 0.0, epoch)
    assert elevation > 80.0


def test_umbra_discharges_battery_and_sunlight_does_not():
    epoch = Time("2024-06-15T00:00:00Z")
    sun = sun_position_m(epoch)
    s_hat = sun / np.linalg.norm(sun)
    dark = _engine_at(-s_hat * 7.0e6, epoch, scenario=ScenarioConfig(ground_stations=[]))
    assert dark.get_state_snapshot().in_eclipse is True
    snap, _events = dark.step(120.0)
    assert snap.battery_soc < 1.0

    lit = _engine_at(s_hat * 7.0e6, epoch, scenario=ScenarioConfig(ground_stations=[]))
    assert lit.get_state_snapshot().in_eclipse is False
    lat, lon = subsatellite_latlon(lit.orbit.r, lit._now())
    lit.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    lit_snap, events = lit.step(10.0, "imaging")
    assert lit.last_command_ok is True
    assert lit_snap.imaging_count == 1
    assert any(event.kind == EventKind.IMAGING for event in events)


def test_prograde_burn_increases_speed_and_spends_delta_v():
    epoch = Time("2024-01-01T00:00:00Z")
    sun = sun_position_m(epoch)
    s_hat = sun / np.linalg.norm(sun)
    engine = _engine_at(s_hat * 7.0e6, epoch, scenario=ScenarioConfig(ground_stations=[]))
    speed_before = float(np.linalg.norm(engine.orbit.v))
    snap, events = engine.step(10.0, "maneuver", {"dv_m_s": 10.0, "direction": "prograde"})
    assert engine.last_command_ok is True
    assert snap.delta_v_used_m_s == pytest.approx(10.0, abs=1e-6)
    assert float(np.linalg.norm(engine.orbit.v)) > speed_before + 5.0
    assert any(event.kind == EventKind.MANEUVER for event in events)


def test_low_soc_emits_constraint_once():
    epoch = Time("2024-01-01T00:00:00Z")
    sun = sun_position_m(epoch)
    s_hat = sun / np.linalg.norm(sun)
    scenario = ScenarioConfig(initial_battery_soc=0.1, ground_stations=[])
    engine = _engine_at(
        s_hat * 7.0e6,
        epoch,
        scenario=scenario,
        constraints=Constraints(battery_soc_min=0.2),
    )
    _snap, first = engine.step(5.0)
    _snap, second = engine.step(5.0)
    assert sum(event.kind == Kind.CONSTRAINT_VIOLATION for event in first) == 1
    assert all(event.kind != Kind.CONSTRAINT_VIOLATION for event in second)


def test_planned_imaging_runs_when_lit():
    epoch = Time("2024-01-01T00:00:00Z")
    sun = sun_position_m(epoch)
    s_hat = sun / np.linalg.norm(sun)
    scenario = ScenarioConfig(
        ground_stations=[],
        planned_events=[PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=10.0)],
    )
    engine = _engine_at(s_hat * 7.0e6, epoch, scenario=scenario)
    lat, lon = subsatellite_latlon(engine.orbit.r, engine._now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon))
    snap, events = engine.step(10.0)
    assert snap.imaging_count == 1
    assert any(event.kind == EventKind.IMAGING for event in events)


def test_downlink_does_not_invent_bits_and_the_recorder_balances():
    epoch = Time("2024-03-01T00:00:00Z")
    r_ecef = geodetic_to_ecef(0.0, 0.0, 400_000.0)
    from space_env.core.ground import GroundStation

    engine = _engine_at(
        ecef_to_eci(r_ecef, epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    engine.step(5.0, "comms")
    empty = engine.get_state_snapshot()
    assert empty.data_downlink_Mb == 0.0
    assert empty.delivered_bits == 0.0
    engine._stored_bits = 2.0e6
    engine._generated_bits = 2.0e6
    engine.step(5.0, "comms")
    filled = engine.get_state_snapshot()
    assert filled.delivered_bits > 0.0
    assert filled.generated_bits == pytest.approx(
        filled.stored_bits + filled.delivered_bits + filled.lost_bits
    )


def test_drag_context_uses_the_spacecraft_not_the_station():
    epoch = Time("2024-06-01T00:00:00Z")
    r_ecef = geodetic_to_ecef(10.0, 20.0, 500_000.0)
    from space_env.core.frames import ecef_to_geodetic, eci_to_ecef
    from space_env.core.ground import GroundStation

    engine = _engine_at(
        ecef_to_eci(r_ecef, epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Far", lat_deg=-40.0, lon_deg=140.0, min_elevation_deg=5.0)]
        ),
    )
    engine.step(1.0, "wait")
    lat, lon = engine._body_latlon()
    body_lat, body_lon, _alt = ecef_to_geodetic(eci_to_ecef(engine.orbit.r, engine._now()))
    assert lat == pytest.approx(body_lat)
    assert lon == pytest.approx(body_lon)
    assert abs(lat - (-40.0)) > 20.0


def test_failed_comms_blocks_downlink():
    epoch = Time("2024-03-01T00:00:00Z")
    r_ecef = geodetic_to_ecef(0.0, 0.0, 400_000.0)
    from space_env.core.ground import GroundStation

    engine = _engine_at(
        ecef_to_eci(r_ecef, epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    assert engine.get_state_snapshot().in_contact is True
    engine.spacecraft.state.health["comms"] = SubsystemHealth.FAILED
    engine.step(5.0, "comms")
    assert engine.last_command_ok is False
    assert engine.get_state_snapshot().comms_sessions == 0


def test_space_packet_roundtrip():
    payload = b"\x01\x02\x03\x04"
    packet = encode_space_packet(apid=42, seq_count=7, payload=payload, packet_type=1)
    header = decode_space_packet_header(packet)
    assert header["apid"] == 42
    assert header["sequence_count"] == 7
    assert header["packet_type"] == 1
    assert header["data_length"] == len(payload) - 1
    assert packet[6:] == payload
