"""Регрессии R1.2 и R1.3: один регистр бит, окно связи и один нагреватель."""
import numpy as np
import pytest

from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.sun import sun_position_m
from space_env.core.thermal import environmental_heat_w, step_temperature_c
from space_env.core.time import Time
from space_env.mission.events import EventKind, PlannedEvent
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from tests.test_dynamics import _engine_at


def _balance(engine: SimulationEngine) -> None:
    snap = engine.get_state_snapshot()
    assert snap.initial_bits + snap.produced_bits + snap.received_bits == pytest.approx(
        snap.stored_bits + snap.delivered_bits + snap.relayed_out_bits + snap.lost_bits
    )


def test_overflow_counts_the_discarded_product():
    epoch = Time("2026-10-20T00:00:00Z")
    engine = _engine_at(np.array([7.0e6, 0.0, 0.0]), epoch, scenario=ScenarioConfig(ground_stations=[]))
    engine.spacecraft.config.payload.memory_bits = 10.0
    engine.spacecraft.config.payload.image_bits = 8.0
    engine.spacecraft.config.payload.store_rate_bps = 0.0
    engine._store_capture("frame-a", 0.0, 1.0)
    engine._store_capture("frame-a", 0.0, 1.0)
    engine._store_capture("frame-b", 0.0, 1.0)
    snap = engine.get_state_snapshot()
    assert snap.produced_bits == pytest.approx(16.0)
    assert snap.stored_bits == pytest.approx(10.0)
    assert snap.lost_bits == pytest.approx(6.0)
    assert snap.imaging_count == 2
    assert engine._products[0]["quality"] == "full"
    assert engine._products[1]["quality"] == "partial"
    _balance(engine)


def test_streaming_duration_matches_six_short_steps():
    epoch = Time("2026-10-20T12:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    tangent = np.cross(np.array([0.0, 0.0, 1.0]), position)
    velocity = tangent / np.linalg.norm(tangent) * 7500.0

    def make() -> SimulationEngine:
        craft_orbit_engine = _engine_at(position, epoch, scenario=ScenarioConfig(ground_stations=[]))
        craft_orbit_engine.orbit = craft_orbit_engine.orbit.with_state(position, velocity, epoch)
        craft_orbit_engine.spacecraft.config.payload.store_rate_bps = 1000.0
        craft_orbit_engine.spacecraft.config.payload.memory_bits = 0.0
        lat, lon = __import__("space_env.core.access", fromlist=["subsatellite_latlon"]).subsatellite_latlon(
            position, epoch
        )
        craft_orbit_engine.scenario.ground_targets.append(GroundTarget(name="scene", lat_deg=lat, lon_deg=lon))
        return craft_orbit_engine

    coarse = make()
    coarse.step(60.0, "imaging")
    fine = make()
    for _ in range(6):
        fine.step(10.0, "imaging")
    assert coarse.get_state_snapshot().produced_bits == pytest.approx(fine.get_state_snapshot().produced_bits, rel=0.02)
    assert coarse.get_state_snapshot().produced_bits > 1000.0 * 30.0
    assert fine.get_state_snapshot().imaging_count == 1
    _balance(coarse)
    _balance(fine)


def test_imaging_energy_follows_the_window_not_the_step():
    epoch = Time("2026-10-20T12:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    from space_env.mission.events import EventKind, PlannedEvent

    def make() -> SimulationEngine:
        engine = _engine_at(
            position,
            epoch,
            scenario=ScenarioConfig(
                ground_stations=[],
                planned_events=[PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=30.0, duration_sec=10.0)],
            ),
        )
        engine.spacecraft.config.payload.imaging_power_w = 800.0
        return engine

    coarse = make()
    coarse.step(60.0, "wait")
    fine = make()
    for _ in range(6):
        fine.step(10.0, "wait")
    assert coarse._payload_energy_j == pytest.approx(800.0 * 10.0)
    assert fine._payload_energy_j == pytest.approx(coarse._payload_energy_j)


def test_receipt_names_the_request_and_the_station():
    epoch = Time("2026-10-20T12:00:00Z")
    engine = _engine_at(
        ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    engine.accept_bits(2.0e6, origin="imaging", request_id="request-X")
    engine._in_contact = True
    engine._link_margin_db = 10.0
    engine._station_name = "Equator"
    moved = engine.transmit_downlink(1.0)
    report = engine.request_report("request-X", deadline_sec=10.0, destination="Equator")
    assert moved > 0.0
    assert report["receipts"][0]["request_id"] == "request-X"
    assert report["receipts"][0]["destination"] == "Equator"
    assert report["delivered_bits"] == pytest.approx(moved * 1.0e6)
    assert report["on_time"] is True


def test_cancel_stops_the_rest_of_the_imaging_window():
    epoch = Time("2026-10-20T12:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    from space_env.core.access import subsatellite_latlon
    from space_env.mission.events import EventKind, PlannedEvent

    lat, lon = subsatellite_latlon(position, epoch)
    engine = _engine_at(
        position,
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[],
            ground_targets=[GroundTarget(name="scene", lat_deg=lat, lon_deg=lon)],
            planned_events=[PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=0.0, duration_sec=30.0)],
        ),
    )
    engine.spacecraft.config.payload.store_rate_bps = 1000.0
    engine.spacecraft.config.payload.half_fov_deg = 70.0
    engine.step(5.0, "wait")
    engine.cancel_imaging()
    engine.step(10.0, "wait")
    assert engine.produced_bits == pytest.approx(5000.0, rel=0.05)


def test_planned_imaging_window_does_not_depend_on_the_step_size():
    epoch = Time("2026-10-20T12:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    from space_env.core.access import subsatellite_latlon
    from space_env.mission.events import EventKind, PlannedEvent

    lat, lon = subsatellite_latlon(position, epoch)

    def make(planned: bool) -> SimulationEngine:
        scene = ScenarioConfig(
            ground_stations=[],
            ground_targets=[GroundTarget(name="scene", lat_deg=lat, lon_deg=lon)],
            planned_events=[
                PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=30.0, duration_sec=10.0)
            ]
            if planned
            else [],
        )
        engine = _engine_at(position, epoch, scenario=scene)
        engine.spacecraft.config.payload.store_rate_bps = 1000.0
        engine.spacecraft.config.payload.memory_bits = 0.0
        return engine

    manual = make(False)
    manual.step(30.0, "wait")
    manual.step(10.0, "imaging")
    manual.step(20.0, "wait")
    planned = make(True)
    snap, events = planned.step(60.0, "wait")
    short = make(True)
    for _ in range(6):
        short.step(10.0, "wait")
    assert manual.get_state_snapshot().produced_bits == pytest.approx(10_000.0, rel=0.02)
    assert snap.produced_bits == pytest.approx(manual.get_state_snapshot().produced_bits, rel=0.02)
    assert short.get_state_snapshot().produced_bits == pytest.approx(manual.get_state_snapshot().produced_bits, rel=0.02)
    assert [event.time_sec for event in events if event.kind == EventKind.IMAGING] == pytest.approx([30.0], abs=1.0)


def test_terminal_time_is_not_spent_twice_on_the_same_clock():
    epoch = Time("2026-10-20T12:00:00Z")
    engine = _engine_at(
        ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    engine.preload_bits(1.0e9)
    engine._in_contact = True
    engine._link_margin_db = 10.0
    first = engine.transmit_downlink(5.0)
    second = engine.transmit_downlink(10.0)
    assert first == pytest.approx(5.0)
    assert second == pytest.approx(5.0)
    assert engine.time_sec == pytest.approx(0.0)
    assert engine.delivered_bits == pytest.approx(10.0e6)


def test_relay_keeps_the_request_identity():
    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.vehicles["SC-1"].accept_bits(5.0e6, origin="imaging", request_id="request-X")
    moved = fleet.relay("SC-1", "SC-2", 60.0)
    assert moved > 0.0
    received = fleet.vehicles["SC-2"]._products[-1]
    assert received["request_id"] == "request-X"
    assert received["product_id"] == fleet.vehicles["SC-1"]._products[0]["product_id"]


def test_downlink_stops_at_the_end_of_contact_and_a_repeat_adds_nothing():
    epoch = Time("2024-03-01T00:00:00Z")
    engine = _engine_at(
        ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), epoch),
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)]
        ),
    )
    engine.preload_bits(5.0e6)
    engine.step(5.0, "comms")
    once = engine.get_state_snapshot().delivered_bits
    assert once > 0.0
    again = engine.transmit_downlink(5.0)
    assert again == 0.0
    assert engine.get_state_snapshot().delivered_bits == pytest.approx(once)
    _balance(engine)


def test_relay_updates_the_receiver_engine_and_does_not_repeat():
    fleet = Fleet.string_of_pearls(2, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.collect("SC-1", 5.0)
    moved = fleet.relay("SC-1", "SC-2", 60.0)
    assert moved > 0.0
    assert fleet.vehicles["SC-2"].stored_bits == pytest.approx(moved * 1.0e6)
    assert fleet.buffer_mb["SC-2"] == pytest.approx(moved)
    assert fleet.relay("SC-1", "SC-2", 60.0) == 0.0
    _balance(fleet.vehicles["SC-1"])
    _balance(fleet.vehicles["SC-2"])


def test_planned_event_at_zero_fires_once_at_zero():
    epoch = Time("2026-06-21T00:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    engine = _engine_at(
        position,
        epoch,
        scenario=ScenarioConfig(
            ground_stations=[],
            planned_events=[PlannedEvent(kind=EventKind.IMAGING, scheduled_at_sec=0.0, duration_sec=0.0)],
        ),
    )
    from space_env.core.access import subsatellite_latlon

    lat, lon = subsatellite_latlon(position, epoch)
    engine.scenario.ground_targets.append(GroundTarget(name="now", lat_deg=lat, lon_deg=lon))
    _snap, events = engine.step(10.0)
    imaging = [event for event in events if event.kind == EventKind.IMAGING]
    assert len(imaging) == 1
    assert imaging[0].time_sec == pytest.approx(0.0, abs=1.0)
    _snap, later = engine.step(10.0)
    assert all(event.kind != EventKind.IMAGING for event in later)


def test_heater_enters_the_thermal_node_once():
    epoch = Time("2026-10-20T00:00:00Z")
    sun = sun_position_m(epoch)
    position = sun / np.linalg.norm(sun) * 7.0e6
    engine = _engine_at(position, epoch, scenario=ScenarioConfig(ground_stations=[]))
    thermal = engine.spacecraft.config.thermal
    engine.spacecraft.config.thermal.heat_capacity_J_K = 500.0
    engine.spacecraft.state.thermal.temp_C = -30.0
    before = engine.spacecraft.state.thermal.temp_C
    engine.step(1.0, "wait")
    flux = engine._flux
    q_solar, q_albedo, q_ir = environmental_heat_w(
        r_sat_m=engine.orbit.r,
        r_sun_m=sun_position_m(engine.now()),
        solar_flux_w_m2=flux,
        absorptivity=thermal.absorptivity,
        emissivity=thermal.emissivity,
        solar_area_m2=thermal.projected_area_m2,
        earth_area_m2=thermal.earth_facing_area_m2,
    )
    once = step_temperature_c(
        before,
        1.0,
        q_sun_w=q_solar + q_albedo + q_ir,
        q_diss_w=engine.spacecraft.config.power.consumption_nominal_W,
        q_heater_w=thermal.heater_power_W,
        radiating_area_m2=thermal.radiating_area_m2,
        emissivity=thermal.emissivity,
        heat_capacity_j_k=thermal.heat_capacity_J_K,
    )
    twice = step_temperature_c(
        before,
        1.0,
        q_sun_w=q_solar + q_albedo + q_ir,
        q_diss_w=engine.spacecraft.config.power.consumption_nominal_W + thermal.heater_power_W,
        q_heater_w=thermal.heater_power_W,
        radiating_area_m2=thermal.radiating_area_m2,
        emissivity=thermal.emissivity,
        heat_capacity_j_k=thermal.heat_capacity_J_K,
    )
    assert engine.spacecraft.state.thermal.heater_on is True
    assert abs(engine.spacecraft.state.thermal.temp_C - once) < abs(engine.spacecraft.state.thermal.temp_C - twice)
