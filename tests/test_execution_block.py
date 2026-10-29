"""Исполнение геометрии, скважности радара, погодной потери и цепочки DTN."""
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_eci, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet

EPOCH = Time("2024-01-01T00:00:00Z")


def _over_target(**fields) -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    axis = np.array([0.0, 0.0, 1.0])
    velocity = np.cross(radius, axis)
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(Orbit.from_vectors(radius, velocity, EPOCH), craft, scenario=ScenarioConfig(ground_stations=[]))
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(GroundTarget(name="here", lat_deg=lat, lon_deg=lon, **fields))
    return engine


def test_geometry_blocks_the_exposure_until_the_request_is_actually_met():
    point = _over_target()
    point.step(10.0, "imaging")
    assert point._imaging_count == 1
    assert point._exposure_reason == ""

    strip = _over_target(geometry="strip", length_km=5000.0)
    strip.step(10.0, "imaging")
    assert strip._imaging_count == 0
    assert strip._exposure_reason == "полоса длиннее кадра"

    polygon = _over_target(
        geometry="polygon",
        vertices=[
            {"lat_deg": 0.0, "lon_deg": 0.0},
            {"lat_deg": 20.0, "lon_deg": 0.0},
            {"lat_deg": 0.0, "lon_deg": 20.0},
        ],
    )
    polygon.step(10.0, "imaging")
    assert polygon._imaging_count == 0
    assert polygon._exposure_reason == "полигон шире кадра"

    waiting = _over_target(depends_on="parent")
    waiting.step(10.0, "imaging")
    assert waiting._imaging_count == 0
    assert waiting._exposure_reason == "ждёт предшественника"

    revisit = _over_target(geometry="revisit", looks=2, gap_sec=1.0)
    revisit.step(10.0, "imaging")
    assert revisit._imaging_count == 0
    assert revisit._exposure_reason == "мало повторных проходов"
    revisit.step(10.0, "imaging")
    assert revisit._imaging_count == 1
    assert revisit._exposure_reason == ""

    stereo = _over_target(geometry="stereo", min_stereo_deg=5.0, max_stereo_deg=30.0)
    stereo.step(10.0, "imaging")
    assert stereo._imaging_count == 0
    assert stereo._exposure_reason == "нет стереопары"
    stereo._looks["here"] = [(-10.0, 20.0)]
    stereo.step(10.0, "imaging")
    assert stereo._imaging_count == 1


def test_radar_duty_and_energy_stop_the_frame_before_it_is_stored():
    duty = _over_target()
    duty.spacecraft.config.payload.images_in_eclipse = True
    duty.spacecraft.config.payload.sar_duty_fraction = 1.0e-8
    duty.step(10.0, "imaging")
    assert duty._imaging_count == 0
    assert duty._exposure_reason == "скважность"
    assert duty._sar_used_sec == pytest.approx(0.0)

    allowed = _over_target()
    allowed.spacecraft.config.payload.images_in_eclipse = True
    allowed.spacecraft.config.payload.sar_duty_fraction = 1.0
    allowed.step(10.0, "imaging")
    assert allowed._imaging_count == 1
    assert allowed._sar_used_sec > 0.0

    hungry = _over_target()
    hungry.spacecraft.config.payload.images_in_eclipse = True
    hungry.spacecraft.config.payload.sar_duty_fraction = 1.0
    hungry.spacecraft.config.payload.imaging_power_w = 100_000.0
    hungry.spacecraft.config.power.battery_capacity_Wh = 10.0
    hungry.spacecraft.state.power.battery_soc = 0.2
    hungry.step(10.0, "imaging")
    assert hungry._imaging_count == 0
    assert hungry._exposure_reason == "энергия"
    assert hungry._sar_used_sec == pytest.approx(0.0)


def test_weather_loss_keeps_the_contact_and_drops_the_bits():
    station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=5.0)
    position = ecef_to_eci(geodetic_to_ecef(0.0, 0.0, 400_000.0), EPOCH)
    speed = np.sqrt(MU_EARTH / np.linalg.norm(position))
    axis = np.array([0.0, 0.0, 1.0])
    direction = np.cross(position, axis)
    direction = direction / np.linalg.norm(direction)

    def engine(loss: float) -> SimulationEngine:
        craft = Spacecraft(config=SpacecraftConfig())
        craft.reset_state()
        built = SimulationEngine(
            Orbit.from_vectors(position, direction * speed, EPOCH),
            craft,
            scenario=ScenarioConfig(ground_stations=[station]),
        )
        built.accept_bits(2.0e6, origin="imaging", request_id="frame")
        built.spacecraft.config.comms.weather_loss_db = loss
        return built

    clear = engine(0.0)
    clear.step(5.0, "comms")
    assert clear.delivered_bits > 0.0
    storm = engine(80.0)
    storm.step(5.0, "comms")
    assert storm.get_state_snapshot().in_contact is True
    assert storm.delivered_bits == pytest.approx(0.0)


def test_forward_stops_at_a_dead_edge_and_unacked_custody_comes_back_only_on_a_live_link():
    fleet = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.collect("SC-1", 5.0)
    arrived = fleet.forward("SC-1", ["SC-2", "SC-3"], 0.01)
    assert arrived["arrived"] is True
    assert arrived["holder"] == "SC-3"
    assert fleet.vehicles["SC-3"].stored_bits > 0.0

    blocked = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    blocked.collect("SC-1", 5.0)
    stopped = blocked.forward("SC-1", ["SC-2", "SC-3"], 0.01, down={frozenset({"SC-2", "SC-3"})})
    assert stopped["arrived"] is False
    assert stopped["holder"] == "SC-2"
    assert stopped["reason"] == "ребро выключено"
    assert blocked.vehicles["SC-3"].stored_bits == pytest.approx(0.0)

    custody = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    custody.vehicles["SC-1"].spacecraft.config.comms.ack_sec = 5.0
    custody.collect("SC-1", 1.0)
    assert custody.relay("SC-1", "SC-2", 0.01) > 0.0
    piece = next(item for item in custody.vehicles["SC-2"].products() if item["origin"] == "relay")
    assert piece["ack_from"] == "SC-1"
    assert piece["acked"] is False
    assert custody.return_unacked("SC-2", 0.01) == pytest.approx(0.0)
    assert custody.acknowledge("SC-2", piece["product_id"]) is True
    custody.vehicles["SC-2"]._time_sec = 30.0
    assert custody.return_unacked("SC-2", 0.01) == pytest.approx(0.0)
    assert custody.vehicles["SC-2"].stored_bits > 0.0

    late = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    late.vehicles["SC-1"].spacecraft.config.comms.ack_sec = 5.0
    before = late.vehicles["SC-1"].stored_bits
    late.collect("SC-1", 1.0)
    late.relay("SC-1", "SC-2", 0.01)
    holder = late.vehicles["SC-2"]
    holder._time_sec = 30.0
    assert late.return_unacked("SC-2", 0.01, down={frozenset({"SC-1", "SC-2"})}) == pytest.approx(0.0)
    assert holder.stored_bits > 0.0
    returned = late.return_unacked("SC-2", 0.01)
    assert returned > 0.0
    assert holder.stored_bits == pytest.approx(0.0)
    home = [item for item in late.vehicles["SC-1"].products() if item.get("origin") == "relay"]
    assert home
    assert "ack_from" not in home[0]
    assert late.vehicles["SC-1"].stored_bits > before
