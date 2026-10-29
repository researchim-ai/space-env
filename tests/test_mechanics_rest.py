"""Интервалы терминала, снимок целей, полоса кадра и маршрут по живой линии."""
import numpy as np

from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.sun import sun_position_m
from space_env.core.time import Time
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet
from space_env.simulation.terminals import assign_terminal_intervals
from space_env.simulation.world import MissionWorld

EPOCH = Time("2024-01-01T00:00:00Z")


def test_one_terminal_keeps_the_overlap_and_releases_the_free_tail():
    claims = [
        ("Equator", "a", 0.0, 8.0),
        ("Equator", "b", 4.0, 10.0),
    ]
    allowed = assign_terminal_intervals(claims, {"Equator": 1})
    assert allowed["a"] == [(0.0, 8.0)]
    assert allowed["b"] == [(8.0, 10.0)]
    apart = assign_terminal_intervals(
        [("Equator", "a", 0.0, 3.0), ("Equator", "b", 5.0, 9.0)],
        {"Equator": 1},
    )
    assert apart["a"] == [(0.0, 3.0)]
    assert apart["b"] == [(5.0, 9.0)]
    both = assign_terminal_intervals(claims, {"Equator": 2})
    assert both["a"] == [(0.0, 8.0)]
    assert both["b"] == [(4.0, 10.0)]


def test_snapshot_restores_targets_without_the_scenario_file():
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    velocity = np.cross(radius, np.array([0.0, 0.0, 1.0]))
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(
        GroundTarget(name="strip", lat_deg=lat, lon_deg=lon, geometry="strip", length_km=10.0)
    )
    world = MissionWorld({"a": engine}, epoch="2024-01-01T00:00:00Z", horizon_sec=30.0)
    saved = world.snapshot()
    engine.scenario.ground_targets.clear()
    world.restore(saved)
    restored = world.engines["a"].scenario.ground_targets
    assert len(restored) == 1
    assert restored[0].name == "strip"
    assert restored[0].geometry == "strip"
    assert restored[0].length_km == 10.0


def _optical(length_km: float, geometry: str = "strip") -> SimulationEngine:
    sun = sun_position_m(EPOCH)
    radius = sun / np.linalg.norm(sun) * 7.0e6
    speed = np.sqrt(MU_EARTH / np.linalg.norm(radius))
    velocity = np.cross(radius, np.array([0.0, 0.0, 1.0]))
    velocity = velocity / np.linalg.norm(velocity) * speed
    craft = Spacecraft(config=SpacecraftConfig())
    craft.reset_state()
    engine = SimulationEngine(
        Orbit.from_vectors(radius, velocity, EPOCH),
        craft,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
    engine.scenario.ground_targets.append(
        GroundTarget(name="here", lat_deg=lat, lon_deg=lon, geometry=geometry, length_km=length_km)
    )
    return engine


def test_a_short_strip_records_its_length_and_a_long_one_does_not():
    short = _optical(10.0)
    short.step(10.0, "imaging")
    assert short._imaging_count == 1
    assert short._products[0]["geometry"] == "strip"
    assert short._products[0]["coverage_m"] == 10_000.0
    long = _optical(5000.0)
    long.step(10.0, "imaging")
    assert long._imaging_count == 0
    polygon = _optical(0.0, geometry="polygon")
    center = polygon.scenario.ground_targets[0]
    polygon.scenario.ground_targets[0] = GroundTarget(
        name="here",
        lat_deg=center.lat_deg,
        lon_deg=center.lon_deg,
        geometry="polygon",
        vertices=[
            {"lat_deg": center.lat_deg, "lon_deg": center.lon_deg},
            {"lat_deg": center.lat_deg + 0.01, "lon_deg": center.lon_deg},
            {"lat_deg": center.lat_deg, "lon_deg": center.lon_deg + 0.01},
        ],
    )
    polygon.step(10.0, "imaging")
    assert polygon._imaging_count == 1
    assert polygon._products[0]["geometry"] == "polygon"
    assert polygon._products[0]["vertices"] == 3


def test_route_live_skips_a_dead_edge_and_advances_the_clock():
    fleet = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.collect("SC-1", 5.0)
    fleet.collect("SC-2", 1.0)
    fleet.collect("SC-3", 0.1)
    assert fleet.route("SC-2") == "SC-3"
    blocked = fleet.route_live("SC-2", 30.0, down={frozenset({"SC-2", "SC-3"})})
    assert blocked["arrived"] is True
    assert blocked["path"] == ["SC-1"]
    assert blocked["time_sec"] == 30.0
