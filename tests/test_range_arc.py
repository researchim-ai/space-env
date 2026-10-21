"""Сдвиг вдоль трека восстанавливается по геометрическим дальностям той же дуги."""
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.frames import elevation_deg
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.time import Time
from space_env.estimation.range_arc import fit_along_track_m, geometric_range_m


def test_along_track_offset_is_recovered_from_ranges():
    epoch = Time("2024-06-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    lat, lon = subsatellite_latlon(orbit.r, epoch)
    truth_m = 2_000.0
    speed = float(np.linalg.norm(orbit.v))
    r_truth = orbit.r + truth_m * orbit.v / speed
    ranges = []
    position, velocity = r_truth.copy(), orbit.v.copy()
    elapsed = 0.0
    while elapsed <= 360.0 + 1e-9:
        moment = epoch + elapsed
        if elevation_deg(position, lat, lon, 0.0, moment) >= 10.0:
            ranges.append((elapsed, geometric_range_m(position, moment, lat, lon)))
        if elapsed >= 360.0:
            break
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            30.0,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_drag=False,
            include_j2=True,
            include_third_body=False,
            substep_sec=10.0,
        )
        elapsed += 30.0
    assert len(ranges) >= 4
    solution = fit_along_track_m(orbit.r, orbit.v, epoch, ranges, lat, lon)
    assert abs(solution.along_track_m - truth_m) < 25.0
    assert solution.rms_m < 25.0


def test_radius_and_along_track_rate_come_back_with_a_blind_cross_track():
    from space_env.estimation.range_arc import fit_rtn_arc
    from space_env.mission.scenarios import ScenarioConfig
    from space_env.simulation.fleet import Fleet

    epoch = Time("2024-06-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    lat, lon = subsatellite_latlon(orbit.r, epoch)
    radial_m, along_m, rate_m_s = 400.0, 1_200.0, 0.15
    radius = float(np.linalg.norm(orbit.r))
    speed = float(np.linalg.norm(orbit.v))
    position = orbit.r + radial_m * orbit.r / radius + along_m * orbit.v / speed
    velocity = orbit.v + rate_m_s * orbit.v / speed
    ranges = []
    elapsed = 0.0
    while elapsed <= 240.0 + 1e-9:
        moment = epoch + elapsed
        if elevation_deg(position, lat, lon, 0.0, moment) >= 10.0:
            ranges.append((elapsed, geometric_range_m(position, moment, lat, lon)))
        if elapsed >= 240.0:
            break
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            40.0,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_drag=False,
            include_j2=True,
            include_third_body=False,
            substep_sec=10.0,
        )
        elapsed += 40.0
    solution = fit_rtn_arc(orbit.r, orbit.v, epoch, ranges, lat, lon)
    assert solution.radial_m == pytest.approx(radial_m, abs=80.0)
    assert solution.along_m == pytest.approx(along_m, abs=80.0)
    assert solution.along_rate_m_s == pytest.approx(rate_m_s, abs=0.05)
    assert solution.covariance_rtn_m2[2, 2] > solution.covariance_rtn_m2[1, 1] * 10.0
    fleet = Fleet.string_of_pearls(2, separation_deg=0.02, scenario=ScenarioConfig(ground_stations=[]))
    fleet.use_arc_covariance("SC-1", "SC-2", solution.covariance_rtn_m2)
    stored = fleet.pair_covariance("SC-1", "SC-2")
    assert stored[1, 1] == pytest.approx(solution.covariance_rtn_m2[1, 1])
    assert fleet.screen()[0].probability >= 0.0


def test_second_station_recovers_cross_track_with_range_noise():
    from space_env.estimation.range_arc import fit_two_station_arc

    epoch = Time("2024-06-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    lat, lon = subsatellite_latlon(orbit.r, epoch)
    stations = ((lat, lon), (lat + 12.0, lon))
    truth = np.array([250.0, 600.0, 500.0, 0.05])
    from space_env.core.relative import rtn_basis

    basis = rtn_basis(orbit.r, orbit.v)
    position = orbit.r + basis @ np.array([truth[0], truth[1], truth[2]])
    velocity = orbit.v + basis @ np.array([0.0, truth[3], 0.0])
    rng = np.random.default_rng(1)
    ranges = []
    elapsed = 0.0
    while elapsed <= 180.0 + 1e-9:
        moment = epoch + elapsed
        for station_lat, station_lon in stations:
            if elevation_deg(position, station_lat, station_lon, 0.0, moment) < 5.0:
                continue
            ranges.append(
                (
                    elapsed,
                    geometric_range_m(position, moment, station_lat, station_lon) + float(rng.normal(0.0, 20.0)),
                    station_lat,
                    station_lon,
                )
            )
        if elapsed >= 180.0:
            break
        position, velocity, _samples = propagate_state(
            position,
            velocity,
            30.0,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_drag=False,
            include_j2=True,
            include_third_body=False,
            substep_sec=10.0,
        )
        elapsed += 30.0
    solution = fit_two_station_arc(orbit.r, orbit.v, epoch, ranges, range_sigma_m=20.0)
    assert solution.cross_m == pytest.approx(500.0, abs=150.0)
    assert solution.radial_m == pytest.approx(250.0, abs=150.0)
    assert solution.covariance_rtn_m2[2, 2] < 5_000.0**2
    assert solution.covariance_rtn_m2[3, 3] == pytest.approx(10.0**2)
