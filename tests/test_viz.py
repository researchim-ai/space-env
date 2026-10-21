"""Экспорт траектории для просмотрщика."""
import numpy as np

from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet
from space_env.viz.export import export_trajectory


def test_export_trajectory_has_two_vehicles_and_finite_positions():
    fleet = Fleet.string_of_pearls(2, separation_deg=15.0, scenario=ScenarioConfig(ground_stations=[]))
    document = export_trajectory(fleet, steps=2, dt_sec=30.0)
    assert document["version"] == 1
    assert len(document["frames"]) == 3
    assert document["frames"][1]["t"] == 30.0
    for frame in document["frames"]:
        assert set(frame["vehicles"]) == {"SC-1", "SC-2"}
        assert len(frame["sun"]) == 3
        assert len(frame["pairs"]) == 1
        assert "earth_blocked" in frame["pairs"][0]
        assert "buffer_mb" in frame["vehicles"]["SC-1"]
        assert frame["route"] == []
        for vehicle in frame["vehicles"].values():
            assert np.isfinite(vehicle["r"]).all()
            assert np.linalg.norm(vehicle["r"]) > document["earth_radius_m"]


def test_export_keeps_route_and_memory_fill():
    fleet = Fleet.string_of_pearls(2, separation_deg=1.5, scenario=ScenarioConfig(ground_stations=[]))
    document = export_trajectory(
        fleet,
        steps=0,
        dt_sec=30.0,
        route=[{"from": "SC-1", "to": "SC-2"}],
        memory_bits={"SC-1": 2.7e10},
        memory_capacity_bits=2.4e12,
    )
    frame = document["frames"][0]
    assert frame["route"] == [{"from": "SC-1", "to": "SC-2"}]
    assert frame["vehicles"]["SC-1"]["memory_bits"] == 2.7e10
    assert document["memory_capacity_bits"] == 2.4e12
