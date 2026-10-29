"""Три пункта плана: геометрия заявки, операционный SAR и следующий хоп DTN."""
import math
from pathlib import Path

import pytest

from space_env.mission.geometry_requests import (
    classify_request,
    dependency_open,
    polygon_fits,
    revisit_possible,
    stereo_possible,
    strip_fits,
)
from space_env.mission.sar_ops import SarDuty, incidence_deg, sar_look
from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.simulation.dtn import order_for_forward, product_to_send
from space_env.simulation.fleet import Fleet
from space_env.mission.scenarios import ScenarioConfig

_NOMINAL = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "family_nominal.yaml"


def test_point_strip_polygon_revisit_stereo_and_dependency():
    assert strip_fits(100.0, 200.0) is True
    assert strip_fits(500.0, 200.0) is False
    assert polygon_fits(0.0, 0.0, [(0.01, 0.0), (-0.01, 0.0), (0.0, 0.01)], 10_000.0) is True
    assert polygon_fits(0.0, 0.0, [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], 10_000.0) is False
    assert revisit_possible([0.0, 100.0, 400.0], looks=2, gap_sec=300.0) is True
    assert revisit_possible([0.0, 100.0], looks=2, gap_sec=300.0) is False
    assert stereo_possible([2.0, 15.0], 5.0, 20.0) is True
    assert stereo_possible([2.0, 4.0], 5.0, 20.0) is False
    assert dependency_open("parent", {"parent"}) is True
    assert dependency_open("parent", set()) is False
    blocked = classify_request(
        {"geometry": "polygon", "lat_deg": 0.0, "lon_deg": 0.0, "vertices": [
            {"lat_deg": 0.0, "lon_deg": 0.0},
            {"lat_deg": 10.0, "lon_deg": 0.0},
            {"lat_deg": 0.0, "lon_deg": 10.0},
        ], "depends_on": ""},
        footprint_diameter_m=10_000.0,
        hit_times_sec=[0.0],
        off_nadir_deg=[0.0],
        done_ids=set(),
    )
    assert blocked == {"reachable": False, "reason": "полигон шире кадра"}
    waiting = classify_request(
        {"geometry": "point", "lat_deg": 0.0, "lon_deg": 0.0, "depends_on": "parent"},
        footprint_diameter_m=10_000.0,
        hit_times_sec=[0.0],
        off_nadir_deg=[0.0],
        done_ids=set(),
    )
    assert waiting["reason"] == "ждёт предшественника"
    loaded = load_scenario_v2(_NOMINAL)
    assert loaded["requests"][0]["geometry"] == "point"
    assert loaded["requests"][0]["looks"] == 1


def test_sar_look_stops_on_horizon_duty_and_energy():
    assert incidence_deg(90.0) == pytest.approx(0.0)
    assert incidence_deg(30.0) == pytest.approx(60.0)
    assert incidence_deg(-1.0) is None
    duty = SarDuty(orbit_sec=100.0, duty_fraction=0.1)
    dark = sar_look(
        elevation_deg=-5.0, altitude_m=500_000.0, beam_half_deg=5.0,
        power_w=1000.0, seconds=5.0, soc=1.0, capacity_wh=100.0, duty=duty,
    )
    assert dark["ok"] is False and dark["reason"] == "ниже горизонта"
    first = sar_look(
        elevation_deg=60.0, altitude_m=500_000.0, beam_half_deg=5.0,
        power_w=1000.0, seconds=5.0, soc=1.0, capacity_wh=100.0, duty=duty,
    )
    assert first["ok"] is True
    assert first["incidence_deg"] == pytest.approx(30.0)
    assert first["swath_m"] > 0.0
    second = sar_look(
        elevation_deg=60.0, altitude_m=500_000.0, beam_half_deg=5.0,
        power_w=1000.0, seconds=6.0, soc=1.0, capacity_wh=100.0, duty=duty,
    )
    assert second["reason"] == "скважность"
    assert duty.used_sec == pytest.approx(5.0)
    hungry = SarDuty(orbit_sec=1000.0, duty_fraction=1.0)
    starved = sar_look(
        elevation_deg=60.0, altitude_m=500_000.0, beam_half_deg=5.0,
        power_w=3600.0, seconds=10.0, soc=0.01, capacity_wh=10.0, duty=hungry,
    )
    assert starved["reason"] == "энергия"
    assert hungry.used_sec == pytest.approx(0.0)
    assert math.isfinite(first["swath_m"])


def test_dtn_changes_hop_when_the_link_drops_and_sends_priority_first():
    fleet = Fleet.string_of_pearls(3, separation_deg=1.0, scenario=ScenarioConfig(ground_stations=[]))
    fleet.collect("SC-1", 5.0)
    fleet.collect("SC-3", 0.1)
    assert fleet.route("SC-2") == "SC-3"
    assert fleet.route("SC-2", down={frozenset({"SC-2", "SC-3"})}) == "SC-1"
    assert fleet.route("SC-2", down={frozenset({"SC-2", "SC-3"}), frozenset({"SC-1", "SC-2"})}) is None
    source = fleet.vehicles["SC-2"]
    source.accept_bits(1.0e6, origin="collect", request_id="low")
    source.accept_bits(1.0e6, origin="collect", request_id="high")
    source._products[0]["priority"] = 1
    source._products[1]["priority"] = 9
    chosen = product_to_send(source.products())
    assert chosen["request_id"] == "high"
    source._products = order_for_forward(source._products)
    moved = fleet.relay("SC-2", "SC-3", 0.01)
    assert moved > 0.0
    arrived = [item for item in fleet.vehicles["SC-3"].products() if item["request_id"] == "high"]
    assert arrived
    assert arrived[0]["origin"] == "relay"
    again = fleet.relay("SC-2", "SC-3", 0.01)
    assert again == pytest.approx(0.0)
