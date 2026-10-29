"""Спрос вне трассы: точка, длинная полоса и уровень сервиса."""
from pathlib import Path

import pytest
import yaml

from space_env.mission.demand import independent_demand
from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.mission.tasking import service_level

_NOMINAL = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "family_nominal.yaml"


def test_a_long_strip_on_the_track_is_still_one_frame_too_long():
    demand = independent_demand(1)
    strip = next(item for item in demand["issued"] if item["id"] == "strip-long")
    assert strip["length_km"] * 1000.0 > demand["footprint_diameter_m"]
    assert strip["reachable"] is False
    assert strip["reason"] == "полоса длиннее кадра"
    assert demand["unreachable"] >= 1
    assert demand["reachable"] >= 1


def test_service_level_keeps_unreachable_demand_in_the_denominator():
    issued = [
        {"id": "a", "priority": 1, "reachable": True},
        {"id": "b", "priority": 1, "reachable": True},
        {"id": "c", "priority": 1, "reachable": False},
    ]
    level = service_level(issued, {"a"})
    assert level["reachable"] == 2
    assert level["unreachable"] == 1
    assert level["on_time_of_reachable"] == pytest.approx(0.5)
    assert level["on_time_of_issued"] == pytest.approx(1.0 / 3.0)


def test_old_scenario_stays_a_point_and_a_strip_needs_a_length(tmp_path):
    loaded = load_scenario_v2(_NOMINAL)
    assert loaded["requests"][0]["geometry"] == "point"
    assert loaded["requests"][0]["length_km"] == 0.0
    broken = tmp_path / "strip.yaml"
    text = _NOMINAL.read_text(encoding="utf-8")
    broken.write_text(text + "    geometry: strip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="длина"):
        load_scenario_v2(broken)
    whole = yaml.safe_load(_NOMINAL.read_text(encoding="utf-8"))
    whole["requests"][0]["geometry"] = "strip"
    whole["requests"][0]["length_km"] = 12.0
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(whole), encoding="utf-8")
    parsed = load_scenario_v2(path)
    assert parsed["requests"][0]["geometry"] == "strip"
    assert parsed["requests"][0]["length_km"] == pytest.approx(12.0)
