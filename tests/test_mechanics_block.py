"""Знание орбиты, цена уклонения и компенсация торможения."""
import numpy as np
import pytest

from space_env.core.access import subsatellite_latlon
from space_env.core.lifetime import drag_makeup_m_s
from space_env.core.orbit import Orbit
from space_env.core.propagate import propagate_state
from space_env.core.time import Time
from space_env.estimation.knowledge import knowledge_claim, range_shift_m
from space_env.mission.avoidance import avoidance_plan


def test_one_station_is_blind_across_track_and_is_not_an_operational_fix():
    claim = knowledge_claim(1, 180.0)
    assert claim["operational_od"] is False
    assert claim["cross_track"] is False
    longer = knowledge_claim(2, 180.0)
    assert longer["operational_od"] is False
    assert longer["cross_track"] is False
    assert knowledge_claim(2, 3600.0)["operational_od"] is False
    epoch = Time("2024-06-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    lat, lon = subsatellite_latlon(orbit.r, epoch)
    position, velocity, _samples = propagate_state(
        orbit.r,
        orbit.v,
        90.0,
        mass_kg=100.0,
        drag_area_m2=0.0,
        drag_coefficient=0.0,
        include_drag=False,
        include_j2=False,
        substep_sec=10.0,
    )
    along, cross = range_shift_m(position, velocity, epoch + 90.0, lat, lon)
    assert along > cross * 5.0
    assert cross < 50.0


def test_avoidance_spends_the_mission_budget_or_leaves_the_miss_alone():
    refused = avoidance_plan(
        100.0,
        hard_body_radius_m=10.0,
        sigma_m=50.0,
        dv_budget_m_s=1.0,
        mission_reserve_m_s=0.8,
        target_miss_m=1_000.0,
        dv_per_m=0.001,
    )
    assert refused["accepted"] is False
    assert refused["reason"] == "бюджет миссии"
    assert refused["miss_m"] == pytest.approx(100.0)
    assert refused["pc_after"] == pytest.approx(refused["pc_before"])
    accepted = avoidance_plan(
        100.0,
        hard_body_radius_m=10.0,
        sigma_m=50.0,
        dv_budget_m_s=2.0,
        mission_reserve_m_s=0.2,
        target_miss_m=1_000.0,
        dv_per_m=0.001,
    )
    assert accepted["accepted"] is True
    assert accepted["dv_m_s"] == pytest.approx(0.9)
    assert accepted["pc_after"] < accepted["pc_before"]
    assert accepted["mission_reserve_m_s"] == pytest.approx(0.2)
    safe = avoidance_plan(
        2_000.0,
        hard_body_radius_m=10.0,
        sigma_m=50.0,
        dv_budget_m_s=2.0,
        mission_reserve_m_s=0.2,
        target_miss_m=1_000.0,
        dv_per_m=0.001,
    )
    assert safe["reason"] == "промах уже достаточен"
    assert safe["dv_m_s"] == pytest.approx(0.0)


def test_drag_makeup_costs_more_on_a_low_light_craft_than_the_budget_allows():
    low = drag_makeup_m_s(200_000.0, mass_kg=10.0, area_m2=1.0, drag_coefficient=2.2, horizon_sec=86_400.0)
    high = drag_makeup_m_s(800_000.0, mass_kg=1_000.0, area_m2=0.01, drag_coefficient=2.2, horizon_sec=86_400.0)
    assert low > high * 10.0
    assert low > 1.0
    assert high < 1.0
