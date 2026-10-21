"""Сутки трёх аппаратов: заявки, сеансы и один бюджет Δv."""
import gymnasium as gym
import pytest

from space_env.mission.campaign import DAY_SEC, _share_stations, build_day_campaign, orbital_period_s
from space_env.mission.workplan import WorkItem, WorkPlan, build_acquisition_plan
from space_env.rl.expert import expert_action


def _passes_do_not_overlap(plans) -> None:
    rows = []
    for plan in plans:
        for item in plan.required_items():
            if item.kind == "comms":
                rows.append((plan.vehicle, item))
    for index, (left_vehicle, left) in enumerate(rows):
        for right_vehicle, right in rows[index + 1 :]:
            if left_vehicle == right_vehicle or left.name != right.name:
                continue
            assert left.end_sec <= right.start_sec or right.end_sec <= left.start_sec


def test_day_campaign_schedules_three_buses_under_one_budget():
    campaign = build_day_campaign()
    assert campaign.horizon_sec == DAY_SEC
    assert campaign.vehicles == ("sentinel-2a", "landsat-9", "cubesat-6u")
    assert campaign.dv_spent_m_s == 1.0
    assert campaign.dv_spent_m_s <= campaign.dv_budget_m_s
    hot = next(item for item in campaign.conjunctions if item.name == "HOT")
    warm = next(item for item in campaign.conjunctions if item.name == "WARM")
    debris = next(item for item in campaign.conjunctions if item.name == "debris-fragment")
    assert hot.required and hot.dv_m_s == 1.0 and hot.direction == "normal"
    assert hot.start_sec == pytest.approx(3_600.0, abs=120.0)
    assert warm.start_sec == pytest.approx(21_600.0, abs=120.0)
    assert debris.start_sec == pytest.approx(43_200.0, abs=120.0)
    assert debris.required is False and debris.probability < 1.0e-4
    sentinel = campaign.plans[0]
    assert len(sentinel.targets) >= 10
    assert any(item.required and item.kind == "imaging" for item in sentinel.items)
    assert any(item.required and item.kind == "comms" for item in sentinel.items)
    landsat = campaign.plans[1]
    cubesat = campaign.plans[2]
    assert any(item.required and item.kind == "comms" for item in landsat.items)
    assert len(cubesat.targets) >= 10
    assert any(item.required and item.kind == "imaging" for item in cubesat.items)
    assert any(item.kind == "comms" and item.required is False and "сеанса" in item.reason for item in cubesat.items)
    above = [item for item in campaign.conjunctions if item.probability >= 1.0e-4]
    assert len(above) >= 2
    assert sum(item.required for item in above) == 1
    coast = dict(campaign.coast_pc)
    flown = dict(campaign.flown_pc)
    assert campaign.burn_at_sec == pytest.approx(2_100.0, abs=60.0)
    assert coast["HOT"] > 1.0e-4 and flown["HOT"] < 1.0e-4
    assert coast["WARM"] > 1.0e-4 and flown["WARM"] < 1.0e-4
    assert flown["debris-fragment"] < 1.0e-4
    for name in ("sentinel-2b", "landsat-9", "cubesat-6u", "debris-class"):
        other = next(item for item in campaign.conjunctions if item.name == name)
        assert other.required is False and other.probability < 1.0e-4
        assert flown[name] < 1.0e-4
    coast_plan = build_acquisition_plan(
        "sentinel-2a",
        horizon_sec=DAY_SEC,
        epoch="2026-10-20T00:00:00Z",
        nu_deg=180.0,
        order_every_sec=orbital_period_s("sentinel-2a"),
    )
    assert campaign.plans[0].targets != coast_plan.targets
    _passes_do_not_overlap(campaign.plans)


def test_empty_budget_burns_nobody():
    campaign = build_day_campaign(dv_budget_m_s=0.0)
    assert campaign.dv_spent_m_s == 0.0
    assert all(item.required is False for item in campaign.conjunctions)
    assert len(campaign.plans[0].targets) >= 10


def test_day_mission_expert_accepts_the_plan():
    env = gym.make("space_env/DayMission-v0")
    obs, info = env.reset(seed=0)
    assert info["lines"]
    _obs, _reward, terminated, _truncated, step = env.step(expert_action("space_env/DayMission-v0", obs))
    assert terminated and step["success"]
    assert step["orders"] >= 10
    assert step["dumps"] >= 1
    assert step["hot_coast_pc"] > 1.0e-4
    assert step["hot_flown_pc"] < 1.0e-4
    env.close()
    refused = gym.make("space_env/DayMission-v0")
    refused.reset(seed=0)
    _obs, _reward, _terminated, _truncated, step = refused.step(1)
    assert step["success"] is False
    refused.close()


def test_three_days_keep_the_same_budget():
    campaign = build_day_campaign(horizon_sec=3 * DAY_SEC)
    assert campaign.horizon_sec == 3 * DAY_SEC
    assert campaign.vehicles == ("sentinel-2a", "landsat-9", "cubesat-6u")
    assert campaign.dv_spent_m_s == 1.0
    coast = dict(campaign.coast_pc)
    flown = dict(campaign.flown_pc)
    assert coast["HOT"] > 1.0e-4 and flown["HOT"] < 1.0e-4
    assert flown["debris-fragment"] < 1.0e-4
    assert sum(item.required for item in campaign.conjunctions) == 1
    one_day = build_day_campaign()
    assert len(campaign.plans[0].targets) > len(one_day.plans[0].targets)
    assert any(item.required and item.kind == "comms" for item in campaign.plans[0].items)
    _passes_do_not_overlap(campaign.plans)


def test_shared_dish_keeps_the_busier_pass():
    def book(vehicle: str, bits: float, start: float) -> WorkPlan:
        shot = WorkItem("imaging", "T", 0.0, 10.0, True, "снять", bits=bits)
        dump = WorkItem("comms", "Svalbard", start, start + 100.0, True, "сдать", bits=1.0)
        return WorkPlan(vehicle, 1_000.0, (shot, dump), 1.0e12, ())

    plans = _share_stations((book("sentinel-2a", 8.0, 100.0), book("landsat-9", 2.0, 120.0)))
    kept = next(item for item in plans[0].items if item.kind == "comms")
    dropped = next(item for item in plans[1].items if item.kind == "comms")
    assert kept.required is True
    assert dropped.required is False
    assert "занято" in dropped.reason
