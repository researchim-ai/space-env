"""Сутки трёх аппаратов: заявки, сеансы и один бюджет Δv."""
import gymnasium as gym
import pytest

from space_env.mission.campaign import DAY_SEC, _Draft, _assign_dishes, _share_stations, build_day_campaign, orbital_period_s
from space_env.rl.rewards import load_rewards
from space_env.mission.workplan import Opportunity, WorkItem, WorkPlan, build_acquisition_plan, schedule_acquisition
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
    assert campaign.vehicles == ("sentinel-2a", "landsat-9", "sar-small", "cubesat-6u")
    assert campaign.dv_spent_m_s == 1.0
    assert campaign.dv_spent_m_s <= campaign.dv_budget_m_s
    hot = next(item for item in campaign.conjunctions if item.name == "HOT")
    warm = next(item for item in campaign.conjunctions if item.name == "WARM")
    debris = next(item for item in campaign.conjunctions if item.name == "debris-fragment")
    assert hot.required and hot.dv_m_s == 1.0 and hot.direction == "prograde"
    assert 10 * 3_600.0 < hot.start_sec < warm.start_sec < DAY_SEC
    assert debris.required is False and debris.probability < 1.0e-4
    sentinel = campaign.plans[0]
    assert len(sentinel.targets) >= 10
    assert any(item.required and item.kind == "imaging" for item in sentinel.items)
    assert any(item.required and item.kind == "comms" for item in sentinel.items)
    landsat = next(plan for plan in campaign.plans if plan.vehicle == "landsat-9")
    cubesat = next(plan for plan in campaign.plans if plan.vehicle == "cubesat-6u")
    radar = next(plan for plan in campaign.plans if plan.vehicle == "sar-small")
    assert any(item.required and item.kind == "comms" for item in radar.items)
    assert any(item.required and item.kind == "comms" for item in landsat.items)
    assert len(cubesat.targets) >= 10
    assert any(item.required and item.kind == "imaging" for item in cubesat.items)
    assert any(item.required and item.kind == "comms" and item.name == "Moscow" for item in cubesat.items)
    denied = [
        (plan.vehicle, item)
        for plan in campaign.plans
        for item in plan.items
        if item.name == "Svalbard" and "занято" in item.reason
    ]
    assert denied
    for vehicle, item in denied:
        holder = item.reason.split("аппаратом ", 1)[1].rstrip(".")
        assert holder != vehicle
        assert holder in ("sentinel-2a", "landsat-9", "sar-small")
    involved = {vehicle for vehicle, _item in denied}
    involved.update(item.reason.split("аппаратом ", 1)[1].rstrip(".") for _vehicle, item in denied)
    assert {"sentinel-2a", "landsat-9", "sar-small"} <= involved
    above = [item for item in campaign.conjunctions if item.probability >= 1.0e-4]
    assert len(above) >= 2
    assert sum(item.required for item in above) == 1
    coast = dict(campaign.coast_pc)
    flown = dict(campaign.flown_pc)
    assert campaign.burn_at_sec == pytest.approx(0.0)
    assert coast["HOT"] > 1.0e-4 and flown["HOT"] < 1.0e-4
    assert coast["WARM"] > 1.0e-4 and flown["WARM"] < 1.0e-4
    assert flown["debris-fragment"] < 1.0e-4
    for name in ("sentinel-2b", "landsat-9", "cubesat-6u", "debris-class", "plane-offset"):
        other = next(item for item in campaign.conjunctions if item.name == name)
        assert other.required is False and other.probability < 1.0e-4
        assert flown[name] < 1.0e-4
    elements = dict((name, (da, di, dnu)) for name, da, di, dnu in campaign.population)
    assert elements["HOT"][0] == pytest.approx(-40.0)
    assert elements["HOT"][2] == pytest.approx(-0.032455)
    assert elements["debris-fragment"][1] == pytest.approx(0.2)
    assert elements["plane-offset"][1] == pytest.approx(0.05)
    assert elements["plane-offset"][2] == pytest.approx(20.0)
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
    assert campaign.vehicles == ("sentinel-2a", "landsat-9", "sar-small", "cubesat-6u")
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


def test_lost_dish_drops_the_image_that_needed_the_dump():
    memory = 100.0
    cap = 1.0e6
    downlink = 100.0

    def draft(vehicle: str, opportunities: tuple[Opportunity, ...]) -> _Draft:
        items = schedule_acquisition(
            list(opportunities),
            memory_bits=memory,
            imaging_cap_s=cap,
            downlink_bps=downlink,
        )
        plan = WorkPlan(vehicle, 1_000.0, tuple(items), memory, ())
        return _Draft(plan, opportunities, cap, downlink)

    keeper = draft(
        "sentinel-2a",
        (
            Opportunity("imaging", "A", 0.0, 10.0, 10.0),
            Opportunity("comms", "Svalbard", 100.0, 110.0, 100.0),
        ),
    )
    loser = draft(
        "landsat-9",
        (
            Opportunity("imaging", "C", 0.0, 10.0, 10.0),
            Opportunity("comms", "Svalbard", 105.0, 130.0, 100.0),
            Opportunity("imaging", "D", 200.0, 210.0, 10.0),
        ),
    )
    plans = _assign_dishes((keeper, loser))
    assert any(item.required and item.kind == "comms" for item in plans[0].items)
    assert not any(item.required and item.kind == "comms" for item in plans[1].items)
    assert any(item.name == "D" and item.required is False for item in plans[1].items)
    assert any("занято" in item.reason for item in plans[1].items)


def test_lost_svalbard_moves_the_dump_to_the_free_station():
    memory = 100.0
    cap = 1.0e6
    downlink = 100.0

    def draft(vehicle: str, opportunities: tuple[Opportunity, ...]) -> _Draft:
        items = schedule_acquisition(
            list(opportunities),
            memory_bits=memory,
            imaging_cap_s=cap,
            downlink_bps=downlink,
        )
        plan = WorkPlan(vehicle, 1_000.0, tuple(items), memory, ())
        return _Draft(plan, opportunities, cap, downlink)

    keeper = draft(
        "sentinel-2a",
        (
            Opportunity("imaging", "A", 0.0, 10.0, 10.0),
            Opportunity("comms", "Svalbard", 100.0, 160.0, 100.0),
        ),
    )
    loser = draft(
        "landsat-9",
        (
            Opportunity("imaging", "C", 0.0, 10.0, 10.0),
            Opportunity("comms", "Svalbard", 140.0, 200.0, 100.0),
            Opportunity("comms", "Sioux Falls", 300.0, 360.0, 100.0),
        ),
    )
    plans = _assign_dishes((keeper, loser))
    assert any(item.required and item.name == "Svalbard" for item in plans[0].items)
    assert any(item.required and item.name == "Sioux Falls" for item in plans[1].items)
    assert not any(item.required and item.name == "Svalbard" for item in plans[1].items)
    assert any(item.name == "Svalbard" and "занято" in item.reason for item in plans[1].items)


def test_day_shift_executes_every_required_job():
    env = gym.make("space_env/DayShift-v0", max_episode_steps=1200)
    obs, info = env.reset(seed=0)
    assert info["jobs"] >= 10
    done = False
    last = {}
    for _ in range(1200):
        obs, _reward, terminated, truncated, last = env.step(expert_action("space_env/DayShift-v0", obs))
        done = bool(terminated or truncated)
        if done:
            break
    assert done and last["success"]
    assert last["burned"] is True
    assert last["missed"] == 0
    assert last["done"] == last["jobs"]
    for vehicle in ("sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"):
        assert last["engine_images"][vehicle] > 0
        assert last["engine_downlink_mb"][vehicle] > 0.0
    env.close()
    refused = gym.make("space_env/DayShift-v0", max_episode_steps=400)
    refused.reset(seed=0)
    _obs, _reward, terminated, _truncated, info = refused.step(0)
    assert terminated and info["success"] is False and info["missed"] == 1
    refused.close()


def test_dish_award_walks_every_shared_window():
    env = gym.make("space_env/DishAward-v0")
    obs, info = env.reset(seed=0)
    assert info["station"] == "Svalbard"
    assert info["remaining"] >= 2
    done = False
    step = {}
    for _ in range(24):
        obs, _reward, terminated, truncated, step = env.step(expert_action("space_env/DishAward-v0", obs))
        done = bool(terminated or truncated)
        if done:
            break
    assert done and step["success"]
    assert step["awards"] >= 2
    assert step["remaining"] == 0
    assert {"sentinel-2a", "landsat-9", "sar-small"} <= set(step["seen"])
    assert set(step["dumps"]) == {"sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"}
    assert all(step["dumps"].values())
    assert step["dumps"]["cubesat-6u"] == "Moscow"
    assert step["downlink_mb"]
    assert all(value > 0.0 for value in step["downlink_mb"].values())
    env.close()
    refused = gym.make("space_env/DishAward-v0")
    refused.reset(seed=0)
    _obs, _reward, terminated, _truncated, step = refused.step(3)
    assert terminated and step["success"] is False
    refused.close()


def test_operator_shift_awards_the_dishes_and_flies_them():
    env = gym.make("space_env/OperatorShift-v0", max_episode_steps=1600)
    obs, info = env.reset(seed=0)
    assert info["phase"] == "award"
    assert info["vehicles"] == ["sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"]
    assert info["horizon_sec"] == 86_400.0
    assert info["remaining"] >= 2
    done = False
    step = {}
    for _ in range(1600):
        obs, _reward, terminated, truncated, step = env.step(expert_action("space_env/OperatorShift-v0", obs))
        done = bool(terminated or truncated)
        if done:
            break
    assert done and step["success"]
    assert step["phase"] == "execute"
    assert step["awards"] >= 2
    assert step["burned"] is True
    assert step["done"] == step["jobs"]
    for vehicle in ("sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"):
        assert step["engine_images"][vehicle] > 0
        assert step["engine_downlink_mb"][vehicle] > 0.0
    env.close()
    refused = gym.make("space_env/OperatorShift-v0")
    refused.reset(seed=0)
    _obs, _reward, terminated, _truncated, step = refused.step(0)
    assert terminated and step["success"] is False and step["phase"] == "award"
    refused.close()


def test_operator_shift_accepts_a_shorter_fleet():
    env = gym.make(
        "space_env/OperatorShift-v0",
        keys=("landsat-9", "sar-small"),
        horizon_sec=7_200.0,
        dv_budget_m_s=0.0,
    )
    obs, info = env.reset(seed=0)
    assert info["vehicles"] == ["landsat-9", "sar-small"]
    assert info["horizon_sec"] == 7_200.0
    assert obs.shape == (1 + 3 * 2 + 5,)
    assert env.action_space.n == 6
    env.close()


def test_operator_shift_can_stop_after_the_awards():
    env = gym.make("space_env/OperatorShift-v0", until="award", max_episode_steps=24)
    obs, _info = env.reset(seed=0)
    total = 0.0
    done = False
    step = {}
    for _ in range(24):
        obs, reward, terminated, truncated, step = env.step(expert_action("space_env/OperatorShift-v0", obs))
        total += float(reward)
        done = bool(terminated or truncated)
        if done:
            break
    assert done and step["success"]
    assert step["until"] == "award"
    assert step["phase"] == "award"
    assert step["awards"] >= 2
    assert step["engine_images"] == {}
    weights = load_rewards()["operator_shift"]
    assert total == pytest.approx(step["awards"] * weights["step_done"] + weights["task_done"])
    env.close()
    with pytest.raises(ValueError):
        gym.make("space_env/OperatorShift-v0", until="coast")
