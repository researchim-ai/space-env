"""План работы: выбор сеанса, заполнение памяти и исполнение списка дел."""
import numpy as np

from space_env.mission.events import EventKind
from space_env.mission.workplan import (
    Opportunity,
    build_acquisition_plan,
    maneuver_item,
    schedule_acquisition,
)
from space_env.rl.expert import expert_action


def _opp(kind, name, start, end, rate, margin=0.0):
    return Opportunity(kind, name, start, end, rate, margin)


def test_weak_pass_is_left_for_a_faster_window():
    items = schedule_acquisition(
        [
            _opp("imaging", "S2-1", 0.0, 60.0, 10.0),
            _opp("comms", "Svalbard", 200.0, 400.0, 25.0, 1.0),
            _opp("comms", "Kiruna", 800.0, 1000.0, 100.0, 8.0),
        ],
        memory_bits=5_000.0,
        imaging_cap_s=10_000.0,
        downlink_bps=100.0,
    )
    required = [item for item in items if item.required]
    assert [item.name for item in required] == ["S2-1", "Kiruna"]
    skipped = next(item for item in items if item.name == "Svalbard")
    assert skipped.required is False
    assert "не занимать" in skipped.reason
    assert required[0].end_sec <= required[1].start_sec


def test_full_store_keeps_the_early_pass():
    items = schedule_acquisition(
        [
            _opp("imaging", "S2-1", 0.0, 10.0, 100.0),
            _opp("imaging", "S2-2", 20.0, 30.0, 100.0),
            _opp("comms", "Svalbard", 15.0, 25.0, 10.0),
            _opp("comms", "Kiruna", 40.0, 50.0, 100.0),
        ],
        memory_bits=1_000.0,
        imaging_cap_s=10_000.0,
        downlink_bps=100.0,
    )
    comms = [item for item in items if item.kind == "comms" and item.name == "Svalbard"]
    assert comms[0].required
    assert not any("не занимать Svalbard" in item.reason for item in items)


def test_maneuver_is_attached_by_the_caller():
    burn = maneuver_item(300.0, 0.5, "retrograde", "300 с: импульс 0.5 м/с ретроградно.")
    plan = build_acquisition_plan("sentinel-2a", horizon_sec=600.0, sample_sec=60.0, extra_items=[burn])
    events = plan.as_planned_events()
    assert any(event.kind == EventKind.MANEUVER and event.payload["dv_m_s"] == 0.5 for event in events)
    assert any("импульс" in line for line in plan.lines())


def test_sentinel_plan_names_tasks_and_expert_follows_them():
    import gymnasium as gym

    env = gym.make("space_env/WorkPlan-v0", max_episode_steps=90)
    obs, info = env.reset(seed=0)
    assert info["plan"]
    assert info["tasks"]
    assert env.observation_space.contains(obs)
    done = False
    steps = 0
    while not done and steps < 90:
        action = expert_action("space_env/WorkPlan-v0", obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    assert info["success"]
    assert all(task["done"] for task in info["tasks"])
    env.close()
    assert np.isfinite(obs).all()
