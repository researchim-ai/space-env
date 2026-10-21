"""Планы манёвра на горизонт: порог Pc, период фазы, сопротивление и обход луча."""
import pytest

from space_env.mission.horizon import (
    build_conjunction_plan,
    build_decay_plan,
    build_reroute_schedule,
    build_two_burn_phase_plan,
    schedule_conjunctions,
)


def test_queue_spends_the_only_metre_on_the_hottest_pair():
    tasks = schedule_conjunctions(
        [("HOT", 2e-3, 20.0, 40.0), ("WARM", 1e-3, 10.0, 5.0), ("COLD", 1e-8, 0.0, 5000.0)],
        budget_m_s=1.0,
        burn_m_s=1.0,
    )
    by_name = {item.name: item for item in tasks}
    assert by_name["HOT"].required and by_name["HOT"].direction == "normal"
    assert by_name["WARM"].required is False and "бюджет" in by_name["WARM"].reason
    assert by_name["COLD"].required is False and "ниже порога" in by_name["COLD"].reason


def test_screened_queue_matches_the_threshold():
    plan = build_conjunction_plan(budget_m_s=1.0)
    by_name = {item.name: item for item in plan.tasks}
    assert by_name["HOT"].required and by_name["HOT"].direction == "normal"
    assert by_name["HOT"].probability > 1e-4
    assert by_name["WARM"].probability > 1e-4 and by_name["WARM"].required is False
    assert by_name["COLD"].probability < 1e-4 and by_name["COLD"].required is False


def test_second_burn_stops_the_phase_rate():
    plan = build_two_burn_phase_plan()
    burn = next(item for item in plan.tasks if item.direction == "retrograde")
    assert burn.start_sec == pytest.approx(6027.0, abs=30.0)
    assert abs(plan.metric) < 0.05
    assert "накопленная фаза осталась" in plan.lines()[-1]


def test_decay_burn_follows_the_altitude():
    low = build_decay_plan(altitude_m=400_000.0)
    high = build_decay_plan(altitude_m=780_000.0)
    assert low.metric > 5.0
    assert low.tasks[0].required and low.tasks[0].direction == "prograde"
    assert low.tasks[0].dv_m_s > 0.0
    assert high.metric < 5.0
    assert high.tasks[0].required is False


def test_burn_drops_hot_pair_below_the_threshold():
    from space_env.mission.horizon import execute_conjunction_plan

    done = execute_conjunction_plan()
    assert done.reference > 1.0e-4
    assert done.metric < 1.0e-4
    assert any(item.name == "WARM" and item.required is False for item in done.tasks)
    assert "Отдельный импульс на WARM не ставился" in done.lines()[-1]


def test_perigee_raise_finishes_above_the_coast():
    from space_env.mission.horizon import execute_decay_plan

    low = execute_decay_plan(altitude_m=400_000.0)
    high = execute_decay_plan(altitude_m=780_000.0)
    assert low.tasks[0].required and low.metric > 10.0
    assert high.tasks[0].required is False


def test_queue_and_decay_experts_meet_the_same_criterion():
    import gymnasium as gym

    from space_env.rl.expert import expert_action

    queue = gym.make("space_env/ConjunctionQueue-v0")
    obs, _info = queue.reset(seed=0)
    obs, _reward, _term, _trunc, info = queue.step(expert_action("space_env/ConjunctionQueue-v0", obs))
    assert info["success"] and info["pc"]["COLD"] < 1.0e-4
    queue.close()
    low = gym.make("space_env/DecayWatch-v0")
    obs, _info = low.reset(seed=0)
    _obs, _reward, _term, _trunc, info = low.step(expert_action("space_env/DecayWatch-v0", obs))
    assert info["success"] and info["gain_m"] > 10.0
    low.close()
    high = gym.make("space_env/DecayWatchHigh-v0")
    obs, _info = high.reset(seed=0)
    _obs, _reward, _term, _trunc, info = high.step(expert_action("space_env/DecayWatchHigh-v0", obs))
    assert info["success"] and info["burned"] is False
    high.close()


def test_closed_direct_link_uses_the_open_detour():
    direct = build_reroute_schedule(close_direct=False)
    around = build_reroute_schedule(close_direct=True)
    assert direct.metric == 1.0
    assert [item.name for item in direct.required_items()] == ["B"]
    assert around.metric == 1.0
    assert [item.name for item in around.required_items()] == ["B", "C"]
    assert any("B–A закрыт" in line for line in around.lines())
