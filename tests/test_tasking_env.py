"""Среда заявок: жадный план сдаёт проход, ранний срок и ожидание — нет."""
import gymnasium as gym

from space_env.envs.tasking_env import packaged_tasking
from space_env.eval.tasking_report import tasking_shift_report
from space_env.mission.tasking import requests_from_scenario, run_policy
from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.rl.expert import expert_action


def _run(path: str, expert: bool) -> dict:
    env = gym.make("space_env/TaskingShift-v0", scenario=path, max_episode_steps=40)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4,)
        assert "correct_action" not in info
        done = False
        while not done:
            action = expert_action("space_env/TaskingShift-v0", obs) if expert else 0
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        return info
    finally:
        env.close()


def test_greedy_delivers_kiruna_and_the_early_deadline_does_not():
    deliver = _run(str(packaged_tasking("tasking_deliver.yaml")), expert=True)
    late = _run(str(packaged_tasking("tasking_late.yaml")), expert=True)
    waited = _run(str(packaged_tasking("tasking_deliver.yaml")), expert=False)
    assert deliver["success"] is True
    assert deliver["request"] == "kiruna-pass"
    assert deliver["station"] == "Kiruna"
    assert deliver["time_sec"] > 400.0
    assert deliver["delivered_bits"] >= 8.0e6
    assert late["success"] is False
    assert late["request"] == "kiruna-early"
    assert late["time_sec"] <= late["deadline_sec"] + 1e-6
    assert waited["success"] is False
    assert waited["delivered_bits"] == 0.0


def test_env_success_matches_the_greedy_executor():
    path = packaged_tasking("tasking_deliver.yaml")
    info = _run(str(path), expert=True)
    spec = load_scenario_v2(path)
    world = world_from_scenario(path, attach_stations=True)
    report = run_policy(world, requests_from_scenario(spec), step_sec=30.0, policy="priority")
    assert info["success"] is True
    assert "kiruna-pass" in report["score"]["on_time"]
    assert any(item["destination"] == "Kiruna" for item in report["receipts"])


def test_tasking_report_keeps_two_files_and_skips_learned():
    report = tasking_shift_report()
    assert "learned" not in report
    assert [row["scenario"] for row in report["rows"]] == ["tasking_deliver.yaml", "tasking_late.yaml"]
    assert report["rows"][0]["greedy_success"] is True
    assert report["rows"][0]["random_success_rate"] < 1.0
    assert report["rows"][1]["greedy_success"] is False
    assert report["rows"][1]["random_success_rate"] == 0.0
