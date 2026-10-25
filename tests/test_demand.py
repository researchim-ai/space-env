"""Заявка из файла: наблюдение без кода команды, успех — сданные биты."""
import gymnasium as gym

from space_env.envs.demand_env import packaged_demand
from space_env.eval.demand_report import demand_shift_report
from space_env.rl.expert import expert_action


def _run(path: str, expert: bool) -> dict:
    env = gym.make("space_env/DemandShift-v0", scenario=path, max_episode_steps=40)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (5,)
        assert "correct_action" not in info
        station = env.unwrapped._engine.scenario.ground_stations[0]
        assert station.name == "Kiruna"
        assert station.alt_m == 402.2
        assert float(obs[2]) == 0.0
        assert float(obs[3]) == 0.0
        done = False
        while not done:
            action = expert_action("space_env/DemandShift-v0", obs) if expert else 0
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        return info
    finally:
        env.close()


def test_two_request_files_do_not_share_one_baked_day():
    deliver = _run(str(packaged_demand("demand_deliver.yaml")), expert=True)
    late = _run(str(packaged_demand("demand_late.yaml")), expert=True)
    waited = _run(str(packaged_demand("demand_deliver.yaml")), expert=False)
    assert deliver["success"] is True
    assert deliver["deliver_to"] == "Kiruna"
    assert deliver["time_sec"] > 400.0
    assert deliver["delivered_bits"] > 0.0
    assert deliver["generated_bits"] > 0.0
    assert deliver["demand"] != late["demand"]
    assert late["success"] is False
    assert late["delivered_bits"] == 0.0
    assert late["time_sec"] <= late["deadline_sec"] + 1e-6
    assert waited["success"] is False
    assert waited["delivered_bits"] == 0.0


def test_demand_report_keeps_the_two_files_and_skips_learned():
    report = demand_shift_report()
    assert "learned" not in report
    assert [row["demand"] for row in report["rows"]] == ["kiruna-pass", "kiruna-early"]
    assert report["rows"][0]["success"] is True
    assert report["rows"][1]["success"] is False
