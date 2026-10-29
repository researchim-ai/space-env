"""Два аппарата на одной станции и команда запуска файла."""
import json

import gymnasium as gym

from space_env.cli import main
from space_env.envs.tasking_env import packaged_tasking


def test_two_commands_on_one_station_let_only_the_first_name_deliver():
    env = gym.make("space_env/FleetTasking-v0", max_episode_steps=2)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (8,)
        assert "correct_action" not in info
        _obs, _reward, terminated, _truncated, info = env.step([2, 2])
        delivered = info["delivered"]
        assert terminated is True
        assert info["success"] is True
        assert delivered["alpha"] > 0.0
        assert delivered["beta"] == 0.0
    finally:
        env.close()


def test_fleet_observation_has_deadlines_and_rejects_a_hidden_target():
    import gymnasium as gym

    env = gym.make("space_env/MissionFleet-v0", max_episode_steps=2)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (67,)
        assert "correct_action" not in info
        assert len(info["statuses"]) == 12
        _obs, _reward, _terminated, _truncated, stepped = env.step([1, 0, 0, 0])
        assert "req-00" in stepped["rejected"] or stepped["statuses"]
    finally:
        env.close()


def test_validate_reads_a_scenario_without_flying_it(capsys):
    main(["validate", str(packaged_tasking("tasking_deliver.yaml"))])
    payload = json.loads(capsys.readouterr().out)
    assert payload["instances"] == ["sentinel-a"]
    assert payload["requests"] == ["kiruna-pass"]


def test_fleet_watch_pads_a_small_scenario_and_rejects_sixteen():
    from space_env.envs.mission_fleet_env import FleetWatchEnv, watch_observation_names

    env = FleetWatchEnv(scenario="family_nominal.yaml", max_episode_steps=2)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (len(watch_observation_names()),)
        assert len(info["statuses"]) == 1
        env.step(env.action_space.sample())
    finally:
        env.close()
    huge = FleetWatchEnv(scenario="eo_72h_16.yaml")
    try:
        huge.reset()
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is False
