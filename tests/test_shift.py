"""Смена группировки, индексы карточки и отчёт обучения."""
import gymnasium as gym
import numpy as np

from space_env.envs.cards import ENV_CARDS, observation_index
from space_env.rl.expert import expert_action
from space_env.rl.report import training_report


def test_expert_reads_packet_and_hot_pc_from_the_card():
    ops = "space_env/ConstellationOps-v0"
    assert observation_index(ops, "доля пакета 1") == 1
    assert observation_index(ops, "угол места 1") == 2
    obs = np.zeros(len(ENV_CARDS[ops].observation))
    obs[observation_index(ops, "доля пакета 2")] = 1.0
    obs[observation_index(ops, "угол места 4")] = 1.0
    obs[observation_index(ops, "угол места 1")] = 0.2
    action = expert_action(ops, obs)
    assert list(action) == [1, 2]
    queue = "space_env/ConjunctionQueue-v0"
    assert observation_index(queue, "Pc горячей пары") == 0


def test_shift_delivers_without_burning_the_cold_pair():
    env = gym.make("space_env/FleetShift-v0", max_episode_steps=8)
    obs, info = env.reset(seed=0)
    assert info["cold_pc"] < 1.0e-4
    done = False
    while not done:
        obs, _reward, terminated, truncated, info = env.step(expert_action("space_env/FleetShift-v0", obs))
        done = bool(terminated or truncated)
    assert info["success"]
    assert info["wasted_burn"] is False
    env.close()


def test_burn_on_the_cold_pair_fails_the_shift():
    env = gym.make("space_env/FleetShift-v0", max_episode_steps=1)
    _obs, info = env.reset(seed=0)
    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0, 4]))
    assert info["wasted_burn"] is True
    assert info["success"] is False
    env.close()


def test_training_report_keeps_expert_and_random_numbers():
    report = training_report(
        "space_env/ConstellationOps-v0",
        episodes=2,
        max_episode_steps=8,
        seed=0,
        eval_episodes=1,
    )
    assert report["expert_success_rate"] == 1.0
    assert np.isfinite(report["random_mean_reward"])
    assert np.isfinite(report["trained_mean_last"])
    assert isinstance(report["learned"], bool)
