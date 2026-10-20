"""Тесты RL-сред (Gymnasium)."""
import numpy as np
import pytest

import gymnasium as gym

from space_env.envs import (
    AnomalyResponseEnv,
    ConjunctionAssessmentEnv,
    MissionPlannerEnv,
    ResourceManagerEnv,
)


def test_resource_manager_env():
    env = ResourceManagerEnv(step_sec=60.0, max_episode_steps=10, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    obs2, reward, term, trunc, info2 = env.step(0)
    assert obs2.shape == (5,)
    assert env.action_space.contains(0)
    env.close()


def test_mission_planner_env():
    env = MissionPlannerEnv(max_episode_steps=15, seed=123)
    obs, info = env.reset(seed=123)
    assert obs.shape == (7,)
    obs2, reward, term, trunc, info2 = env.step(1)  # imaging
    assert "imaging" in info2
    env.close()


def test_anomaly_response_env():
    env = AnomalyResponseEnv(max_episode_steps=10, seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (5,)
    obs2, reward, term, trunc, info2 = env.step(3)  # reboot
    assert obs2.shape == (5,)
    env.close()


def test_conjunction_env():
    env = ConjunctionAssessmentEnv(max_episode_steps=5, seed=2)
    obs, info = env.reset(seed=2)
    assert obs.shape == (5,)
    obs2, reward, term, trunc, info2 = env.step(0)
    assert obs2.shape == (5,)
    env.close()


def test_mission_planner_render_and_registry():
    import gymnasium as gym

    env = gym.make("space_env/MissionPlanner-v0", max_episode_steps=2, render_mode="ansi")
    obs, _info = env.reset(seed=0)
    assert obs.shape == (7,)
    _obs, _reward, _term, _trunc, info = env.step(0)
    assert "imaging" in info
    text = env.render()
    assert text is not None and "MissionPlanner" in text
    env.close()


def test_resource_manager_run_episode():
    env = ResourceManagerEnv(max_episode_steps=20, seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(25):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()
