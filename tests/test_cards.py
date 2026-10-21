"""Карточка среды, повтор сида и тренер действия из нескольких чисел."""
import gymnasium as gym
import numpy as np

import space_env
from space_env.envs import registered_env_ids
from space_env.envs.cards import ENV_CARDS
from space_env.rl.expert import expert_action
from space_env.rl.reinforce import train_multidiscrete_reinforce


def _roll(env_id: str, seed: int, steps: int) -> list[list[float]]:
    env = gym.make(env_id, max_episode_steps=steps)
    obs, _info = env.reset(seed=seed)
    trace = [np.asarray(obs, dtype=float).round(6).tolist()]
    for _ in range(steps):
        action = 0 if hasattr(env.action_space, "n") else np.zeros(env.action_space.nvec.shape, dtype=int)
        obs, _reward, terminated, truncated, _info = env.step(action)
        trace.append(np.asarray(obs, dtype=float).round(6).tolist())
        if terminated or truncated:
            break
    env.close()
    return trace


def test_every_env_has_a_card_matching_its_vector():
    assert set(ENV_CARDS) == set(registered_env_ids())
    for env_id in registered_env_ids():
        env = gym.make(env_id)
        width = int(env.observation_space.shape[0])
        env.close()
        card = ENV_CARDS[env_id]
        assert len(card.observation) == width
        assert card.action
        assert "success" not in card.success or card.success


def test_same_seed_repeats_the_observation():
    for env_id in ("space_env/ResourceManager-v0", "space_env/AnomalyResponse-v0", "space_env/ConstellationOps-v0"):
        assert _roll(env_id, 0, 3) == _roll(env_id, 0, 3)


def test_public_package_lists_plans_and_cards():
    assert "space_env/PhaseBurn-v0" in space_env.ENV_CARDS
    assert "space_env/Reroute-v0" in space_env.registered_env_ids()
    assert callable(space_env.make_engine)
    assert callable(space_env.build_two_burn_phase_plan)
    assert callable(space_env.build_reroute_schedule)


def test_multidiscrete_trainer_runs_on_the_chain():
    result = train_multidiscrete_reinforce(
        "space_env/ConstellationOps-v0",
        episodes=2,
        max_episode_steps=2,
        seed=0,
    )
    assert len(result["returns"]) == 2
    assert np.isfinite(result["mean_last"])


def test_phase_burn_and_reroute_experts():
    phase = gym.make("space_env/PhaseBurn-v0")
    obs, _info = phase.reset(seed=0)
    _obs, _reward, _term, _trunc, info = phase.step(expert_action("space_env/PhaseBurn-v0", obs))
    assert info["success"]
    assert abs(info["rate_m_s"]) < 0.05
    phase.close()
    route = gym.make("space_env/Reroute-v0")
    obs, _info = route.reset(seed=0)
    _obs, _reward, _term, _trunc, info = route.step(expert_action("space_env/Reroute-v0", obs))
    assert info["success"] and info["arrived"]
    route.close()
    stuck = gym.make("space_env/Reroute-v0")
    obs, _info = stuck.reset(seed=0)
    _obs, _reward, _term, _trunc, info = stuck.step(0)
    assert info["success"] is False and info["arrived"] is False
    stuck.close()
