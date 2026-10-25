"""Регрессии R1.4 и R1.5: единицы CDM, исход без ложной сдачи, одношаговое обучение."""
import gymnasium as gym
import numpy as np
import pytest

from space_env.core.cdm import parse_cdm_kvn
from space_env.rl.reinforce import evaluate_linear_policy, train_linear_reinforce, train_multidiscrete_reinforce


class _GapBandit(gym.Env):
    """Один шаг. Действие 1 получает 1, действие 0 получает 0."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.ones(1, dtype=np.float32), {}

    def step(self, action):
        reward = 1.0 if int(action) == 1 else 0.0
        return np.ones(1, dtype=np.float32), reward, True, False, {"success": reward > 0.0}


def test_explicit_cdm_units_are_not_scaled_twice():
    message = parse_cdm_kvn(
        """
        MISS_DISTANCE = 100 [m]
        RELATIVE_SPEED = 1 [m/s]
        OBJECT = OBJECT1
        CR_R = 1 [km**2]
        CT_T = 1 [km**2]
        CN_N = 1 [km**2]
        CRDOT_R = 1 [km**2/s]
        CRDOT_RDOT = 1 [km**2/s**2]
        """
    )
    assert message.miss_distance_m == pytest.approx(100.0)
    assert message.relative_speed_m_s == pytest.approx(1.0)
    state = message.objects[0].state_covariance_si
    assert state is not None
    assert state[0, 3] == pytest.approx(1.0e6)
    assert state[3, 3] == pytest.approx(1.0e6)


def test_mismatched_covariance_frames_are_rejected_and_a_negative_variance_is_diagnosed():
    with pytest.raises(ValueError, match="системах"):
        parse_cdm_kvn(
            """
            MISS_DISTANCE = 1 [km]
            OBJECT = OBJECT1
            COV_REF_FRAME = RTN
            CR_R = 1 [km**2]
            OBJECT = OBJECT2
            COV_REF_FRAME = GCRF
            CR_R = 1 [km**2]
            """
        )
    diagnosed = parse_cdm_kvn(
        """
        MISS_DISTANCE = 1 [km]
        OBJECT = OBJECT1
        CR_R = -1 [km**2]
        CTDOT_TDOT = 1 [km**2/s**2]
        """
    )
    assert diagnosed.diagnostics
    assert diagnosed.combined_state_covariance_si() is None


def test_pass_fault_procedure_does_not_claim_a_station_delivery():
    env = gym.make("space_env/PassFault-v0", max_episode_steps=6)
    obs, _info = env.reset(seed=0)
    from space_env.rl.expert import expert_action

    done = False
    info = {}
    while not done:
        obs, _reward, _terminated, truncated, info = env.step(expert_action("space_env/PassFault-v0", obs))
        done = bool(truncated)
    env.close()
    assert info["operation_completed"] is True
    assert info["request_delivered"] is False
    assert info["delivered"] is False
    assert info["mission_succeeded"] is False
    assert info["unmet_reason"] == "команда не принята"


def _preferred(weights: np.ndarray, bias: np.ndarray) -> float:
    logits = weights @ np.ones(1) + bias
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return float(probs[1])


def test_multidiscrete_resume_continues_the_seed_stream():
    seen: list[int] = []

    class _SeedBandit(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.MultiDiscrete([2, 2])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            seen.append(int(seed or 0))
            return np.ones(1, dtype=np.float32), {}

        def step(self, action):
            return np.ones(1, dtype=np.float32), 0.0, True, False, {"success": False}

    first = train_multidiscrete_reinforce("seed-bandit", episodes=3, seed=0, env=_SeedBandit())
    seen.clear()
    train_multidiscrete_reinforce(
        "seed-bandit",
        episodes=3,
        seed=0,
        env=_SeedBandit(),
        weights=first["weights"],
        bias=first["bias"],
        baseline=first["baseline"],
        baseline_count=first["baseline_count"],
        rng_state=first["rng_state"],
    )
    assert seen == [3, 4, 5]


def test_one_step_bandit_moves_the_weights_and_resume_repeats():
    probabilities = []
    for seed in (0, 1, 2):
        frozen = train_linear_reinforce("bandit", episodes=20, max_episode_steps=1, seed=seed, lr=0.0, env=_GapBandit())
        learned = train_linear_reinforce("bandit", episodes=40, max_episode_steps=1, seed=seed, lr=0.4, env=_GapBandit())
        assert not np.allclose(learned["weights"], frozen["weights"])
        probabilities.append(_preferred(learned["weights"], learned["bias"]))
    assert all(item > 0.8 for item in probabilities)
    first = train_linear_reinforce("bandit", episodes=6, max_episode_steps=1, seed=4, lr=0.3, env=_GapBandit())
    second = train_linear_reinforce(
        "bandit",
        episodes=6,
        max_episode_steps=1,
        seed=4,
        lr=0.3,
        env=_GapBandit(),
        weights=first["weights"],
        bias=first["bias"],
        baseline=first["baseline"],
        baseline_count=first["baseline_count"],
        rng_state=first["rng_state"],
    )
    whole = train_linear_reinforce("bandit", episodes=12, max_episode_steps=1, seed=4, lr=0.3, env=_GapBandit())
    assert second["returns"] == pytest.approx(whole["returns"][6:])
    held = evaluate_linear_policy("bandit", learned["weights"], learned["bias"], episodes=2, max_episode_steps=1, env=_GapBandit())
    assert held["success_rate"] == 1.0
    assert np.allclose(learned["weights"], train_linear_reinforce(
        "bandit",
        episodes=0,
        max_episode_steps=1,
        seed=0,
        env=_GapBandit(),
        weights=learned["weights"],
        bias=learned["bias"],
    )["weights"])
