"""
ConjunctionAssessmentEnv: решение — маневрировать или нет при предупреждении о сближении.
Состояние: относительная геометрия (или miss_distance, prob), доступный Δv.
Награда: снижение риска столкновения vs экономия Δv.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ConjunctionAssessmentEnv(gym.Env):
    """
    Упрощённо: state = [miss_distance_km/100, collision_prob, delta_v_remaining_norm].
    action: 0 = не маневрировать, 1 = маневр (тратим фиксированный Δv, снижаем prob).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 10, seed: int | None = None):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)  # 0 no maneuver, 1 maneuver
        self._steps = 0
        self._miss_km = 1.0
        self._prob = 0.0
        self._dv_remaining = 1.0
        self._dv_budget = 10.0  # м/с

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._miss_km = float(self.np_random.uniform(0.5, 20.0))
        self._prob = float(self.np_random.uniform(0.0, 0.01))
        self._dv_remaining = self._dv_budget
        return self._get_obs(), {"miss_km": self._miss_km, "prob": self._prob}

    def _get_obs(self) -> np.ndarray:
        miss_norm = np.clip(self._miss_km / 20.0, 0.0, 1.0)
        prob_norm = np.clip(self._prob * 100.0, 0.0, 1.0)
        dv_norm = self._dv_remaining / max(1e-6, self._dv_budget)
        return np.array([miss_norm, prob_norm, dv_norm], dtype=np.float32)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(np.asarray(action).flat[0])
        self._steps += 1
        reward = 0.0
        if action == 1 and self._dv_remaining >= 2.0:
            self._dv_remaining -= 2.0
            self._miss_km += 3.0  # манёвр увеличивает промах
            self._prob *= 0.2  # снижаем вероятность
            reward += 0.3  # снизили риск
            reward -= 0.05  # потратили Δv
        else:
            if self._prob > 0.005:
                reward -= 0.1 * self._prob * 100  # штраф за риск
            if action == 1 and self._dv_remaining < 2.0:
                reward -= 0.1  # хотел маневр, но нет Δv
        terminated = self._steps >= 1  # одно решение за эпизод (упрощённо)
        truncated = self._steps >= self.max_episode_steps
        return self._get_obs(), float(reward), terminated, truncated, {"miss_km": self._miss_km, "prob": self._prob}
