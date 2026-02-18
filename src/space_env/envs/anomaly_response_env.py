"""
AnomalyResponseEnv: реакция на аномалию — выбор процедуры (safe mode, сброс нагрузки, перезагрузка).
Состояние: телеметрия + тип аномалии. Награда за восстановление номинала.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# obs: battery_soc, temp_norm, anomaly_power, anomaly_thermal, anomaly_comms (one-hot или 0/1)
# action: 0=none, 1=safe_mode (снизить нагрузку), 2=reduce_power, 3=reboot (сброс в номинал с задержкой)


class AnomalyResponseEnv(gym.Env):
    """
    Упрощённая среда: аномалия уже произошла, агент выбирает процедуру восстановления.
    Состояние: [soc, temp_norm, anomaly_type_onehot(3)] = 5.
    Действия: 0=none, 1=safe_mode, 2=reduce_power, 3=reboot.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 20, seed: int | None = None):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self._steps = 0
        self._anomaly_type = 0  # 0=power, 1=thermal, 2=comms
        self._soc = 1.0
        self._temp_norm = 0.5
        self._recovered = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._anomaly_type = self.np_random.integers(0, 3)
        self._soc = float(self.np_random.uniform(0.3, 0.6))
        self._temp_norm = float(self.np_random.uniform(0.2, 0.8))
        self._recovered = False
        onehot = np.zeros(3, dtype=np.float32)
        onehot[self._anomaly_type] = 1.0
        obs = np.array([self._soc, self._temp_norm, onehot[0], onehot[1], onehot[2]], dtype=np.float32)
        return obs, {"anomaly_type": self._anomaly_type}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(np.asarray(action).flat[0])
        self._steps += 1
        reward = -0.05  # шаг
        if action == 1:  # safe_mode — улучшаем soc и temp
            self._soc = min(1.0, self._soc + 0.05)
            self._temp_norm = np.clip(self._temp_norm + 0.02, 0.0, 1.0)
        if action == 2:  # reduce_power — меньше разряд
            self._soc = min(1.0, self._soc + 0.08)
        if action == 3:  # reboot — сброс к номиналу (упрощённо)
            self._soc = 0.9
            self._temp_norm = 0.5
            self._recovered = True
            reward += 1.0
        if self._soc >= 0.85 and 0.3 <= self._temp_norm <= 0.7 and not self._recovered:
            self._recovered = True
            reward += 0.5
        terminated = self._recovered
        truncated = self._steps >= self.max_episode_steps
        onehot = np.zeros(3, dtype=np.float32)
        onehot[self._anomaly_type] = 1.0
        obs = np.array([self._soc, self._temp_norm, onehot[0], onehot[1], onehot[2]], dtype=np.float32)
        return obs, float(reward), terminated, truncated, {"recovered": self._recovered}
