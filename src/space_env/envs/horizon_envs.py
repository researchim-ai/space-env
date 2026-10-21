"""
Короткие среды на уже посчитанные решения горизонта.
ConjunctionQueue: один импульс по нормали или ожидание, Pc через 1200 с.
DecayWatch: поднять перигей или ждать, большая полуось через 3 ч.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.mission.horizon import (
    PC_THRESHOLD,
    build_conjunction_plan,
    build_decay_plan,
    build_reroute_schedule,
    build_two_burn_phase_plan,
    coast_conjunction,
    execute_decay_plan,
)
from space_env.rl.rewards import load_rewards

_COAST_SEC = 1200.0


class ConjunctionQueueEnv(gym.Env):
    """
    Наблюдение: Pc трёх объектов, доля бюджета, боковой промах горячей пары.
    Действия: 0 ждать, 1 импульс 1 м/с по нормали.
    Успех — после 1200 с горячая пара ниже 10⁻⁴, холодная тоже ниже, сожжён один импульс.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._steps = 0
        self._pcs = {"HOT": 0.0, "WARM": 0.0, "COLD": 0.0}
        self._cross_m = 0.0
        self._weights = load_rewards()["conjunction_queue"]

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                float(np.clip(self._pcs["HOT"] / 1.0e-2, 0.0, 1.0)),
                float(np.clip(self._pcs["WARM"] / 1.0e-2, 0.0, 1.0)),
                float(np.clip(self._pcs["COLD"] / 1.0e-2, 0.0, 1.0)),
                1.0,
                float(np.clip(0.5 + self._cross_m / 1_000.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        plan = build_conjunction_plan()
        for item in plan.tasks:
            if item.name in self._pcs:
                self._pcs[item.name] = item.probability
        hot = next(item for item in plan.tasks if item.name == "HOT")
        self._cross_m = 40.0 if hot.direction == "normal" else 0.0
        return self._obs(), {"plan": plan.lines()}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(np.asarray(action).flat[0])
        direction = "normal" if action == 1 else None
        self._pcs = coast_conjunction(direction)
        for name in ("HOT", "WARM", "COLD"):
            self._pcs.setdefault(name, 0.0)
        self._steps += 1
        burned = direction is not None
        success = burned and self._pcs["HOT"] < PC_THRESHOLD and self._pcs["COLD"] < PC_THRESHOLD
        reward = self._weights["avoided"] if success else -self._weights["missed"]
        info = {"pc": dict(self._pcs), "burned": burned, "success": success}
        return self._obs(), float(reward), True, False, info


class DecayWatchEnv(gym.Env):
    """
    Наблюдение: потеря полуоси, нужен ли импульс, высота.
    Действия: 0 ждать, 1 импульс из плана.
    Успех на 400 км — полуось после импульса выше прогона без него.
    Успех на 780 км — импульса не было.
    """

    metadata = {"render_modes": []}

    def __init__(self, altitude_m: float = 400_000.0, horizon_sec: float = 3.0 * 3600.0, max_episode_steps: int = 1):
        super().__init__()
        self.altitude_m = altitude_m
        self.horizon_sec = horizon_sec
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._loss_m = 0.0
        self._dv_m_s = 0.0
        self._should_burn = False
        self._weights = load_rewards()["decay_watch"]

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                float(np.clip(self._loss_m / 50.0, 0.0, 1.0)),
                1.0 if self._should_burn else 0.0,
                float(np.clip(self.altitude_m / 1_000_000.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        plan = build_decay_plan(altitude_m=self.altitude_m, horizon_sec=self.horizon_sec)
        self._loss_m = plan.metric
        self._should_burn = plan.tasks[0].required
        self._dv_m_s = plan.tasks[0].dv_m_s
        return self._obs(), {"plan": plan.lines(), "should_burn": self._should_burn}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(np.asarray(action).flat[0])
        burned = action == 1
        if self._should_burn:
            done = execute_decay_plan(altitude_m=self.altitude_m, horizon_sec=self.horizon_sec)
            success = burned and done.metric > 10.0
            gain = done.metric
        else:
            success = not burned
            gain = 0.0
        reward = self._weights["correct"] if success else -self._weights["wrong"]
        info = {"gain_m": gain, "burned": burned, "should_burn": self._should_burn, "success": success}
        return self._obs(), float(reward), True, False, info


class PhaseBurnEnv(gym.Env):
    """
    Наблюдение: второй импульс назначен на период ведомого, доля эпизода.
    Действия: 0 ждать, 1 импульс 0.5 м/с ретроградно.
    Успех — скорость фазы за следующие 300 с меньше 0.05 м/с.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._weights = load_rewards()["phase_burn"]

    def _obs(self) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self._obs(), {"scheduled": "один период ведомого, затем −0.5 м/с"}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        burn = int(np.asarray(action).flat[0]) == 1
        plan = build_two_burn_phase_plan(burn=burn)
        success = burn and abs(plan.metric) < 0.05
        reward = self._weights["stopped"] if success else -self._weights["drifting"]
        info = {"rate_m_s": plan.metric, "burned": burn, "success": success, "plan": plan.lines()}
        return self._obs(), float(reward), True, False, info


class RerouteEnv(gym.Env):
    """
    Наблюдение: прямой луч закрыт, обход через C доступен как решение.
    Действия: 0 оставить пакет на B, 1 вести B→C→A.
    Успех — пакет на A, и каждый скачок плана перенёс биты.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._weights = load_rewards()["reroute"]

    def _obs(self) -> np.ndarray:
        return np.array([1.0, 1.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self._obs(), {"closed": "B-A", "detour": "B-C-A"}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        use_detour = int(np.asarray(action).flat[0]) == 1
        plan = build_reroute_schedule(close_direct=True, use_detour=use_detour)
        success = use_detour and plan.metric == 1.0
        reward = self._weights["delivered"] if success else -self._weights["stuck"]
        info = {"arrived": plan.metric == 1.0, "detour": use_detour, "success": success, "plan": plan.lines()}
        return self._obs(), float(reward), True, False, info


class DecayWatchHighEnv(DecayWatchEnv):
    """Та же среда на 780 км: план перигей не поднимает."""

    def __init__(self, max_episode_steps: int = 1, horizon_sec: float = 3.0 * 3600.0):
        super().__init__(altitude_m=780_000.0, horizon_sec=horizon_sec, max_episode_steps=max_episode_steps)
