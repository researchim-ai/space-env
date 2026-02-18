"""
ResourceManagerEnv: управление нагрузками и ресурсами (батарея, термика).
Состояние: орбита (высота), eclipse, SoC, температура. Действия: уровень нагрузки.
Награда: выживание без разряда/перегрева, бонус за выполнение плана.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core import Orbit, Time
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, PowerConfig, PropulsionConfig, ThermalConfig
from space_env.envs.base import make_box_obs, normalize_altitude_km, normalize_temp
from space_env.simulation import SimulationEngine


# Высота LEO 200–1000 км, период ~90–105 мин
ALT_KM_MIN = 200.0
ALT_KM_MAX = 1000.0
TEMP_MIN = -20.0
TEMP_MAX = 50.0
DEFAULT_STEP_SEC = 60.0
DEFAULT_MAX_STEPS = 150  # ~2.5 витка по 60 с


class ResourceManagerEnv(gym.Env):
    """
    Среда: поддержание SoC и температуры в пределах за счёт выбора уровня нагрузки.
    action: 0 = номинальная нагрузка, 1 = пониженная, 2 = минимальная (только служебные).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        step_sec: float = DEFAULT_STEP_SEC,
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
    ):
        super().__init__()
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self._seed = seed
        # obs: altitude_norm, in_eclipse, battery_soc, temp_norm, time_in_episode_norm
        self.observation_space = make_box_obs((5,))
        self.action_space = gym.spaces.Discrete(3)  # 0 nominal, 1 reduced, 2 minimal
        self._engine: SimulationEngine | None = None
        self._steps = 0

    def _build_engine(self, seed: int | None = None) -> SimulationEngine:
        epoch = Time("2024-01-01T00:00:00Z")
        orbit = Orbit.from_keplerian(
            6800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch
        )
        sc = Spacecraft(config=SpacecraftConfig(
            power=PowerConfig(
                solar_panel_area_m2=2.0,
                battery_capacity_Wh=200.0,
                consumption_nominal_W=15.0,
                consumption_peak_W=40.0,
            ),
            thermal=ThermalConfig(temp_min_C=TEMP_MIN, temp_max_C=TEMP_MAX),
            propulsion=PropulsionConfig(delta_v_total_m_s=50.0, mass_kg=50.0),
        ))
        sc.reset_state()
        return SimulationEngine(orbit, sc, seed=seed)

    def _obs_from_engine(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(5, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        alt_norm = normalize_altitude_km(snap.altitude_km, ALT_KM_MIN, ALT_KM_MAX)
        temp_norm = normalize_temp(snap.temp_C, TEMP_MIN, TEMP_MAX)
        time_norm = min(1.0, self._steps / max(1, self.max_episode_steps))
        return np.array(
            [
                alt_norm,
                float(snap.in_eclipse),
                snap.battery_soc,
                temp_norm,
                time_norm,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._engine = self._build_engine(seed=self.np_random.integers(0, 2**31) if seed is None else seed)
        obs = self._obs_from_engine()
        return obs, {"time_sec": 0.0}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Call reset() first")
        action = int(np.asarray(action).flat[0])
        # Установка потребления по действию: 0=nominal, 1=0.6*nominal, 2=0.3*nominal
        cfg = self._engine.spacecraft.config.power
        consumption = [cfg.consumption_nominal_W, cfg.consumption_nominal_W * 0.6, cfg.consumption_nominal_W * 0.3][
            min(action, 2)
        ]
        self._engine.spacecraft.state.power.consumption_W = consumption
        snap, events = self._engine.step(self.step_sec)
        self._steps += 1

        # Награда: штраф за нарушение, малый бонус за шаг в норме
        reward = 0.0
        terminated = False
        if snap.battery_soc < 0.2:
            reward -= 1.0
            terminated = True
        if snap.temp_C < TEMP_MIN or snap.temp_C > TEMP_MAX:
            reward -= 0.5
            terminated = True
        if not terminated:
            reward += 0.01  # выживание
        truncated = self._steps >= self.max_episode_steps
        if truncated and not terminated:
            reward += 0.1  # дошли до конца эпизода

        obs = self._obs_from_engine()
        info = {"time_sec": snap.time_sec, "battery_soc": snap.battery_soc, "in_eclipse": snap.in_eclipse}
        return obs, float(reward), terminated, truncated, info
