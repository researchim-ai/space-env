"""
ResourceManagerEnv: уровень нагрузки так, чтобы пережить тень и не выйти за температуру.
Нагреватели включаются сами у нижней границы и сажают батарею.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.spacecraft import ThermalConfig
from space_env.envs.base import format_ansi, make_box_obs, make_leo_engine, normalize_altitude_km, normalize_temp
from space_env.mission.constraints import Constraints
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

ALT_KM_MIN = 200.0
ALT_KM_MAX = 1000.0
TEMP_MIN = -20.0
TEMP_MAX = 50.0
DEFAULT_STEP_SEC = 60.0
DEFAULT_MAX_STEPS = 150
_COMMANDS = {0: "nominal_power", 1: "reduced_power", 2: "minimal_power"}


class ResourceManagerEnv(gym.Env):
    """
    Наблюдение: высота, тень, SoC, температура, доля эпизода.
    Действия: 0 номинал, 1 пониженная нагрузка, 2 минимальная.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        step_sec: float = DEFAULT_STEP_SEC,
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._seed = seed
        self.observation_space = make_box_obs((5,))
        self.action_space = gym.spaces.Discrete(3)
        self._engine: SimulationEngine | None = None
        self._steps = 0
        self._weights = load_rewards()["resource_manager"]
        self._last_reward = 0.0

    def _build_engine(self, seed: int | None) -> SimulationEngine:
        return make_leo_engine(
            seed=seed,
            thermal=ThermalConfig(temp_min_C=TEMP_MIN, temp_max_C=TEMP_MAX),
            constraints=Constraints(battery_soc_min=0.2, temp_min_C=TEMP_MIN, temp_max_C=TEMP_MAX),
        )

    def _obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(5, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        return np.array(
            [
                normalize_altitude_km(snap.altitude_km, ALT_KM_MIN, ALT_KM_MAX),
                float(snap.in_eclipse),
                snap.battery_soc,
                normalize_temp(snap.temp_C, TEMP_MIN, TEMP_MAX),
                min(1.0, self._steps / max(1, self.max_episode_steps)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._last_reward = 0.0
        used_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        self._engine = self._build_engine(used_seed)
        return self._obs(), {"time_sec": 0.0}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = int(np.asarray(action).flat[0])
        command = _COMMANDS.get(action, "nominal_power")
        snap, _events = self._engine.step(self.step_sec, command)
        weights = self._weights
        reward = 0.0
        terminated = False
        if snap.battery_soc < 0.2:
            reward -= weights["soc_failure"]
            terminated = True
        if snap.temp_C < TEMP_MIN or snap.temp_C > TEMP_MAX:
            reward -= weights["thermal_failure"]
            terminated = True
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        if not terminated:
            reward += weights["survival_step"]
        if truncated and not terminated:
            reward += weights["episode_complete"]
        self._last_reward = float(reward)
        info = {
            "time_sec": snap.time_sec,
            "battery_soc": snap.battery_soc,
            "temp_C": snap.temp_C,
            "in_eclipse": snap.in_eclipse,
            "heater_on": snap.heater_on,
            "success": truncated and not terminated,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._engine is None:
            return None
        snap = self._engine.get_state_snapshot()
        return format_ansi(
            "ResourceManager",
            [
                ("t, с", f"{snap.time_sec:.0f}"),
                ("тень", snap.shadow),
                ("SoC", f"{snap.battery_soc:.2f}"),
                ("T, °C", f"{snap.temp_C:.1f}"),
                ("нагреватель", "вкл" if snap.heater_on else "выкл"),
                ("нагрузка, Вт", f"{self._engine.spacecraft.state.power.consumption_W:.1f}"),
            ],
        )
