"""
AnomalyResponseEnv: отказ виден только по симптомам.
Перегрузка шины снимается сбросом нагрузки, залипший нагреватель — safe mode,
защёлка радио — перезагрузкой в несколько шагов. Перезагрузка не возвращает питание.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.spacecraft import PowerConfig, PropulsionConfig, ThermalConfig
from space_env.envs.base import format_ansi, make_leo_engine, normalize_temp
from space_env.mission.scenarios import ScenarioConfig, ScenarioType
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

_FAULTS = ("overload", "stuck_heater", "radio_latch")
_COMMANDS = {0: "wait", 1: "safe_mode", 2: "reduced_power", 3: "reboot"}
TEMP_MIN = -20.0
TEMP_MAX = 50.0


class AnomalyResponseEnv(gym.Env):
    """
    Наблюдение: SoC, температура, напряжение шины, готовность радио, скорость изменения SoC.
    Типа отказа в векторе нет.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        max_episode_steps: int = 20,
        step_sec: float = 60.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.step_sec = step_sec
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._engine: SimulationEngine | None = None
        self._steps = 0
        self._fault = "overload"
        self._weights = load_rewards()["anomaly"]
        self._last_reward = 0.0

    def _obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(6, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        voltage_norm = float(np.clip((snap.bus_voltage_V - 22.0) / 12.0, 0.0, 1.0))
        rate_norm = float(np.clip(0.5 + snap.soc_rate_per_s * 400.0, 0.0, 1.0))
        return np.array(
            [
                snap.battery_soc,
                normalize_temp(snap.temp_C, TEMP_MIN, TEMP_MAX),
                voltage_norm,
                1.0 if self._engine.radio_ready() else 0.0,
                rate_norm,
                0.0 if snap.in_eclipse else 1.0,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._last_reward = 0.0
        self._fault = _FAULTS[int(self.np_random.integers(0, 3))]
        soc0 = float(self.np_random.uniform(0.55, 0.8))
        scenario = ScenarioConfig(
            scenario_type=ScenarioType.ANOMALY,
            initial_battery_soc=soc0,
            ground_stations=[],
        )
        used_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        self._engine = make_leo_engine(
            seed=used_seed,
            scenario=scenario,
            initial_temp_c=15.0,
            power=PowerConfig(
                solar_panel_area_m2=1.2,
                battery_capacity_Wh=140.0,
                consumption_nominal_W=40.0,
                consumption_peak_W=120.0,
            ),
            thermal=ThermalConfig(
                temp_min_C=TEMP_MIN,
                temp_max_C=TEMP_MAX,
                heater_power_W=0.0,
                heat_capacity_J_K=18_000.0,
            ),
            propulsion=PropulsionConfig(delta_v_total_m_s=20.0, mass_kg=100.0),
        )
        self._engine.arm_fault(self._fault)
        return self._obs(), {}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = int(np.asarray(action).flat[0])
        weights = self._weights
        reward = -weights["step_cost"]
        command = _COMMANDS.get(action, "wait")
        snap, _events = self._engine.step(self.step_sec, command)
        clear = not self._engine.faults_active()
        healthy = clear and snap.battery_soc >= 0.2 and TEMP_MIN <= snap.temp_C <= TEMP_MAX
        if healthy:
            reward += weights["recovered"]
        if snap.battery_soc < 0.12:
            reward -= weights["soc_failure"]
        if snap.temp_C < TEMP_MIN or snap.temp_C > TEMP_MAX:
            reward -= weights["thermal_failure"]
        self._steps += 1
        terminated = healthy or snap.battery_soc < 0.12 or snap.temp_C < TEMP_MIN or snap.temp_C > TEMP_MAX
        truncated = self._steps >= self.max_episode_steps
        self._last_reward = float(reward)
        info = {
            "recovered": healthy,
            "temp_C": snap.temp_C,
            "radio_ready": self._engine.radio_ready(),
            "success": healthy,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._engine is None:
            return None
        snap = self._engine.get_state_snapshot()
        return format_ansi(
            "AnomalyResponse",
            [
                ("SoC", f"{snap.battery_soc:.2f}"),
                ("U, В", f"{snap.bus_voltage_V:.1f}"),
                ("T, °C", f"{snap.temp_C:.1f}"),
                ("радио", "готово" if self._engine.radio_ready() else "нет"),
                ("dSoC/dt", f"{snap.soc_rate_per_s:.5f}"),
            ],
        )
