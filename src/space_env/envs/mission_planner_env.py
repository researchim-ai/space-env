"""
MissionPlannerEnv: выбор следующей задачи (съёмка, сессия связи, маневр, ожидание).
Награда за выполнение целей, штраф за перерасход Δv и пропуск возможностей.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core import Orbit, Time
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, PowerConfig, PropulsionConfig
from space_env.envs.base import make_box_obs, normalize_altitude_km
from space_env.mission import MissionGoal
from space_env.simulation import SimulationEngine

DEFAULT_STEP_SEC = 120.0
DEFAULT_MAX_STEPS = 100
ALT_KM_MIN = 200.0
ALT_KM_MAX = 1000.0


class MissionPlannerEnv(gym.Env):
    """
    Действия: 0 = wait, 1 = imaging (если на свету), 2 = comms (сессия), 3 = maneuver (тратим Δv).
    Цели: imaging_count >= goal_imaging, comms_sessions >= goal_comms, не превысить delta_v_budget.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mission_goal: MissionGoal | None = None,
        step_sec: float = DEFAULT_STEP_SEC,
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
    ):
        super().__init__()
        self.mission_goal = mission_goal or MissionGoal(imaging_count=5, comms_sessions_min=2, delta_v_budget_m_s=20.0)
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self._seed = seed
        # obs: alt_norm, in_eclipse, soc, delta_v_remaining_norm, imaging_done_norm, comms_done_norm
        self.observation_space = make_box_obs((6,))
        self.action_space = gym.spaces.Discrete(4)  # wait, imaging, comms, maneuver
        self._engine: SimulationEngine | None = None
        self._steps = 0
        self._imaging_done = 0
        self._comms_done = 0
        self._delta_v_total = 30.0  # для нормировки

    def _build_engine(self, seed: int | None = None) -> SimulationEngine:
        epoch = Time("2024-01-01T00:00:00Z")
        orbit = Orbit.from_keplerian(6800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
        sc = Spacecraft(config=SpacecraftConfig(
            power=PowerConfig(solar_panel_area_m2=2.0, battery_capacity_Wh=300.0, consumption_nominal_W=10.0, consumption_peak_W=25.0),
            propulsion=PropulsionConfig(delta_v_total_m_s=self._delta_v_total, mass_kg=50.0),
        ))
        sc.reset_state()
        return SimulationEngine(orbit, sc, mission_goal=self.mission_goal, seed=seed)

    def _obs_from_engine(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(6, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        alt_norm = normalize_altitude_km(snap.altitude_km, ALT_KM_MIN, ALT_KM_MAX)
        dv_norm = snap.delta_v_remaining_m_s / max(1e-6, self._delta_v_total)
        goal_img = self.mission_goal.imaging_count or 1
        goal_com = self.mission_goal.comms_sessions_min or 1
        return np.array(
            [
                alt_norm,
                float(snap.in_eclipse),
                snap.battery_soc,
                np.clip(dv_norm, 0.0, 1.0),
                min(1.0, self._imaging_done / goal_img),
                min(1.0, self._comms_done / goal_com),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._imaging_done = 0
        self._comms_done = 0
        self._engine = self._build_engine(seed=self.np_random.integers(0, 2**31) if seed is None else seed)
        return self._obs_from_engine(), {"imaging": 0, "comms": 0}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Call reset() first")
        action = int(np.asarray(action).flat[0])
        snap_before = self._engine.get_state_snapshot()
        snap, events = self._engine.step(self.step_sec)
        reward = 0.0
        # Обработка действий
        if action == 1 and not snap_before.in_eclipse:
            self._imaging_done += 1
            reward += 0.2
        if action == 2:
            # Упрощённо: "сессия связи" возможна с вероятностью (имитация видимости)
            if self.np_random.random() < 0.3:
                self._comms_done += 1
                reward += 0.2
        if action == 3:
            dv_use = 5.0
            self._engine.spacecraft.state.propulsion.delta_v_remaining_m_s = max(
                0.0,
                self._engine.spacecraft.state.propulsion.delta_v_remaining_m_s - dv_use,
            )
            reward -= 0.05
        # Штраф за перерасход Δv
        if self.mission_goal.delta_v_budget_m_s is not None:
            used = self._delta_v_total - snap.delta_v_remaining_m_s
            if used > self.mission_goal.delta_v_budget_m_s:
                reward -= 0.2
        terminated = False
        goal_img = self.mission_goal.imaging_count or 0
        goal_com = self.mission_goal.comms_sessions_min or 0
        if goal_img and self._imaging_done >= goal_img and goal_com and self._comms_done >= goal_com:
            reward += 1.0
            terminated = True
        truncated = self._steps >= self.max_episode_steps
        self._steps += 1
        obs = self._obs_from_engine()
        info = {"imaging": self._imaging_done, "comms": self._comms_done, "time_sec": snap.time_sec}
        return obs, float(reward), terminated, truncated, info
