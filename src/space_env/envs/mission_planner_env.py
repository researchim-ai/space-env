"""
MissionPlannerEnv: съёмка, сеанс связи, манёвр или ожидание.
Съёмка только на свету и в надир, связь только в зоне станции.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.access import ground_track_latlon
from space_env.core.orbit import Orbit
from space_env.core.time import Time
from space_env.envs.base import format_ansi, make_box_obs, make_leo_engine, normalize_altitude_km
from space_env.mission.goals import MissionGoal
from space_env.mission.scenarios import GroundTarget, ScenarioConfig
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

DEFAULT_STEP_SEC = 300.0
DEFAULT_MAX_STEPS = 288
ALT_KM_MIN = 200.0
ALT_KM_MAX = 1000.0
_COMMANDS = {0: "wait", 1: "imaging", 2: "comms", 3: "maneuver"}


class MissionPlannerEnv(gym.Env):
    """
    Наблюдение: высота, тень, SoC, остаток Δv, прогресс съёмки, прогресс связи, контакт.
    Действия: 0 ожидание, 1 съёмка, 2 связь, 3 манёвр (5 м/с, проград).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        mission_goal: MissionGoal | None = None,
        step_sec: float = DEFAULT_STEP_SEC,
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.mission_goal = mission_goal or MissionGoal(
            imaging_count=3,
            comms_sessions_min=1,
            delta_v_budget_m_s=20.0,
        )
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._seed = seed
        self.observation_space = make_box_obs((9,))
        self._alt0_km = 400.0
        self._maintain_km = 399.0
        self.action_space = gym.spaces.Discrete(4)
        self._engine: SimulationEngine | None = None
        self._steps = 0
        self._delta_v_total = 80.0
        self._weights = load_rewards()["mission_planner"]
        self._last_reward = 0.0

    def _build_engine(self, seed: int | None) -> SimulationEngine:
        epoch = "2024-01-01T00:00:00Z"
        orbit = Orbit.from_keplerian(6_671_000.0, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, Time(epoch))
        # Три цели на будущей трассе: снимок засчитывается только в поле зрения и на свету.
        points = ground_track_latlon(orbit, [1500.0, 3300.0, 5100.0])
        targets = [GroundTarget(name=f"T{index + 1}", lat_deg=lat, lon_deg=lon) for index, (lat, lon) in enumerate(points)]
        return make_leo_engine(
            epoch=epoch,
            a_m=6_671_000.0,
            seed=seed,
            mission_goal=self.mission_goal,
            scenario=ScenarioConfig(ground_targets=targets),
            propulsion=None,
        )

    def _obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(9, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        goal_img = self.mission_goal.imaging_count or 1
        goal_com = self.mission_goal.comms_sessions_min or 1
        dv_norm = snap.delta_v_remaining_m_s / max(1e-6, self._delta_v_total)
        # Канал удержания: 0, когда высота просела на 2 км относительно старта.
        hold = float(np.clip((snap.altitude_km - (self._alt0_km - 2.0)) / 4.0, 0.0, 1.0))
        return np.array(
            [
                normalize_altitude_km(snap.altitude_km, ALT_KM_MIN, ALT_KM_MAX),
                float(snap.in_eclipse),
                snap.battery_soc,
                float(np.clip(dv_norm, 0.0, 1.0)),
                min(1.0, snap.imaging_count / goal_img),
                min(1.0, snap.comms_sessions / goal_com),
                float(snap.in_contact),
                hold,
                1.0 if snap.seconds_to_aos is None else float(np.clip(snap.seconds_to_aos / (6.0 * 3600.0), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._last_reward = 0.0
        used_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        self._engine = self._build_engine(used_seed)
        snap = self._engine.get_state_snapshot()
        self._alt0_km = snap.altitude_km
        self._maintain_km = snap.altitude_km - 0.6
        return self._obs(), {"imaging": snap.imaging_count, "comms": snap.comms_sessions}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = int(np.asarray(action).flat[0])
        command = _COMMANDS.get(action, "wait")
        payload = {"dv_m_s": 0.5, "direction": "prograde"} if command == "maneuver" else {}
        snap, _events = self._engine.step(self.step_sec, command, payload)
        weights = self._weights
        reward = 0.0
        if command == "imaging":
            reward += weights["imaging_success"] if self._engine.last_command_ok else -weights["invalid_action"]
        elif command == "comms":
            reward += weights["comms_success"] if self._engine.last_command_ok else -weights["invalid_action"]
        elif command == "maneuver":
            if not self._engine.last_command_ok:
                reward -= weights["invalid_action"]
            elif snap.altitude_km < self._maintain_km:
                reward += 0.4
            else:
                reward -= weights["maneuver_cost"]
        if snap.altitude_km < self._alt0_km - 2.0:
            reward -= 0.5
        if self.mission_goal.delta_v_budget_m_s is not None and snap.delta_v_used_m_s > self.mission_goal.delta_v_budget_m_s:
            reward -= weights["delta_v_overrun"]
        goal_img = self.mission_goal.imaging_count or 0
        goal_com = self.mission_goal.comms_sessions_min or 0
        goals_met = bool(goal_img and snap.imaging_count >= goal_img and goal_com and snap.comms_sessions >= goal_com)
        terminated = False
        if goals_met:
            reward += weights["goal_complete"]
            terminated = True
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        self._last_reward = float(reward)
        info = {
            "imaging": snap.imaging_count,
            "comms": snap.comms_sessions,
            "time_sec": snap.time_sec,
            "in_contact": snap.in_contact,
            "command_ok": self._engine.last_command_ok,
            "success": terminated,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._engine is None:
            return None
        snap = self._engine.get_state_snapshot()
        return format_ansi(
            "MissionPlanner",
            [
                ("t, с", f"{snap.time_sec:.0f}"),
                ("высота, км", f"{snap.altitude_km:.1f}"),
                ("тень", snap.shadow),
                ("SoC", f"{snap.battery_soc:.2f}"),
                ("съёмки", str(snap.imaging_count)),
                ("сеансы", str(snap.comms_sessions)),
                ("контакт", snap.contact_station if snap.in_contact else "нет"),
                ("награда", f"{self._last_reward:.3f}"),
            ],
        )
