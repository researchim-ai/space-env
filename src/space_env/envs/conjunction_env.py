"""
ConjunctionAssessmentEnv: маневр или ожидание по прогнозируемому промаху.
Промах и Pc считаются из относительного движения и приближения Чена.
Скрытой метки «опасный эпизод» нет: порог — это Pc.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.spacecraft import PropulsionConfig
from space_env.envs.base import format_ansi, make_leo_engine
from space_env.mission.scenarios import ConjunctionInject, ScenarioConfig, ScenarioType
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

PC_THRESHOLD = 1.0e-4
_DIRECTIONS = {1: "prograde", 2: "normal"}


class ConjunctionAssessmentEnv(gym.Env):
    """
    Наблюдение: радиальный промах, боковой промах, Pc, остаток Δv, время до TCA.
    Действия: 0 ждать, 1 вдоль трека 1 м/с, 2 по нормали 1 м/с.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        max_episode_steps: int = 16,
        step_sec: float = 60.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.step_sec = step_sec
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._engine: SimulationEngine | None = None
        self._steps = 0
        self._dv_budget = 10.0
        self._burn_dv = 1.0
        self._initial_pc = 0.0
        self._spent = 0.0
        self._weights = load_rewards()["conjunction"]
        self._last_reward = 0.0

    def _obs(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(5, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        radial = 0.0 if snap.miss_radial_m is None else snap.miss_radial_m
        cross = 0.0 if snap.miss_cross_m is None else snap.miss_cross_m
        prob = 0.0 if snap.conjunction_probability is None else snap.conjunction_probability
        remaining = snap.delta_v_remaining_m_s / max(self._dv_budget, 1e-6)
        tca_norm = 0.0 if snap.conjunction_tca_passed else float(np.clip(1.0 - self._steps / self.max_episode_steps, 0.0, 1.0))
        return np.array(
            [
                float(np.clip(0.5 + radial / 10_000.0, 0.0, 1.0)),
                float(np.clip(0.5 + cross / 10_000.0, 0.0, 1.0)),
                float(np.clip(prob / 1.0e-3, 0.0, 1.0)),
                float(np.clip(remaining, 0.0, 1.0)),
                tca_norm,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._spent = 0.0
        self._last_reward = 0.0
        miss_m = float(np.exp(self.np_random.uniform(np.log(40.0), np.log(4000.0))))
        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        tca = float(self.np_random.uniform(420.0, 720.0))
        scenario = ScenarioConfig(
            scenario_type=ScenarioType.CONJUNCTION,
            conjunction_injects=[
                ConjunctionInject(
                    at_sec=0.0,
                    miss_distance_km=max(miss_m / 1000.0, 0.01),
                    miss_radial_m=miss_m * np.cos(angle),
                    miss_cross_m=miss_m * np.sin(angle),
                    tca_in_sec=tca,
                    approach_m_s=120.0,
                    hard_body_radius_m=15.0,
                    combined_sigma_m=200.0,
                )
            ],
            ground_stations=[],
        )
        used_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        self._engine = make_leo_engine(
            seed=used_seed,
            scenario=scenario,
            propulsion=PropulsionConfig(delta_v_total_m_s=self._dv_budget, mass_kg=100.0),
        )
        snap = self._engine.get_state_snapshot()
        self._initial_pc = float(snap.conjunction_probability or 0.0)
        return self._obs(), {
            "miss_km": snap.conjunction_miss_km,
            "prob": self._initial_pc,
            "above_threshold": self._initial_pc >= PC_THRESHOLD,
        }

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = int(np.asarray(action).flat[0])
        weights = self._weights
        reward = 0.0
        command = None
        payload: dict = {}
        direction = _DIRECTIONS.get(action)
        if direction is not None:
            command = "maneuver"
            payload = {"dv_m_s": self._burn_dv, "direction": direction}
        snap, _events = self._engine.step(self.step_sec, command, payload)
        if command == "maneuver" and self._engine.last_command_ok:
            self._spent += self._burn_dv
            reward -= weights["delta_v_cost"]
        self._steps += 1
        terminated = snap.conjunction_tca_passed
        truncated = self._steps >= self.max_episode_steps
        prob = float(snap.conjunction_probability or 0.0)
        if terminated or truncated:
            needed = self._initial_pc >= PC_THRESHOLD
            safe_now = prob < PC_THRESHOLD
            if needed and safe_now:
                reward += weights["avoided"]
            elif needed and not safe_now:
                reward -= weights["collision_risk"]
            elif not needed and self._spent > 0.0:
                reward -= weights["unnecessary_burn"]
            elif not needed:
                reward += weights["avoided"]
        self._last_reward = float(reward)
        info = {
            "miss_km": snap.conjunction_miss_km,
            "prob": prob,
            "initial_pc": self._initial_pc,
            "spent_dv": self._spent,
            "success": (terminated or truncated)
            and (
                (self._initial_pc >= PC_THRESHOLD and prob < PC_THRESHOLD)
                or (self._initial_pc < PC_THRESHOLD and self._spent == 0.0)
            ),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._engine is None:
            return None
        snap = self._engine.get_state_snapshot()
        return format_ansi(
            "Conjunction",
            [
                ("промах, км", f"{snap.conjunction_miss_km or 0:.3f}"),
                ("Pc", f"{snap.conjunction_probability or 0:.3e}"),
                ("радиус, м", f"{snap.miss_radial_m or 0:.0f}"),
                ("нормаль, м", f"{snap.miss_cross_m or 0:.0f}"),
                ("TCA", "пройдена" if snap.conjunction_tca_passed else "впереди"),
            ],
        )
