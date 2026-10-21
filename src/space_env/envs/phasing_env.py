"""
Удержание пары Sentinel-2 на 180°.
В начале у ведомого добавлен проградный импульс 0.5 м/с: большая полуось выше, он отстаёт.
Импульс не списывается с бюджета. Бюджет остаётся оценкой Циолковского по 123 кг топлива.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.build import sentinel2_design_fleet
from space_env.catalog.phasing import add_prograde_dv, phase_error_m
from space_env.envs.base import format_ansi
from space_env.simulation.fleet import Fleet

_INJECT_DV_M_S = 0.5
_BURN_DV_M_S = 0.5
_FOLLOWER = "Sentinel-2B"
_LEADER = "Sentinel-2A"


class SentinelPhasingEnv(gym.Env):
    """
    Действия: ждать, +0.5 м/с по скорости ведомого, −0.5 м/с.
    Один встречный импульс гасит ошибку большой полуоси. Лишние импульсы разгоняют дрейф снова.
    Наблюдение: ошибка антипода в долях 20 км, скорость ошибки, остаток Δv, доля эпизода.
    Награда — минус модуль ошибки в километрах. Успех — к концу дрейф почти остановлен.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        step_sec: float = 300.0,
        max_episode_steps: int = 36,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._fleet: Fleet | None = None
        self._error_m = 0.0
        self._rate_m_s = 0.0
        self._steps = 0

    def _measure(self) -> float:
        fleet = self._fleet
        if fleet is None:
            return 0.0
        leader = fleet.vehicles[_LEADER]
        follower = fleet.vehicles[_FOLLOWER]
        return phase_error_m(leader.orbit.r, leader.orbit.v, follower.orbit.r)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._fleet = sentinel2_design_fleet()
        add_prograde_dv(self._fleet.vehicles[_FOLLOWER], _INJECT_DV_M_S)
        self._error_m = self._measure()
        self._rate_m_s = 0.0
        return self._obs(), {"inject_dv_m_s": _INJECT_DV_M_S}

    def _obs(self) -> np.ndarray:
        if self._fleet is None:
            return np.zeros(4, dtype=np.float32)
        remaining = self._fleet.vehicles[_FOLLOWER].spacecraft.state.propulsion.delta_v_remaining_m_s
        total = self._fleet.vehicles[_FOLLOWER].spacecraft.config.propulsion.delta_v_total_m_s
        return np.asarray(
            [
                float(np.clip(self._error_m / 20_000.0, -1.0, 1.0)),
                float(np.clip(self._rate_m_s / 5.0, -1.0, 1.0)),
                float(np.clip(remaining / total, 0.0, 1.0)) if total > 0.0 else 0.0,
                float(self._steps / self.max_episode_steps),
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._fleet is None:
            raise RuntimeError("Сначала вызовите reset()")
        commands: dict[str, str] = {}
        payloads: dict[str, dict] = {}
        if int(action) == 1:
            commands[_FOLLOWER] = "maneuver"
            payloads[_FOLLOWER] = {"dv_m_s": _BURN_DV_M_S, "direction": "prograde"}
        elif int(action) == 2:
            commands[_FOLLOWER] = "maneuver"
            payloads[_FOLLOWER] = {"dv_m_s": _BURN_DV_M_S, "direction": "retrograde"}
        before = self._error_m
        self._fleet.step(self.step_sec, commands or None, payloads or None)
        self._error_m = self._measure()
        self._rate_m_s = (self._error_m - before) / self.step_sec
        reward = -abs(self._error_m) / 1000.0
        self._steps += 1
        terminated = False
        truncated = self._steps >= self.max_episode_steps
        info = {
            "error_m": self._error_m,
            "rate_m_s": self._rate_m_s,
            "success": truncated and abs(self._rate_m_s) < 0.2,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._fleet is None:
            return None
        rows = [("ошибка антипода, км", f"{self._error_m / 1000.0:.2f}"), ("дрейф, м/с", f"{self._rate_m_s:.3f}")]
        return format_ansi("Sentinel-2 phasing", rows)
