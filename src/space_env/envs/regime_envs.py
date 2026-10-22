"""
Классы, у которых решение другое, чем у оптики на низкой орбите.
Радар снимает в тени и платит за это мощностью.
Геостационарный аппарат удерживает большую полуось, а не ловит сеанс.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine
from space_env.catalog.geometry import geosynchronous_semimajor_axis_m
from space_env.core.access import subsatellite_latlon
from space_env.core.constants import MU_EARTH
from space_env.core.orbit import Orbit
from space_env.mission.scenarios import GroundTarget
from space_env.rl.rewards import load_rewards


def _semimajor_m(position: np.ndarray, velocity: np.ndarray) -> float:
    radius = float(np.linalg.norm(position))
    speed_sq = float(np.dot(velocity, velocity))
    return 1.0 / (2.0 / radius - speed_sq / MU_EARTH)


class RadarPassEnv(gym.Env):
    """
    Наблюдение: тень, заряд, сцена ждёт.
    Действия: 0 ждать, 1 включить радар.
    Успех — кадр в тени и заряд после импульса режима не ниже 0.2.
    Оптика в той же тени кадр не пишет: у неё images_in_eclipse выключен.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._engine = None
        self._weights = load_rewards()["radar_pass"]

    def _obs(self) -> np.ndarray:
        snap = self._engine.get_state_snapshot()
        return np.array([float(snap.in_eclipse), snap.battery_soc, 1.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._engine = make_engine("sar-small", epoch="2026-10-20T00:00:00Z", nu_deg=0.0, with_stations=False)
        self._engine.spacecraft.state.power.battery_soc = 0.85
        for _ in range(24):
            snap = self._engine.get_state_snapshot()
            if snap.in_eclipse:
                break
            self._engine.step(300.0, "wait")
        lat, lon = subsatellite_latlon(self._engine.orbit.r, self._engine.now())
        self._engine.scenario.ground_targets = [GroundTarget(name="scene", lat_deg=lat, lon_deg=lon)]
        snap = self._engine.get_state_snapshot()
        return self._obs(), {"eclipse": bool(snap.in_eclipse)}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        before = self._engine.spacecraft.state.power.battery_soc
        eclipse = bool(self._engine.get_state_snapshot().in_eclipse)
        if choice == 1:
            self._engine.step(120.0, "imaging")
        else:
            self._engine.step(120.0, "wait")
        snap = self._engine.get_state_snapshot()
        captured = snap.imaging_count > 0
        alive = snap.battery_soc >= 0.2
        success = eclipse and choice == 1 and captured and alive and snap.battery_soc < before
        reward = self._weights["scene"] if success else -self._weights["missed"]
        info = {
            "success": success,
            "eclipse": eclipse,
            "captured": captured,
            "soc": snap.battery_soc,
        }
        return self._obs(), float(reward), True, False, info


class LongitudeKeepEnv(gym.Env):
    """
    Наблюдение: лишние километры большой полуоси, нужен импульс.
    Действия: 0 ждать, 1 импульс  против скорости на величину Δa·v/(2a).
    Успех — через час большая полуось ближе к геосинхронной, чем у прогона без импульса.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._engine = None
        self._target_m = geosynchronous_semimajor_axis_m()
        self._weights = load_rewards()["longitude_keep"]

    def _raise(self) -> None:
        elements = self._engine.orbit.keplerian()
        self._engine.orbit = Orbit.from_keplerian(
            self._target_m + 2_000.0,
            elements.ecc,
            elements.inc,
            elements.raan,
            elements.argp,
            elements.nu,
            self._engine.orbit.epoch,
        )

    def _obs(self) -> np.ndarray:
        extra_km = (_semimajor_m(self._engine.orbit.r, self._engine.orbit.v) - self._target_m) / 1_000.0
        return np.array([float(np.clip(extra_km / 5.0, 0.0, 1.0)), 1.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._engine = make_engine("geo-comms", epoch="2026-10-20T00:00:00Z", nu_deg=0.0, with_stations=False)
        self._raise()
        return self._obs(), {"target_m": self._target_m}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        speed = math.sqrt(MU_EARTH / self._target_m)
        dv = 2_000.0 * speed / (2.0 * self._target_m)
        if choice == 1:
            self._engine.step(60.0, "maneuver", {"dv_m_s": dv, "direction": "retrograde"})
        else:
            self._engine.step(60.0, "wait")
        self._engine.step(3_600.0, "wait")
        error_m = abs(_semimajor_m(self._engine.orbit.r, self._engine.orbit.v) - self._target_m)
        success = choice == 1 and error_m < 800.0
        if choice == 0:
            success = False
        reward = self._weights["held"] if success else -self._weights["drift"]
        info = {"success": success, "error_m": error_m, "dv_m_s": dv if choice == 1 else 0.0}
        return self._obs(), float(reward), True, False, info
