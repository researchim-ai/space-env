"""
Две постановки группировки, которых не закрывают одиночный сеанс и цепочка.
Спор за одну станцию и пакет заявок на два аппарата с одним слотом памяти.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.references import reference
from space_env.core.access import off_nadir_deg, subsatellite_latlon
from space_env.core.environment import in_eclipse
from space_env.core.frames import eci_to_ecef, elevation_deg, geodetic_to_ecef
from space_env.core.link import link_margin_db
from space_env.core.sun import sun_position_m
from space_env.mission.scenarios import ScenarioConfig
from space_env.rl.rewards import load_rewards
from space_env.simulation.fleet import Fleet

_HALF_FOV_DEG = 30.0


def _visible(engine, station_lat: float, station_lon: float) -> tuple[bool, float]:
    spec = reference("cubesat-6u")
    moment = engine.orbit.epoch
    elevation = elevation_deg(engine.orbit.r, station_lat, station_lon, 0.0, moment)
    slant = float(
        np.linalg.norm(
            eci_to_ecef(engine.orbit.r, moment) - geodetic_to_ecef(station_lat, station_lon, 0.0)
        )
    )
    margin = link_margin_db(
        slant,
        frequency_hz=spec.downlink_hz,
        tx_power_w=spec.tx_power_w,
        tx_gain_dbi=spec.tx_gain_dbi,
        rx_gain_dbi=spec.rx_gain_dbi,
        system_loss_db=3.0,
        required_power_dbw=spec.required_power_dbw,
    )
    return elevation >= 5.0 and margin >= 0.0, margin


class StationContentionEnv(gym.Env):
    """
    Два аппарата над одной станцией, зеркало одно.
    Наблюдение: заполнение A и B, видимость A и B.
    Действие — кого пускать в этом шаге.
    Успех — сдаёт тот, у кого больше памяти, и его линия закрыта.
    """

    metadata = {"render_modes": []}

    def __init__(self, count: int = 2, max_episode_steps: int = 1):
        super().__init__()
        if count < 2:
            raise ValueError("для спора за станцию нужно минимум два аппарата")
        self.count = count
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(count * 2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(count)
        self._buffers = [8.0 / (index + 1) for index in range(count)]
        self._visible = [False] * count
        self._weights = load_rewards()["station_contention"]

    def _obs(self) -> np.ndarray:
        values = [item / 8.0 for item in self._buffers]
        values.extend(float(flag) for flag in self._visible)
        return np.asarray(values, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        fleet = Fleet.string_of_pearls(
            self.count,
            separation_deg=0.25,
            scenario=ScenarioConfig(ground_stations=[]),
        )
        self._names = list(fleet.vehicles)
        engines = [fleet.vehicles[name] for name in self._names]
        lat, lon = subsatellite_latlon(engines[0].orbit.r, engines[0].orbit.epoch)
        self._visible = [_visible(engine, lat, lon)[0] for engine in engines]
        self._original = [8.0 / (index + 1) for index in range(self.count)]
        self._buffers = list(self._original)
        return self._obs(), {"station_lat": lat, "visible": list(self._visible), "count": self.count}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        choice = min(max(choice, 0), self.count - 1)
        delivered = 0.0
        if self._visible[choice]:
            delivered = self._buffers[choice]
            self._buffers[choice] = 0.0
        original = list(self._original)
        visible_best = max(
            (index for index, flag in enumerate(self._visible) if flag),
            key=lambda index: original[index],
            default=None,
        )
        success = visible_best is not None and choice == visible_best and delivered > 0.0
        reward = self._weights["scheduled"] if success else -self._weights["conflict"]
        info = {"delivered_mbit": delivered, "choice": choice, "success": success}
        return self._obs(), float(reward), True, False, info


class OrderBookEnv(gym.Env):
    """
    Три заявки и два аппарата, у каждого один слот.
    Действия: 0 брать заявки с начала списка, 1 брать по приоритету.
    Успех на свету — в слотах приоритеты 3 и 2, мелкая заявка не заняла слот.
    """

    metadata = {"render_modes": []}

    def __init__(self, count: int = 2, max_episode_steps: int = 1):
        super().__init__()
        if count < 2:
            raise ValueError("в пакете заявок минимум два аппарата")
        self.count = count
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._sunlit = True
        self._weights = load_rewards()["order_book"]

    def _obs(self) -> np.ndarray:
        return np.array([1.0 if self._sunlit else 0.0, 1.0, 0.0], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        fleet = Fleet.string_of_pearls(self.count, separation_deg=50.0, scenario=ScenarioConfig(ground_stations=[]))
        self._names = list(fleet.vehicles)
        self._engines = [fleet.vehicles[name] for name in self._names]
        moment = self._engines[0].now()
        self._sunlit = not in_eclipse(self._engines[0].orbit.r, sun_position_m(moment))
        points = [subsatellite_latlon(engine.orbit.r, moment) for engine in self._engines]
        # Мелкая заявка в надире первого аппарата и идёт в списке раньше его приоритета.
        self._orders = [("LOW", 1, points[0])] + [
            (f"HIGH-{index}", 3, point) for index, point in enumerate(points)
        ]
        self._expected = {f"HIGH-{index}" for index in range(self.count)}
        return self._obs(), {"orders": [item[0] for item in self._orders], "sunlit": self._sunlit, "count": self.count}

    def _fits(self, engine, lat: float, lon: float) -> bool:
        return off_nadir_deg(engine.orbit.r, lat, lon, engine.now()) <= _HALF_FOV_DEG

    def _assign(self, by_priority: bool) -> set[str]:
        orders = list(self._orders)
        if by_priority:
            orders.sort(key=lambda item: -item[1])
        taken: set[str] = set()
        used = [False] * self.count
        for name, _priority, (lat, lon) in orders:
            for index, engine in enumerate(self._engines):
                if used[index] or not self._fits(engine, lat, lon):
                    continue
                used[index] = True
                taken.add(name)
                break
        return taken

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        by_priority = choice == 1
        if not self._sunlit:
            success = choice == 0
            reward = self._weights["filled"] if success else -self._weights["wasted"]
            info = {"stored": [], "sunlit": False, "success": success}
            return self._obs(), float(reward), True, False, info
        stored = self._assign(by_priority)
        success = stored == self._expected
        reward = self._weights["filled"] if success else -self._weights["wasted"]
        info = {"stored": sorted(stored), "sunlit": True, "success": success}
        return self._obs(), float(reward), True, False, info
