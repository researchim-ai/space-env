"""
Группировка на общих часах: у каждого свой буфер.
Пакет идёт по цепочке к аппарату с наибольшим углом места над Фэрбанксом.
Один шаг — один луч. Линия та же, что у цепочки 1.5°: 1 Вт и 6 дБи, луч не через Землю.
Фидера Iridium здесь нет.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.frames import elevation_deg
from space_env.core.ground import FAIRBANKS
from space_env.envs.base import format_ansi
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet

_COMMANDS = {0: "wait", 1: "imaging", 2: "relay", 3: "comms"}
_PACKET_MB = 8.0


class ConstellationOpsEnv(gym.Env):
    """
    Действие MultiDiscrete([число аппаратов, 4]): кому и что.
    Команды: ждать, снять кадр, переслать соседу в сторону шлюзового аппарата, сдать на станцию.
    Шлюзовой аппарат — тот, у кого выше угол места над Фэрбанксом. Это геометрия, не радиолиния фидера.
    Наблюдение на аппарат: заряд, доля пакета, угол места, контакт. Дальше — ошибка строя и максимальная Pc.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        count: int = 4,
        separation_deg: float = 1.5,
        step_sec: float = 60.0,
        max_episode_steps: int = 12,
        render_mode: str | None = None,
    ):
        super().__init__()
        if count < 3:
            raise ValueError("в операционной группировке минимум три аппарата")
        self.count = count
        self.separation_deg = separation_deg
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(count * 4 + 2,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([count, 4])
        self._fleet: Fleet | None = None
        self._names: list[str] = []
        self._gateway = 0
        self._design_range_m = 1.0
        self._steps = 0

    def _elevations_deg(self) -> list[float]:
        assert self._fleet is not None
        values = []
        for name in self._names:
            engine = self._fleet.vehicles[name]
            values.append(
                elevation_deg(
                    engine.orbit.r,
                    FAIRBANKS.lat_deg,
                    FAIRBANKS.lon_deg,
                    FAIRBANKS.alt_m,
                    engine.orbit.epoch,
                )
            )
        return values

    def _adjacent_range_m(self) -> list[float]:
        assert self._fleet is not None
        ranges = []
        for left, right in zip(self._names, self._names[1:]):
            pair = next(
                item
                for item in self._fleet.pairs()
                if {item.leader, item.follower} == {left, right}
            )
            ranges.append(pair.range_m)
        return ranges

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._fleet = Fleet.string_of_pearls(
            self.count,
            self.separation_deg,
            scenario=ScenarioConfig(ground_stations=[]),
        )
        self._names = list(self._fleet.vehicles)
        self._design_range_m = float(np.mean(self._adjacent_range_m()))
        elevations = self._elevations_deg()
        self._gateway = int(np.argmax(elevations))
        source = int(np.argmin(elevations))
        if source == self._gateway:
            source = 0 if self._gateway != 0 else self.count - 1
        self._fleet.collect(self._names[source], _PACKET_MB)
        return self._obs(), {"gateway": self._names[self._gateway], "source": self._names[source]}

    def _obs(self) -> np.ndarray:
        if self._fleet is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        elevations = self._elevations_deg()
        values: list[float] = []
        for index, name in enumerate(self._names):
            snap = self._fleet.vehicles[name].get_state_snapshot()
            values.extend(
                [
                    snap.battery_soc,
                    float(np.clip(self._fleet.buffer_mb[name] / _PACKET_MB, 0.0, 1.0)),
                    float(np.clip((elevations[index] + 90.0) / 180.0, 0.0, 1.0)),
                    float(snap.in_contact),
                ]
            )
        ranges = self._adjacent_range_m()
        formation = max(abs(item - self._design_range_m) / self._design_range_m for item in ranges)
        screen = self._fleet.screen()
        probability = max((hit.probability for hit in screen), default=0.0)
        values.append(float(np.clip(formation, 0.0, 1.0)))
        values.append(float(np.clip(probability / 1e-4, 0.0, 1.0)))
        return np.asarray(values, dtype=np.float32)

    def _neighbor_toward_gateway(self, index: int) -> int | None:
        if index == self._gateway:
            return None
        return index + (1 if self._gateway > index else -1)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._fleet is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = np.asarray(action, dtype=int).ravel()
        index = int(np.clip(action[0], 0, self.count - 1))
        command = _COMMANDS.get(int(action[1]), "wait")
        name = self._names[index]
        route: list[dict] = []
        moved = 0.0
        if command == "imaging":
            snap = self._fleet.vehicles[name].get_state_snapshot()
            self._fleet.step(self.step_sec, {name: "imaging"})
            if (not snap.in_eclipse) and snap.pointing == "nadir" and snap.slew_remaining_deg <= 1.0:
                self._fleet.collect(name, _PACKET_MB)
        elif command == "comms":
            self._fleet.step(self.step_sec, {name: "comms"})
            self._fleet.downlink(name, self.step_sec)
        elif command == "relay":
            self._fleet.step(self.step_sec)
            nxt = self._neighbor_toward_gateway(index)
            if nxt is not None:
                moved = self._fleet.relay(name, self._names[nxt], self.step_sec)
                if moved > 0.0:
                    route.append({"from": name, "to": self._names[nxt]})
        else:
            self._fleet.step(self.step_sec)
        gateway_name = self._names[self._gateway]
        arrived = self._fleet.buffer_mb[gateway_name] >= _PACKET_MB - 1e-6
        ranges = self._adjacent_range_m()
        formation = max(abs(item - self._design_range_m) / self._design_range_m for item in ranges)
        reward = moved / _PACKET_MB if route and route[0]["to"] == gateway_name else 0.0
        if formation > 0.05:
            reward -= 1.0
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        info = {
            "gateway": gateway_name,
            "buffer_mb": dict(self._fleet.buffer_mb),
            "route": route,
            "formation_error": formation,
            "success": arrived and formation <= 0.05,
        }
        return self._obs(), float(reward), False, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._fleet is None:
            return None
        rows = [(name, f"{self._fleet.buffer_mb[name]:.1f} Мбит") for name in self._names]
        rows.append(("шлюз", self._names[self._gateway]))
        return format_ansi("Constellation ops", rows)
