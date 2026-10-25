"""
Смена группировки на общий горизонт.
Пакет идёт одним лучом за шаг к аппарату над Фэрбанксом.
Рядом с первым аппаратом висит холодная пара: её Pc ниже 10⁻⁴, импульс на неё портит смену.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.core.frames import elevation_deg
from space_env.core.ground import FAIRBANKS
from space_env.core.relative import rtn_basis
from space_env.envs.base import format_ansi
from space_env.mission.horizon import PC_THRESHOLD
from space_env.mission.scenarios import ScenarioConfig
from space_env.rl.rewards import load_rewards
from space_env.simulation.fleet import Fleet, _make_vehicle

_COMMANDS = {0: "wait", 1: "imaging", 2: "relay", 3: "comms", 4: "maneuver"}
_PACKET_MBIT = 8.0


class FleetShiftEnv(gym.Env):
    """
    Действие MultiDiscrete([число аппаратов, 5]): кому и что.
    Команды: ждать, снять, переслать к шлюзу, связь, импульс 1 м/с по нормали.
    Успех — пакет на шлюзе, строй в допуске, импульс не потрачен на пару ниже 10⁻⁴.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        count: int = 4,
        separation_deg: float = 1.5,
        step_sec: float = 60.0,
        max_episode_steps: int = 8,
        render_mode: str | None = None,
    ):
        super().__init__()
        if count < 3:
            raise ValueError("в смене минимум три аппарата")
        self.count = count
        self.separation_deg = separation_deg
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(count * 4 + 4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([count, 5])
        self._fleet: Fleet | None = None
        self._names: list[str] = []
        self._gateway = 0
        self._design_range_m = 1.0
        self._steps = 0
        self._wasted_burn = False
        self._cold_pc = 0.0
        self._weights = load_rewards()["fleet_shift"]

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
            pair = next(item for item in self._fleet.pairs() if {item.leader, item.follower} == {left, right})
            ranges.append(pair.range_m)
        return ranges

    def _pair_pc(self, name: str) -> float:
        assert self._fleet is not None
        hits = [
            hit.probability
            for hit in self._fleet.screen(range_limit_m=20_000.0)
            if name in (hit.leader, hit.follower)
        ]
        return max(hits, default=0.0)

    def _attach_cold(self, fleet: Fleet) -> Fleet:
        host = fleet.vehicles[self._names[0]]
        basis = rtn_basis(host.orbit.r, host.orbit.v)
        position = host.orbit.r + basis @ np.array([0.0, 0.0, 4_000.0])
        cold = _make_vehicle(
            "COLD",
            6_800_000.0,
            51.6,
            0.0,
            0.0,
            host.orbit.epoch,
            ScenarioConfig(ground_stations=[]),
        )
        cold.orbit = cold.orbit.with_state(position, host.orbit.v.copy(), host.orbit.epoch)
        vehicles = dict(fleet.vehicles)
        vehicles["COLD"] = cold
        return Fleet(vehicles)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._wasted_burn = False
        chain = Fleet.string_of_pearls(
            self.count,
            self.separation_deg,
            scenario=ScenarioConfig(ground_stations=[]),
        )
        self._names = list(chain.vehicles)
        self._fleet = self._attach_cold(chain)
        self._design_range_m = float(np.mean(self._adjacent_range_m()))
        elevations = self._elevations_deg()
        self._gateway = int(np.argmax(elevations))
        source = int(np.argmin(elevations))
        if source == self._gateway:
            source = 0 if self._gateway != 0 else self.count - 1
        self._fleet.collect(self._names[source], _PACKET_MBIT)
        self._cold_pc = self._pair_pc("COLD")
        return self._obs(), {
            "gateway": self._names[self._gateway],
            "source": self._names[source],
            "cold_pc": self._cold_pc,
        }

    def _task_code(self) -> float:
        assert self._fleet is not None
        holder = int(np.argmax([self._fleet.buffer_mb[name] for name in self._names]))
        if self._fleet.buffer_mb[self._names[holder]] >= _PACKET_MBIT - 1e-6 and holder != self._gateway:
            return 1.0
        return 0.0

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
                    float(np.clip(self._fleet.buffer_mb[name] / _PACKET_MBIT, 0.0, 1.0)),
                    float(np.clip((elevations[index] + 90.0) / 180.0, 0.0, 1.0)),
                    float(snap.in_contact),
                ]
            )
        ranges = self._adjacent_range_m()
        formation = max(abs(item - self._design_range_m) / self._design_range_m for item in ranges)
        screen = self._fleet.screen(range_limit_m=20_000.0)
        probability = max((hit.probability for hit in screen), default=0.0)
        values.append(float(np.clip(formation, 0.0, 1.0)))
        values.append(float(np.clip(probability / PC_THRESHOLD, 0.0, 1.0)))
        values.append(self._task_code())
        values.append(float(np.clip(self._cold_pc / PC_THRESHOLD, 0.0, 1.0)))
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
        command = _COMMANDS.get(int(action[1]) if action.size > 1 else 0, "wait")
        name = self._names[index]
        route: list[dict] = []
        moved = 0.0
        burned_cold = False
        if command == "maneuver":
            pair_pc = self._pair_pc(name)
            burned_cold = pair_pc < PC_THRESHOLD
            if burned_cold:
                self._wasted_burn = True
            self._fleet.step(self.step_sec, {name: "maneuver"}, {name: {"dv_m_s": 1.0, "direction": "normal"}})
        elif command == "imaging":
            self._fleet.step(self.step_sec, {name: "imaging"})
        elif command == "comms":
            self._fleet.step(self.step_sec, {name: "comms"})
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
        arrived = self._fleet.buffer_mb[gateway_name] >= _PACKET_MBIT - 1e-6
        ranges = self._adjacent_range_m()
        formation = max(abs(item - self._design_range_m) / self._design_range_m for item in ranges)
        reward = moved / _PACKET_MBIT if route and route[0]["to"] == gateway_name else 0.0
        if burned_cold:
            reward -= self._weights["cold_burn"]
        if formation > 0.05:
            reward -= 1.0
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        success = arrived and formation <= 0.05 and not self._wasted_burn
        if truncated and success:
            reward += self._weights["delivered"]
        info = {
            "gateway": gateway_name,
            "buffer_mb": {item: self._fleet.buffer_mb[item] for item in self._names},
            "route": route,
            "formation_error": formation,
            "wasted_burn": self._wasted_burn,
            "cold_pc": self._cold_pc,
            "success": success,
        }
        return self._obs(), float(reward), False, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._fleet is None:
            return None
        rows = [(name, f"{self._fleet.buffer_mb[name]:.1f} Мбит") for name in self._names]
        rows.append(("шлюз", self._names[self._gateway]))
        return format_ansi("Fleet shift", rows)
