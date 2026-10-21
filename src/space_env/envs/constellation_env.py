"""
ConstellationEnv: один шаг ведёт всю цепочку, действие выбирает аппарат и команду.
Кадр кладётся в буфер, если надир доведён и аппарат не в umbra.
Буфер уходит на Землю в сеансе или соседу по открытой межспутниковой линии.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.envs.base import format_ansi, normalize_altitude_km
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.fleet import Fleet

_COMMANDS = {0: None, 1: "imaging", 2: "comms", 3: "minimal_power", 4: "relay"}
_FRAME_MB = 8.0


class ConstellationEnv(gym.Env):
    """
    Действие MultiDiscrete([число аппаратов, 5]): кому и что.
    Команды: ничего, съёмка, связь, минимальная нагрузка, ретрансляция соседу.
    Шаг цепочки по умолчанию 1.5°: на этой дистанции линия 1 Вт / 6 дБи ещё закрывается.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        count: int = 2,
        separation_deg: float = 1.5,
        step_sec: float = 60.0,
        max_episode_steps: int = 20,
        render_mode: str | None = None,
    ):
        super().__init__()
        if count < 2:
            raise ValueError("в среде минимум два аппарата")
        self.count = count
        self.separation_deg = separation_deg
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(count * 5 + 2,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([count, 5])
        self._fleet: Fleet | None = None
        self._names: list[str] = []
        self._steps = 0

    def _obs(self) -> np.ndarray:
        if self._fleet is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        values: list[float] = []
        for name in self._names:
            snap = self._fleet.vehicles[name].get_state_snapshot()
            values.extend(
                [
                    snap.battery_soc,
                    float(snap.in_eclipse),
                    float(snap.in_contact),
                    normalize_altitude_km(snap.altitude_km),
                    float(np.clip(self._fleet.buffer_mb[name] / 80.0, 0.0, 1.0)),
                ]
            )
        pairs = self._fleet.pairs()
        nearest = min((pair.range_m for pair in pairs), default=0.0)
        open_margin = max((pair.isl_margin_db for pair in pairs if not pair.earth_blocked), default=-20.0)
        values.append(float(np.clip(nearest / 2_000_000.0, 0.0, 1.0)))
        values.append(float(np.clip((open_margin + 20.0) / 40.0, 0.0, 1.0)))
        return np.asarray(values, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._fleet = Fleet.string_of_pearls(
            self.count,
            self.separation_deg,
            scenario=ScenarioConfig(ground_stations=[]),
        )
        self._names = list(self._fleet.vehicles)
        return self._obs(), {"vehicles": self._names}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._fleet is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = np.asarray(action, dtype=int).ravel()
        index = int(action[0])
        command = _COMMANDS.get(int(action[1]))
        name = self._names[index]
        engine_command = command if command in {"imaging", "comms", "minimal_power"} else None
        commands = {name: engine_command} if engine_command else {}
        self._fleet.step(self.step_sec, commands)
        reward = 0.0
        engine = self._fleet.vehicles[name]
        if command in {"imaging", "comms"} and engine.last_command_ok:
            reward += 0.2
        if command == "imaging":
            snap = engine.get_state_snapshot()
            sunlit_nadir = (not snap.in_eclipse) and snap.pointing == "nadir" and snap.slew_remaining_deg <= 1.0
            if sunlit_nadir:
                self._fleet.collect(name, _FRAME_MB)
                reward += 0.2
        elif command == "comms":
            reward += 0.05 * self._fleet.downlink(name, self.step_sec)
        elif command == "relay":
            neighbor = self._fleet.best_neighbor(name)
            moved = 0.0 if neighbor is None else self._fleet.relay(name, neighbor, self.step_sec)
            reward += 0.02 * moved
        low = []
        for vehicle, engine in self._fleet.vehicles.items():
            soc = engine.get_state_snapshot().battery_soc
            if soc < 0.2:
                reward -= 1.0
                low.append(vehicle)
        screen = self._fleet.screen()
        self._steps += 1
        terminated = bool(low)
        truncated = self._steps >= self.max_episode_steps
        info = {
            "commanded": name,
            "screen": [(hit.leader, hit.follower, hit.probability) for hit in screen],
            "buffer_mb": dict(self._fleet.buffer_mb),
            "delivered_mb": self._fleet.delivered_mb,
            "success": truncated and not terminated,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._fleet is None:
            return None
        rows = [(name, f"SoC {self._fleet.vehicles[name].get_state_snapshot().battery_soc:.2f}") for name in self._names]
        nearest = min(pair.range_m for pair in self._fleet.pairs())
        rows.append(("ближайшая пара, км", f"{nearest / 1000.0:.1f}"))
        return format_ansi("Constellation", rows)
