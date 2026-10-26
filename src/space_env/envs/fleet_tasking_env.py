"""Два аппарата, одна станция. Наблюдение не содержит выбранного победителя."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine
from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.simulation.world import MissionWorld

_EPOCH = "2026-10-20T10:00:00Z"
_NAMES = ("alpha", "beta")
_STEP_SEC = 10.0


class FleetTaskingEnv(gym.Env):
    """Централизованный агент. Команда каждого аппарата — ждать, снимать или сдавать."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 4):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self._world: MissionWorld | None = None

    def _require(self) -> MissionWorld:
        if self._world is None:
            raise RuntimeError("сначала reset")
        return self._world

    def _obs(self) -> np.ndarray:
        world = self._require()
        values: list[float] = []
        for name in _NAMES:
            engine = world.engines[name]
            snap = engine.get_state_snapshot()
            stored = 1.0 if engine.stored_bits > 0.0 else 0.0
            contact = 1.0 if snap.in_contact else 0.0
            station = 1.0 if snap.in_contact else 0.0
            values.extend([float(snap.battery_soc), stored, contact, station])
        return np.array(values, dtype=np.float32)

    def _info(self) -> dict:
        world = self._require()
        delivered = {name: float(world.engines[name].delivered_bits) for name in _NAMES}
        winners = [name for name, bits in delivered.items() if bits > 0.0]
        return {
            "success": len(winners) == 1,
            "delivered": delivered,
            "station": "Shared",
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        probe = make_engine("cubesat-6u", epoch=_EPOCH, nu_deg=0.0, with_stations=False)
        lat, lon = subsatellite_latlon(probe.orbit.r, probe.now())
        engines = {}
        for name in _NAMES:
            engine = make_engine("cubesat-6u", epoch=_EPOCH, nu_deg=0.0, with_stations=False)
            engine.spacecraft.config.name = name
            engine.stations = [GroundStation(name="Shared", lat_deg=lat, lon_deg=lon, min_elevation_deg=5.0)]
            engine.scenario.ground_stations = list(engine.stations)
            engine._refresh_environment()
            engine.accept_bits(5.0e6, origin="imaging", request_id=name)
            engines[name] = engine
        self._world = MissionWorld(engines, epoch=_EPOCH, horizon_sec=10.0, seed=0 if seed is None else seed)
        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        world = self._require()
        commands = {0: "wait", 1: "imaging", 2: "comms"}
        orders = {
            name: commands.get(int(np.asarray(action).ravel()[index]), "wait")
            for index, name in enumerate(_NAMES)
        }
        before = {name: world.engines[name].delivered_bits for name in _NAMES}
        world.step(_STEP_SEC, orders)
        info = self._info()
        gained = sum(world.engines[name].delivered_bits - before[name] for name in _NAMES)
        reward = 1.0 if info["success"] and gained > 0.0 else 0.0
        done = world.now_sec() + 1e-9 >= world.horizon_sec or info["success"]
        return self._obs(), reward, done, False, info
