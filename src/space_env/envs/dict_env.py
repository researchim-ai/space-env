"""Словарное наблюдение одного аппарата. FleetWatch и его плоский вектор не меняются."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.envs.mission_fleet_env import packaged_mission
from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.mission.tasking import request_lifecycle, requests_from_scenario

DICT_KEYS = ("clock", "soc", "contact", "open_requests")


class MissionDictEnv(gym.Env):
    """Наблюдение — словарь сущностей, действие — режим одного аппарата."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: str | None = None, max_episode_steps: int = 4):
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self._scenario_path = str(packaged_mission("family_nominal.yaml" if scenario is None else scenario))
        self._step_sec = 30.0
        box = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({key: box for key in DICT_KEYS})
        self.action_space = gym.spaces.Discrete(3)
        self._world = None
        self._requests = []
        self._name = ""
        self._steps = 0

    def _require(self):
        if self._world is None:
            raise RuntimeError("сначала reset")
        return self._world

    def _obs(self) -> dict[str, np.ndarray]:
        world = self._require()
        engine = world.engines[self._name]
        snap = engine.get_state_snapshot()
        now = world.now_sec()
        open_count = sum(1 for item in self._requests if now + 1e-9 >= item.release_sec and now < item.deadline_sec)
        return {
            "clock": np.array([min(1.0, now / max(world.horizon_sec, 1.0))], dtype=np.float32),
            "soc": np.array([float(snap.battery_soc)], dtype=np.float32),
            "contact": np.array([1.0 if snap.in_contact else 0.0], dtype=np.float32),
            "open_requests": np.array([min(1.0, open_count / 4.0)], dtype=np.float32),
        }

    def _info(self) -> dict:
        world = self._require()
        now = world.now_sec()
        receipts = [item for engine in world.engines.values() for item in engine.receipts]
        statuses = [
            {"request_id": item.request_id, "status": request_lifecycle(item, receipts, now)}
            for item in self._requests
        ]
        return {"success": any(item["status"] == "delivered" for item in statuses), "statuses": statuses}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        spec = load_scenario_v2(self._scenario_path)
        self._world = world_from_scenario(self._scenario_path, attach_stations=True, refresh_passes=False)
        self._requests = requests_from_scenario(spec)
        self._name = str(spec["instances"][0]["id"])
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        world = self._require()
        kind = ("wait", "imaging", "comms")[int(action)]
        remaining = world.horizon_sec - world.now_sec()
        if remaining <= 1e-9:
            return self._obs(), 0.0, False, True, self._info()
        world.step(min(self._step_sec, remaining), {self._name: kind})
        self._steps += 1
        info = self._info()
        truncated = self._steps >= self.max_episode_steps or world.now_sec() + 1e-9 >= world.horizon_sec
        return self._obs(), 1.0 if info["success"] else 0.0, False, truncated, info
