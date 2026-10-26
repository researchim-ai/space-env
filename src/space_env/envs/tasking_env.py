"""Один аппарат и заявки файла v2. Наблюдение не содержит номера правильного действия."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.mission.scenarios import GroundTarget
from space_env.mission.tasking import TaskRequest, delivery_score, requests_from_scenario
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

_STEP_SEC = 30.0
_STATION_CODE = {"Kiruna": 0.25, "Svalbard": 0.5, "Matera": 0.75}


class TaskingShiftEnv(gym.Env):
    """Ждать, снимать или сдавать. Успех — заявка закрыта в срок."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: str | None = None, max_episode_steps: int = 40):
        super().__init__()
        self._scenario_path = str(packaged_tasking("tasking_deliver.yaml") if scenario is None else scenario)
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._world: MissionWorld | None = None
        self._requests: list[TaskRequest] = []
        self._vehicle = ""
        self._rewarded = False

    def _require(self) -> tuple[MissionWorld, SimulationEngine]:
        if self._world is None or not self._vehicle:
            raise RuntimeError("сначала reset")
        return self._world, self._world.engines[self._vehicle]

    def _receipts(self) -> list[dict]:
        world, _engine = self._require()
        return [item for engine in world.engines.values() for item in engine.receipts]

    def _score(self) -> dict:
        return delivery_score(self._requests, self._receipts(), [])

    def _success(self) -> bool:
        if not self._requests:
            return False
        return set(self._score()["on_time"]) == {item.request_id for item in self._requests}

    def _obs(self) -> np.ndarray:
        _world, engine = self._require()
        snap = engine.get_state_snapshot()
        size = sum(item.size_bits for item in self._requests) or 1.0
        stored = min(1.0, float(engine.stored_bits) / size)
        contact = 1.0 if snap.in_contact else 0.0
        station = _STATION_CODE.get(snap.contact_station, 1.0) if snap.in_contact else 0.0
        return np.array([float(snap.battery_soc), stored, contact, station], dtype=np.float32)

    def _info(self, success: bool) -> dict:
        world, engine = self._require()
        snap = engine.get_state_snapshot()
        request = self._requests[0]
        return {
            "success": success,
            "scenario": self._scenario_path,
            "request": request.request_id,
            "deadline_sec": float(request.deadline_sec),
            "delivered_bits": float(engine.delivered_bits),
            "weighted_on_time": float(self._score()["weighted_on_time"]),
            "station": snap.contact_station,
            "time_sec": float(engine.time_sec),
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        spec = load_scenario_v2(self._scenario_path)
        if len(spec["instances"]) != 1:
            raise ValueError("TaskingShift ведёт один аппарат")
        world = world_from_scenario(self._scenario_path, attach_stations=True)
        self._requests = requests_from_scenario(spec)
        engine = next(iter(world.engines.values()))
        for request in self._requests:
            if any(target.name == request.request_id for target in engine.scenario.ground_targets):
                continue
            engine.scenario.ground_targets.append(
                GroundTarget(name=request.request_id, lat_deg=request.lat_deg, lon_deg=request.lon_deg)
            )
        self._world = world
        self._vehicle = next(iter(world.engines))
        self._rewarded = False
        return self._obs(), self._info(False)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        world, _engine = self._require()
        choice = int(np.asarray(action).flat[0])
        command = {0: "wait", 1: "imaging", 2: "comms"}.get(choice, "wait")
        world.step(_STEP_SEC, {name: command for name in world.engines})
        success = self._success()
        finished = success or world.engines[self._vehicle].time_sec + 1e-6 >= min(
            world.horizon_sec,
            max(item.deadline_sec for item in self._requests),
        )
        reward = 0.0
        if success and not self._rewarded:
            reward = 1.0
            self._rewarded = True
        return self._obs(), reward, finished, False, self._info(success)


def packaged_tasking(name: str) -> Path:
    """Файл заявки. Сначала каталог запуска, затем пакет."""
    cwd = Path.cwd() / "configs" / "scenarios" / name
    if cwd.is_file():
        return cwd
    return Path(__file__).resolve().parents[1] / "configs" / name
