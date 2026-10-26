"""Флот видит сроки и маски заявок. Правильной команды в наблюдении нет."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from space_env.mission.factory import SHORT_REQUESTS, SHORT_VEHICLES
from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
from space_env.mission.tasking import _focus_target, _visible, request_lifecycle, requests_from_scenario


def packaged_mission(name: str) -> Path:
    cwd = Path.cwd() / "configs" / "scenarios" / name
    if cwd.is_file():
        return cwd
    return Path(__file__).resolve().parents[1] / "configs" / name


def observation_names() -> tuple[str, ...]:
    names = ["доля горизонта"]
    for index in range(SHORT_VEHICLES):
        names.extend(
            [
                f"заряд {index + 1}",
                f"доля объёма {index + 1}",
                f"контакт {index + 1}",
            ]
        )
    for index in range(SHORT_REQUESTS):
        names.extend(
            [
                f"выпущена {index + 1}",
                f"приоритет {index + 1}",
                f"доля срока {index + 1}",
                f"маска 1 заявка {index + 1}",
                f"маска 2 заявка {index + 1}",
            ]
        )
    return tuple(names)


class MissionFleetEnv(gym.Env):
    """Два аппарата и двенадцать заявок. Действие — режим и номер заявки."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: str | None = None, max_episode_steps: int = 8):
        super().__init__()
        self._scenario_path = str(packaged_mission("mission_short.yaml") if scenario is None else scenario)
        self.max_episode_steps = max_episode_steps
        self._step_sec = 60.0
        width = len(observation_names())
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(width,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, SHORT_REQUESTS, 3, SHORT_REQUESTS])
        self._world = None
        self._requests = []
        self._names: list[str] = []

    def _require(self):
        if self._world is None:
            raise RuntimeError("сначала reset")
        return self._world

    def _masks(self) -> list[list[float]]:
        world = self._require()
        masks = []
        for request in self._requests:
            row = []
            for name in self._names:
                engine = world.engines[name]
                visible = request.release_sec <= engine.time_sec and _visible(engine, request)
                row.append(1.0 if visible else 0.0)
            masks.append(row)
        return masks

    def _obs(self) -> np.ndarray:
        world = self._require()
        now = world.now_sec()
        values = [min(1.0, now / world.horizon_sec)]
        for name in self._names:
            engine = world.engines[name]
            snap = engine.get_state_snapshot()
            stored = min(1.0, float(engine.stored_bits) / 8.0e6)
            values.extend([float(snap.battery_soc), stored, 1.0 if snap.in_contact else 0.0])
        masks = self._masks()
        for index, request in enumerate(self._requests):
            released = 1.0 if now + 1e-9 >= request.release_sec else 0.0
            priority = min(1.0, request.priority / 4.0)
            remaining = max(0.0, request.deadline_sec - now) / max(request.deadline_sec, 1.0)
            values.extend([released, priority, remaining, masks[index][0], masks[index][1]])
        return np.array(values, dtype=np.float32)

    def _receipts(self) -> list[dict]:
        world = self._require()
        return [item for engine in world.engines.values() for item in engine.receipts]

    def _info(self, rejected: list[str]) -> dict:
        world = self._require()
        now = world.now_sec()
        receipts = self._receipts()
        statuses = [
            {"request_id": item.request_id, "status": request_lifecycle(item, receipts, now)}
            for item in self._requests
        ]
        return {
            "success": any(item["status"] == "delivered" for item in statuses),
            "statuses": statuses,
            "rejected": rejected,
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        spec = load_scenario_v2(self._scenario_path)
        if len(spec["instances"]) != SHORT_VEHICLES or len(spec["requests"]) != SHORT_REQUESTS:
            raise ValueError("MissionFleet ждёт короткую миссию: 2 аппарата и 12 заявок")
        self._world = world_from_scenario(self._scenario_path, attach_stations=True, refresh_passes=False)
        self._requests = requests_from_scenario(spec)
        self._names = [item["id"] for item in spec["instances"]]
        return self._obs(), self._info([])

    def follow_policy(self) -> dict:
        """Тот же жадный план, что CLI. Наблюдение ему не подсказывает кнопку."""
        from space_env.mission.tasking import run_policy

        return run_policy(self._world, self._requests, step_sec=self._step_sec)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        world = self._require()
        raw = np.asarray(action, dtype=int).ravel()
        modes = {0: "wait", 1: "imaging", 2: "comms"}
        orders = {}
        rejected = []
        ids = {item.request_id for item in self._requests}
        before = sum(1 for item in self._info([])["statuses"] if item["status"] == "delivered")
        for index, name in enumerate(self._names):
            mode = int(raw[index * 2])
            slot = int(raw[index * 2 + 1])
            if mode == 1:
                request = self._requests[slot]
                engine = world.engines[name]
                if request.release_sec <= engine.time_sec and _visible(engine, request):
                    _focus_target(engine, request, ids)
                    orders[name] = "imaging"
                else:
                    orders[name] = "wait"
                    rejected.append(request.request_id)
            else:
                orders[name] = modes.get(mode, "wait")
        world.step(self._step_sec, orders)
        info = self._info(rejected)
        after = sum(1 for item in info["statuses"] if item["status"] == "delivered")
        reward = 1.0 if after > before else 0.0
        done = world.now_sec() + 1e-9 >= world.horizon_sec
        return self._obs(), reward, done, False, info
