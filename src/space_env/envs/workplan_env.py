"""
WorkPlanEnv: выполнить сгенерированный план съёмки и сдачи.
Награда — за закрытое дело из списка, а не за любую съёмку.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine
from space_env.catalog.references import reference
from space_env.catalog.traffic import dump_bits, store_bits
from space_env.core.link import stepped_rate_bps
from space_env.envs.base import format_ansi
from space_env.mission.scenarios import GroundTarget
from space_env.mission.workplan import WorkItem, WorkPlan, build_acquisition_plan
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine

_COMMANDS = {0: "wait", 1: "imaging", 2: "comms"}


class WorkPlanEnv(gym.Env):
    """
    Наблюдение: заряд, тень, контакт, заполнение памяти, доля закрытых дел,
    вид ближайшего дела, попадание шага в его окно, время до него.
    Действия: 0 ждать, 1 снимать, 2 сдавать.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        vehicle: str = "sentinel-2a",
        step_sec: float = 60.0,
        max_episode_steps: int = 100,
        initial_bits: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.vehicle = vehicle
        self.step_sec = step_sec
        self.max_episode_steps = max_episode_steps
        self.initial_bits = initial_bits
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        spec = reference(vehicle)
        self._store_rate_bps = float(spec.store_rate_bps or spec.downlink_bps)
        self._imaging_cap_s = float(spec.max_imaging_s) if spec.max_imaging_s is not None else 1.0e12
        self._engine: SimulationEngine | None = None
        self._plan: WorkPlan | None = None
        self._done: set[int] = set()
        self._buffer_bits = 0.0
        self._delivered_bits = 0.0
        self._imaging_sec = 0.0
        self._steps = 0
        self._weights = load_rewards()["workplan"]

    def _required(self) -> list[WorkItem]:
        if self._plan is None:
            return []
        return [item for item in self._plan.required_items() if item.kind in {"imaging", "comms"}]

    def _active(self, now: float) -> WorkItem | None:
        horizon = now + self.step_sec
        for index, item in enumerate(self._required()):
            if index in self._done:
                continue
            if item.start_sec < horizon and item.end_sec > now:
                return item
        return None

    def _next_pending(self, now: float) -> WorkItem | None:
        active = self._active(now)
        if active is not None:
            return active
        for index, item in enumerate(self._required()):
            if index in self._done or item.end_sec <= now:
                continue
            return item
        return None

    def _obs(self) -> np.ndarray:
        if self._engine is None or self._plan is None:
            return np.zeros(8, dtype=np.float32)
        snap = self._engine.get_state_snapshot()
        now = self._steps * self.step_sec
        required = self._required()
        done_frac = 0.0 if not required else len(self._done) / len(required)
        pending = self._next_pending(now)
        kind = 0.0
        in_window = 0.0
        time_to = 1.0
        if pending is not None:
            kind = 1.0 if pending.kind == "comms" else 0.5
            in_window = 1.0 if self._active(now) is not None else 0.0
            time_to = float(np.clip((pending.start_sec - now) / max(self._plan.horizon_sec, 1.0), 0.0, 1.0))
        return np.array(
            [
                snap.battery_soc,
                float(snap.in_eclipse),
                float(snap.in_contact),
                float(np.clip(self._buffer_bits / max(self._plan.memory_bits, 1.0), 0.0, 1.0)),
                done_frac,
                kind,
                in_window,
                time_to,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        self._done = set()
        self._buffer_bits = self.initial_bits
        self._delivered_bits = 0.0
        self._imaging_sec = 0.0
        horizon = self.max_episode_steps * self.step_sec
        self._plan = build_acquisition_plan(
            self.vehicle,
            horizon_sec=horizon,
            initial_bits=self.initial_bits,
            sample_sec=self.step_sec,
        )
        targets = [
            GroundTarget(name=name, lat_deg=lat, lon_deg=lon) for name, lat, lon in self._plan.targets
        ]
        self._engine = make_engine(self.vehicle, nu_deg=180.0, with_stations=True, ground_targets=targets)
        return self._obs(), self._info(0.0)

    def _info(self, dumped: float) -> dict:
        required = self._required()
        tasks = []
        for index, item in enumerate(required):
            tasks.append(
                {
                    "kind": item.kind,
                    "name": item.name,
                    "start_sec": item.start_sec,
                    "end_sec": item.end_sec,
                    "done": index in self._done,
                }
            )
        finished = bool(required) and len(self._done) == len(required)
        snap = None if self._engine is None else self._engine.get_state_snapshot()
        alive = snap is None or snap.battery_soc >= 0.2
        return {
            "plan": [] if self._plan is None else self._plan.lines(),
            "tasks": tasks,
            "delivered_bits": self._delivered_bits,
            "dumped_bits": dumped,
            "success": finished and alive,
        }

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._engine is None or self._plan is None:
            raise RuntimeError("Сначала вызовите reset()")
        action = int(np.asarray(action).flat[0])
        command = _COMMANDS.get(action, "wait")
        now = self._steps * self.step_sec
        active = self._active(now)
        before = self._engine.get_state_snapshot().imaging_count
        _snap, _events = self._engine.step(self.step_sec, None if command == "wait" else command)
        snap = self._engine.get_state_snapshot()
        completed = False
        if command == "imaging" and snap.imaging_count > before:
            self._buffer_bits, self._imaging_sec, _stored = store_bits(
                self._buffer_bits,
                self.step_sec,
                rate_bps=self._store_rate_bps,
                memory_bits=self._plan.memory_bits,
                imaging_sec=self._imaging_sec,
                imaging_cap_sec=self._imaging_cap_s,
            )
            if active is not None and active.kind == "imaging" and active.name in self._engine.captured_targets():
                self._done.add(self._required().index(active))
                completed = True
        dumped = 0.0
        if command == "comms" and snap.link_ok and snap.link_margin_db >= 0.0:
            rate = stepped_rate_bps(snap.link_margin_db, self._engine.spacecraft.config.comms.data_rate_max_bps)
            self._buffer_bits, dumped = dump_bits(self._buffer_bits, self.step_sec, rate)
            self._delivered_bits += dumped
            if dumped > 0.0 and active is not None and active.kind == "comms":
                self._done.add(self._required().index(active))
                completed = True
        reward = self._weights["task_done"] if completed else 0.0
        self._steps += 1
        required = self._required()
        finished = bool(required) and len(self._done) == len(required)
        terminated = snap.battery_soc < 0.2 or finished
        if snap.battery_soc < 0.2:
            reward -= self._weights["soc_failure"]
            terminated = True
            finished = False
        truncated = self._steps >= self.max_episode_steps and not finished
        info = self._info(dumped)
        if snap.battery_soc < 0.2:
            info["success"] = False
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._plan is None:
            return None
        rows = [(item.reason, "да" if index in self._done else "нет") for index, item in enumerate(self._required())]
        return format_ansi(self.vehicle, rows or [("план", "пуст")])
