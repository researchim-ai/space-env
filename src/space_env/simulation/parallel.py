"""Параллельный шаг нескольких аппаратов. Контракт шага совпадает с ParallelEnv.

Каждый аппарат видит только свой заряд и свой контакт. Награда шага — его собственные
сданные биты. Пакет pettingzoo не импортируется: совместное действие исполняет тот же world.step.
"""
from __future__ import annotations

import math
from typing import ClassVar

import gymnasium as gym
import numpy as np

_MODES = ("wait", "imaging", "comms")


class ParallelFleet:
    """Один мир, отдельное наблюдение и награда у каждого аппарата."""

    metadata: ClassVar[dict] = {"name": "parallel_fleet", "is_parallelizable": True}

    def __init__(self, world, dt_sec: float = 1.0) -> None:
        self.world = world
        self.dt_sec = float(dt_sec)
        if not math.isfinite(self.dt_sec) or self.dt_sec <= 0:
            raise ValueError("шаг должен быть положительным и конечным")
        self._initial = world.snapshot()
        self.possible_agents = sorted(world.engines)
        self.agents = list(self.possible_agents)
        space = gym.spaces.Dict(
            {
                "soc": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "contact": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.observation_spaces = {name: space for name in self.possible_agents}
        self.action_spaces = {name: gym.spaces.Discrete(len(_MODES)) for name in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if options:
            raise ValueError("options для ParallelFleet не поддерживаются")
        self.world.restore(self._initial)
        if seed is not None:
            self.world.rng = np.random.default_rng(seed)
            for index, name in enumerate(self.possible_agents):
                self.action_spaces[name].seed(seed + index)
                self.world.engines[name]._rng.seed(seed + index)
        self.agents = list(self.possible_agents)
        return self._observe(), {name: {} for name in self.agents}

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        if not self.agents:
            if actions:
                raise ValueError("эпизод завершён; вызовите reset")
            return {}, {}, {}, {}, {}
        unknown = set(actions) - set(self.agents)
        if unknown:
            raise KeyError(f"нет аппарата {min(unknown)}")
        missing = [name for name in self.agents if name not in actions]
        if missing:
            raise KeyError(f"нет действия {missing[0]}")
        if any(not self.action_spaces[name].contains(actions[name]) for name in self.agents):
            raise ValueError("действие вне action_space")
        orders = {name: _MODES[int(actions[name])] for name in self.agents}
        before = {name: float(self.world.engines[name].delivered_bits) for name in self.agents}
        remaining = self.world.horizon_sec - self.world.now_sec()
        if remaining > 1e-9:
            self.world.step(min(self.dt_sec, remaining), orders)
        rewards = {
            name: float(self.world.engines[name].delivered_bits) - before[name] for name in self.agents
        }
        done = self.world.now_sec() + 1e-9 >= self.world.horizon_sec
        terminations = {name: False for name in self.agents}
        truncations = {name: done for name in self.agents}
        infos = {name: {"delivered_bits": float(self.world.engines[name].delivered_bits)} for name in self.agents}
        observations = self._observe()
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observe(self) -> dict:
        observations = {}
        for name in self.agents:
            snap = self.world.engines[name].get_state_snapshot()
            observations[name] = {
                "soc": np.array([float(snap.battery_soc)], dtype=np.float32),
                "contact": np.array([1.0 if snap.in_contact else 0.0], dtype=np.float32),
            }
        return observations
