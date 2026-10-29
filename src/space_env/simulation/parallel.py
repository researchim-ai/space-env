"""Параллельный шаг нескольких аппаратов. Контракт шага совпадает с ParallelEnv.

Каждый аппарат видит только свой заряд и свой контакт. Награда шага — его собственные
сданные биты. Пакет pettingzoo не импортируется: совместное действие исполняет тот же world.step.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

_MODES = ("wait", "imaging", "comms")


class ParallelFleet:
    """Один мир, отдельное наблюдение и награда у каждого аппарата."""

    metadata = {"name": "parallel_fleet", "is_parallelizable": True}

    def __init__(self, world, dt_sec: float = 1.0) -> None:
        self.world = world
        self.dt_sec = float(dt_sec)
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

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        del seed
        self.agents = list(self.possible_agents)
        return self._observe(), {name: {} for name in self.agents}

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        missing = [name for name in self.agents if name not in actions]
        if missing:
            raise KeyError(f"нет действия {missing[0]}")
        orders = {name: _MODES[int(actions[name])] for name in self.agents}
        before = {name: float(self.world.engines[name].delivered_bits) for name in self.agents}
        self.world.step(self.dt_sec, orders)
        rewards = {
            name: float(self.world.engines[name].delivered_bits) - before[name] for name in self.agents
        }
        done = self.world.now_sec() + 1e-9 >= self.world.horizon_sec
        terminations = {name: False for name in self.agents}
        truncations = {name: done for name in self.agents}
        infos = {name: {"delivered_bits": float(self.world.engines[name].delivered_bits)} for name in self.agents}
        return self._observe(), rewards, terminations, truncations, infos

    def _observe(self) -> dict:
        observations = {}
        for name in self.agents:
            snap = self.world.engines[name].get_state_snapshot()
            observations[name] = {
                "soc": np.array([float(snap.battery_soc)], dtype=np.float32),
                "contact": np.array([1.0 if snap.in_contact else 0.0], dtype=np.float32),
            }
        return observations
