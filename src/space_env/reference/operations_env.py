"""Gymnasium facade for the explicitly simplified reference downlink profile."""
from __future__ import annotations

import copy
import math
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from space_env.reference.operations import ReferenceDownlinkMission


class ReferenceDownlinkEnv(gym.Env):
    """Joint action:0 wait,1..N station; full deterministic resource telemetry.

    Reward is newly completed on-time product Mbits, never a geometric contact
    or partially delivered product. Fixed trajectories cannot react to maneuvers.
    """
    metadata: ClassVar = {"render_modes": []}

    def __init__(self, spec: dict, *, dt_sec: float = 30.):
        if isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec <= 1e-9:
            raise ValueError("environment step must be finite and exceed 1ns")
        self._mission_spec = copy.deepcopy(spec)
        self.dt_sec = dt_sec
        self.mission = ReferenceDownlinkMission(self._mission_spec)
        self.actors = self.mission.actors
        self.stations = list(self.mission.stations)
        self.product_ids = list(self.mission.products)
        self.action_space = spaces.MultiDiscrete([len(self.stations)+1]*len(self.actors))
        self.observation_space = spaces.Dict({
            "clock": spaces.Box(0, self.mission.horizon_sec, (1,), dtype=np.float64),
            "soc": spaces.Box(0, 1, (len(self.actors),), dtype=np.float64),
            "stored_bits": spaces.Box(0, np.array([self.mission.vehicles[a].memory_bits for a in self.actors]), dtype=np.float64),
            "contacts": spaces.MultiBinary((len(self.actors), len(self.stations))),
        })
        product_high = np.array([[p.size_bits, self.mission.horizon_sec, self.mission.horizon_sec,
                                  p.priority, len(self.actors)-1, len(self.stations)]
                                 for p in self.mission.products.values()], dtype=np.float64).reshape((-1, 6))
        self.observation_space["products"] = spaces.Box(np.zeros_like(product_high), product_high, dtype=np.float64)
        self._done = False

    def _observation(self):
        view = self.mission.observation()
        products = np.array([[self.mission.remaining[p.id], p.ready_sec, p.deadline_sec, p.priority,
                              self.actors.index(p.actor), 0 if not p.destination else self.stations.index(p.destination)+1]
                             for p in self.mission.products.values()], dtype=np.float64).reshape((-1, 6))
        return {"products": products, "clock": np.array([view["time_sec"]]),
                "soc": np.array([view["actors"][a]["soc"] for a in self.actors]),
                "stored_bits": np.array([view["actors"][a]["stored_bits"] for a in self.actors]),
                "contacts": np.array([[s in view["actors"][a]["contacts"] for s in self.stations]
                                      for a in self.actors], dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("reference downlink reset options are unsupported")
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.mission = ReferenceDownlinkMission(self._mission_spec)
        self._done = False
        return self._observation(), {"profile": self.mission.report()["profile"], "actors": list(self.actors), "stations": list(self.stations),
                                     "product_ids": list(self.product_ids),
                                     "product_fields": ["remaining_bits", "ready_sec", "deadline_sec", "priority", "actor_index", "destination_index"]}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset is required after episode completion")
        if not self.action_space.contains(action):
            raise ValueError("invalid joint station action")
        before = self.mission.report()["on_time_bits"]
        report = self.mission.step(min(self.dt_sec, self.mission.horizon_sec-self.mission.clock),
                                   {a: None if int(choice) == 0 else self.stations[int(choice)-1]
                                    for a, choice in zip(self.actors, action)})
        terminated = all(p["status"] != "incomplete" for p in report["products"])
        truncated = self.mission.clock >= self.mission.horizon_sec-1e-9 and not terminated
        self._done = terminated or truncated
        report["success"] = all(p["status"] == "delivered_on_time" for p in report["products"])
        return self._observation(), (report["on_time_bits"]-before)/1e6, terminated, truncated, report
