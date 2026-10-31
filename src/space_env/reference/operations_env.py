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
    """Joint action:0 wait,1..N station; optional explicit product/station choices.

    Reward is newly completed on-time product Mbits, never a geometric contact
    or partially delivered product. Fixed trajectories cannot react to maneuvers.
    """
    metadata: ClassVar = {"render_modes": []}

    def __init__(self, spec: dict, *, dt_sec: float = 30., product_selection: bool = False, cpu_control: bool = False,
                 cpu_job_selection: bool = False):
        if isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec <= 1e-9:
            raise ValueError("environment step must be finite and exceed 1ns")
        self._mission_spec = copy.deepcopy(spec)
        self.dt_sec = dt_sec
        self.mission = ReferenceDownlinkMission(self._mission_spec)
        self.actors = self.mission.actors
        self.stations = list(self.mission.stations)
        self.product_ids = list(self.mission.products)
        if not isinstance(cpu_job_selection, bool):
            raise TypeError("cpu_job_selection must be boolean")
        if cpu_job_selection and cpu_control:
            raise ValueError("choose cpu_control or cpu_job_selection")
        self.cpu_job_selection = cpu_job_selection
        self.cpu_product_ids = {a: [p.id for p in self.mission.products.values() if p.actor == a] for a in self.actors}
        if not isinstance(product_selection, bool):
            raise TypeError("product_selection must be boolean")
        self.action_choices = {
            a: [{"station": None, "product": None}]
            + [{"station": s, "product": None} for s in self.stations]
            + ([{"station": s, "product": p.id} for s in self.stations
                for p in self.mission.products.values()
                if p.actor == a and (not p.destination or p.destination == s)] if product_selection else [])
            for a in self.actors}
        if not isinstance(cpu_control, bool):
            raise TypeError("cpu_control must be boolean")
        if cpu_control:
            self.action_choices = {a: rows+[{**row, "processing": False} for row in rows]
                                   for a, rows in self.action_choices.items()}
        self.action_space = spaces.MultiDiscrete([len(self.action_choices[a]) for a in self.actors])
        if cpu_job_selection:
            self.action_space = spaces.Dict({"radio": self.action_space,
                                            "cpu": spaces.MultiDiscrete([len(self.cpu_product_ids[a])+2 for a in self.actors])})
        self.observation_space = spaces.Dict({
            "clock": spaces.Box(0, self.mission.horizon_sec, (1,), dtype=np.float64),
            "soc": spaces.Box(0, 1, (len(self.actors),), dtype=np.float64),
            "stored_bits": spaces.Box(0, np.array([self.mission.vehicles[a].memory_bits for a in self.actors]), dtype=np.float64),
            "contacts": spaces.MultiBinary((len(self.actors), len(self.stations))),
            "station_permitted": spaces.MultiBinary((len(self.actors), len(self.stations))),
            "thermal_locked": spaces.MultiBinary(len(self.actors)),
            "thermal_enabled": spaces.MultiBinary(len(self.actors)),
            "temperature_k": spaces.Box(0, np.array([max(v.thermal.initial_k, v.thermal.ambient_k+v.thermal.heat_tx_w/v.thermal.conductance_w_per_k)
                                                      if v.thermal else 0 for v in self.mission.vehicles.values()]), dtype=np.float64),
            "tx_seconds": spaces.Box(0, self.mission.horizon_sec, (len(self.actors),), dtype=np.float64),
            "tx_quota_remaining_sec": spaces.Box(0, self.mission.horizon_sec, (len(self.actors),), dtype=np.float64),
        })
        product_high = np.array([[p.size_bits, self.mission.horizon_sec, self.mission.horizon_sec,
                                  p.priority, len(self.actors)-1, len(self.stations)]
                                 for p in self.mission.products.values()], dtype=np.float64).reshape((-1, 6))
        self.observation_space["products"] = spaces.Box(np.zeros_like(product_high), product_high, dtype=np.float64)
        self.observation_space["processing_remaining_sec"] = spaces.Box(0, np.array([p.processing_sec for p in self.mission.products.values()]), dtype=np.float64)
        self.observation_space["workspace_bits"] = spaces.Box(0, np.array([self.mission.vehicles[p.actor].memory_bits for p in self.mission.products.values()]), dtype=np.float64)
        self.observation_space["processing_seconds"] = spaces.Box(0, self.mission.horizon_sec, (len(self.actors),), dtype=np.float64)
        self.observation_space["dependency_edges"] = spaces.Box(0, 1, (len(self.product_ids), len(self.product_ids)), dtype=np.int8)
        self.observation_space["retained"] = spaces.Box(0, 1, (len(self.product_ids),), dtype=np.int8)
        self._done = False

    def _observation(self):
        view = self.mission.observation()
        products = np.array([[self.mission.remaining[p.id], p.ready_sec, p.deadline_sec, p.priority,
                              self.actors.index(p.actor), 0 if not p.destination else self.stations.index(p.destination)+1]
                             for p in self.mission.products.values()], dtype=np.float64).reshape((-1, 6))
        return {"products": products, "clock": np.array([view["time_sec"]]),
                "processing_remaining_sec": np.array([self.mission.processing_remaining[p] for p in self.product_ids]),
                "workspace_bits": np.array([self.mission.products[p].workspace_bits for p in self.product_ids]),
                "processing_seconds": np.array([self.mission.processing_seconds[a] for a in self.actors]),
                "dependency_edges": np.array([[parent in self.mission.products[child].depends_on for parent in self.product_ids]
                                               for child in self.product_ids], dtype=np.int8).reshape((len(self.product_ids), len(self.product_ids))),
                "retained": np.array([bool(self.mission.retained_for(p)) for p in self.product_ids], dtype=np.int8),
                "thermal_locked": np.array([self.mission.thermal_locked[a] for a in self.actors], dtype=np.int8),
                "thermal_enabled": np.array([self.mission.vehicles[a].thermal is not None for a in self.actors], dtype=np.int8),
                "temperature_k": np.array([self.mission.temperature[a] for a in self.actors]),
                "tx_seconds": np.array([self.mission.tx_seconds[a] for a in self.actors]),
                "tx_quota_remaining_sec": np.array([self.mission.horizon_sec if self.mission.vehicles[a].max_tx_sec is None else
                                                    min(self.mission.horizon_sec, max(0., self.mission.vehicles[a].max_tx_sec-self.mission.tx_seconds[a])) for a in self.actors]),
                "station_permitted": np.array([[self.mission.station_permitted(a, s) for s in self.stations] for a in self.actors], dtype=np.int8),
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
                                     "action_choices": copy.deepcopy(self.action_choices),
                                     "cpu_product_ids": copy.deepcopy(self.cpu_product_ids),
                                     "cpu_job_selection": self.cpu_job_selection,
                                     "cpu_commands": {"auto": 0, "pause": 1, "first_product": 2} if self.cpu_job_selection else None,
                                     "product_fields": ["remaining_bits", "ready_sec", "deadline_sec", "priority", "actor_index", "destination_index"]}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset is required after episode completion")
        if not self.action_space.contains(action):
            raise ValueError("invalid joint station action")
        before = self.mission.report()["on_time_bits"]
        radio = action["radio"] if self.cpu_job_selection else action
        choices = {a: self.action_choices[a][int(choice)] for a, choice in zip(self.actors, radio)}
        processing = {a: choice.get("processing", True) for a, choice in choices.items()}
        cpu_jobs = {}
        if self.cpu_job_selection:
            for a, cpu in zip(self.actors, action["cpu"]):
                processing[a] = int(cpu) != 1
                if int(cpu) >= 2:
                    cpu_jobs[a] = self.cpu_product_ids[a][int(cpu)-2]
        report = self.mission.step(min(self.dt_sec, self.mission.horizon_sec-self.mission.clock),
                                   {a: choice["station"] for a, choice in choices.items()},
                                   products={a: choice["product"] for a, choice in choices.items()
                                             if choice["product"] is not None},
                                   processing=processing, processing_jobs=cpu_jobs)
        terminated = all(p["status"] != "incomplete" for p in report["products"])
        truncated = self.mission.clock >= self.mission.horizon_sec-1e-9 and not terminated
        self._done = terminated or truncated
        report["success"] = all(p["status"] == "delivered_on_time" for p in report["products"])
        return self._observation(), (report["on_time_bits"]-before)/1e6, terminated, truncated, report
