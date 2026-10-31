"""Fixed-capacity Gym episode with causal, initially hidden operation events."""
from __future__ import annotations

import copy
import math
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from space_env.reference.events import validate_event_envelope
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.simulation.snapshot import canonical_json


class ReferenceEventDownlinkEnv(gym.Env):
    """Scripted events become visible only at their timestamp; no future tape in info.

    Each actor action: 0 wait; then station blocks of [automatic, slot1..slotN].
    Empty or incompatible slots wait, with invalid actors reported. No slot reuse.
    A step returns at the next event boundary so an agent can immediately react.
    """
    metadata: ClassVar = {"render_modes": []}

    def __init__(self, spec: dict, events: list[dict], *, product_slots: int = 16, dt_sec: float = 30., cpu_control: bool = False,
                 cpu_job_selection: bool = False):
        if isinstance(product_slots, bool) or not isinstance(product_slots, int) or not 1 <= product_slots <= 1000:
            raise ValueError("product_slots must be an integer in [1,1000]")
        if isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec <= 1e-9:
            raise ValueError("dt_sec must be finite and exceed 1ns")
        self._spec, self._events = copy.deepcopy(spec), copy.deepcopy(events)
        self.product_slots, self.dt_sec = product_slots, dt_sec
        if not isinstance(cpu_control, bool):
            raise TypeError("cpu_control must be boolean")
        self.cpu_control = cpu_control
        if not isinstance(cpu_job_selection, bool):
            raise TypeError("cpu_job_selection must be boolean")
        if cpu_job_selection and cpu_control:
            raise ValueError("choose cpu_control or cpu_job_selection")
        self.cpu_job_selection = cpu_job_selection
        self.mission = ReferenceDownlinkMission(spec)
        self.actors, self.stations = list(self.mission.actors), list(self.mission.stations)
        if not isinstance(events, list):
            raise TypeError("events must be a list")
        canonical_json(events)
        if len(events) > 10000:
            raise ValueError("event tape limit exceeded")
        previous = -1.
        ids = set()
        group_size = 0
        counts = {a: sum(p.actor == a for p in self.mission.products.values()) for a in self.actors}
        for event in events:
            validate_event_envelope(event)
            if event["id"] in ids:
                raise ValueError("duplicate event ID in tape")
            ids.add(event["id"])
            time = event["time_sec"]
            if isinstance(time, bool) or not math.isfinite(time) or not 0 <= time < self.mission.horizon_sec or time < previous:
                raise ValueError("event tape must be ordered and within horizon")
            if previous >= 0 and 0 < time-previous <= 1e-9:
                raise ValueError("distinct event times must be more than 1ns apart")
            group_size = group_size+1 if time == previous else 1
            if group_size > 1000:
                raise ValueError("event group limit exceeded")
            previous = time
            if event.get("type") == "onboard_product_added":
                actor = event["payload"].get("actor")
                if actor not in counts:
                    raise ValueError("event product has unknown actor")
                counts[actor] += 1
        if any(n > product_slots for n in counts.values()):
            raise ValueError("event tape exceeds per-actor product slot capacity")
        n = len(self.actors)
        self._action_width = 1+len(self.stations)*(product_slots+1)
        self.action_space = spaces.MultiDiscrete([self._action_width*(2 if cpu_control else 1)]*n)
        if cpu_job_selection:
            self.action_space = spaces.Dict({"radio": self.action_space,
                                            "cpu": spaces.MultiDiscrete([product_slots+2]*n)})
        product_high = np.zeros((n, product_slots, 5), dtype=np.float64)
        for i, actor in enumerate(self.actors):
            product_high[i, :, :] = [self.mission.vehicles[actor].memory_bits,
                                     self.mission.horizon_sec, self.mission.horizon_sec,
                                     np.finfo(np.float64).max, len(self.stations)]
        self.observation_space = spaces.Dict({
            "clock": spaces.Box(0, self.mission.horizon_sec, (1,), dtype=np.float64),
            "soc": spaces.Box(0, 1, (n,), dtype=np.float64),
            "stored_bits": spaces.Box(0, np.array([self.mission.vehicles[a].memory_bits for a in self.actors]), dtype=np.float64),
            "contacts": spaces.MultiBinary((n, len(self.stations))),
            "station_permitted": spaces.MultiBinary((n, len(self.stations))),
            "reserved": spaces.MultiBinary((n, len(self.stations))),
            "reservation_required": spaces.MultiBinary(len(self.stations)),
            "open_terminals": spaces.Box(0, np.array([self.mission.stations[s].terminals for s in self.stations]), dtype=np.int64),
            "thermal_locked": spaces.MultiBinary(n),
            "thermal_enabled": spaces.MultiBinary(n),
            "temperature_k": spaces.Box(0, np.array([max(v.thermal.initial_k, v.thermal.ambient_k+v.thermal.heat_tx_w/v.thermal.conductance_w_per_k)
                                                      if v.thermal else 0 for v in self.mission.vehicles.values()]), dtype=np.float64),
            "tx_seconds": spaces.Box(0, self.mission.horizon_sec, (n,), dtype=np.float64),
            "tx_quota_remaining_sec": spaces.Box(0, self.mission.horizon_sec, (n,), dtype=np.float64),
            "slot_active": spaces.MultiBinary((n, product_slots)),
            "slot_cancelled": spaces.MultiBinary((n, product_slots)),
            "transmitter_rates_bps": spaces.Box(0, np.array([self.mission.vehicles[a].rate_bps for a in self.actors]), dtype=np.float64),
            "station_rates_bps": spaces.Box(0, max(v.rate_bps for v in self.mission.vehicles.values()), (len(self.stations),), dtype=np.float64),
            "products": spaces.Box(np.zeros_like(product_high), product_high, dtype=np.float64),
            "processing_remaining_sec": spaces.Box(0, np.finfo(np.float64).max, (n, product_slots), dtype=np.float64),
            "workspace_bits": spaces.Box(0, np.array([[self.mission.vehicles[a].memory_bits]*product_slots for a in self.actors]), dtype=np.float64),
            "processing_seconds": spaces.Box(0, self.mission.horizon_sec, (n,), dtype=np.float64),
            "dependency_edges": spaces.MultiBinary((n, product_slots, product_slots)),
            "retained": spaces.MultiBinary((n, product_slots)),
        })
        self._cursor, self._done = 0, False

    def _slots(self, actor):
        return [p for p in self.mission.products.values() if p.actor == actor]

    def _observe(self):
        view = self.mission.observation()
        products = np.zeros((len(self.actors), self.product_slots, 5), dtype=np.float64)
        active = np.zeros((len(self.actors), self.product_slots), dtype=np.int8)
        cancelled = np.zeros_like(active)
        processing = np.zeros_like(active, dtype=np.float64)
        workspace = np.zeros_like(processing)
        edges = np.zeros((len(self.actors), self.product_slots, self.product_slots), dtype=np.int8)
        retained = np.zeros_like(active)
        for i, actor in enumerate(self.actors):
            for j, p in enumerate(self._slots(actor)):
                products[i, j] = [self.mission.remaining[p.id], p.ready_sec, p.deadline_sec,
                                  p.priority, 0 if not p.destination else self.stations.index(p.destination)+1]
                active[i, j] = 1
                cancelled[i, j] = int(p.id in self.mission.cancelled)
                processing[i, j] = self.mission.processing_remaining[p.id]
                workspace[i, j] = p.workspace_bits
                retained[i, j] = bool(self.mission.retained_for(p.id))
                for k, parent in enumerate(self._slots(actor)):
                    edges[i, j, k] = int(parent.id in p.depends_on)
        max_rate = max(v.rate_bps for v in self.mission.vehicles.values())
        return {"clock": np.array([self.mission.clock]), "products": products, "slot_active": active,
                "slot_cancelled": cancelled,
                "processing_remaining_sec": processing, "workspace_bits": workspace,
                "processing_seconds": np.array([self.mission.processing_seconds[a] for a in self.actors]),
                "dependency_edges": edges, "retained": retained,
                "thermal_locked": np.array([self.mission.thermal_locked[a] for a in self.actors], dtype=np.int8),
                "thermal_enabled": np.array([self.mission.vehicles[a].thermal is not None for a in self.actors], dtype=np.int8),
                "temperature_k": np.array([self.mission.temperature[a] for a in self.actors]),
                "tx_seconds": np.array([self.mission.tx_seconds[a] for a in self.actors]),
                "tx_quota_remaining_sec": np.array([self.mission.horizon_sec if self.mission.vehicles[a].max_tx_sec is None else
                                                    min(self.mission.horizon_sec, max(0., self.mission.vehicles[a].max_tx_sec-self.mission.tx_seconds[a])) for a in self.actors]),
                "station_permitted": np.array([[self.mission.station_permitted(a, s) for s in self.stations] for a in self.actors], dtype=np.int8),
                "reserved": np.array([[self.mission.booking_capacity(a, s)[0] for s in self.stations] for a in self.actors], dtype=np.int8),
                "reservation_required": np.array([self.mission.reservation_required[s] for s in self.stations], dtype=np.int8),
                "open_terminals": np.array([self.mission.booking_capacity(self.actors[0], s)[1] for s in self.stations], dtype=np.int64),
                "transmitter_rates_bps": np.array([self.mission.transmitter_rates[a] for a in self.actors]),
                "station_rates_bps": np.array([max_rate if self.mission.station_rates[s] is None
                                              else min(max_rate, self.mission.station_rates[s]) for s in self.stations]),
                "soc": np.array([view["actors"][a]["soc"] for a in self.actors]),
                "stored_bits": np.array([view["actors"][a]["stored_bits"] for a in self.actors]),
                "contacts": np.array([[s in view["actors"][a]["contacts"] for s in self.stations]
                                      for a in self.actors], dtype=np.int8)}

    def _due(self):
        end = self._cursor
        while end < len(self._events) and self._events[end]["time_sec"] == self.mission.clock:
            end += 1
        if end > self._cursor:
            self.mission.apply_events(self._events[self._cursor:end])
        self._cursor = end

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("reset options unsupported")
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.mission = ReferenceDownlinkMission(self._spec)
        self._cursor, self._done = 0, False
        self._due()
        return self._observe(), self._info()

    def _info(self):
        return {"report": self.mission.report(), "actors": list(self.actors), "stations": list(self.stations),
                "cpu_pause_offset": self._action_width if self.cpu_control else None,
                "cpu_job_selection": self.cpu_job_selection,
                "cpu_commands": {"auto": 0, "pause": 1, "first_product": 2} if self.cpu_job_selection else None,
                "product_ids": {a: [p.id for p in self._slots(a)] for a in self.actors},
                "product_fields": ["remaining_bits", "ready_sec", "deadline_sec", "priority", "destination_index"]}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required after episode completion")
        if not self.action_space.contains(action):
            raise ValueError("invalid joint action")
        checkpoint = copy.deepcopy(self.mission), self._cursor, self._done
        try:
            actions, selections, invalid, processing = {}, {}, [], {}
            cpu_jobs, invalid_cpu = {}, []
            radio = action["radio"] if self.cpu_job_selection else action
            for actor, choice in zip(self.actors, radio):
                choice = int(choice)
                processing[actor] = choice < self._action_width
                choice %= self._action_width
                if int(choice) == 0:
                    continue
                station_i, slot = divmod(int(choice)-1, self.product_slots+1)
                station = self.stations[station_i]
                if slot:
                    rows = self._slots(actor)
                    if slot > len(rows) or (rows[slot-1].destination and rows[slot-1].destination != station):
                        invalid.append(actor)
                        continue
                    selections[actor] = rows[slot-1].id
                actions[actor] = station
            if self.cpu_job_selection:
                for actor, cpu in zip(self.actors, action["cpu"]):
                    cpu = int(cpu)
                    processing[actor] = cpu != 1
                    if cpu >= 2:
                        rows = self._slots(actor)
                        if cpu-2 >= len(rows):
                            processing[actor] = False
                            invalid_cpu.append(actor)
                        else:
                            cpu_jobs[actor] = rows[cpu-2].id
            before = self.mission.report()["on_time_bits"]
            start = self.mission.clock
            end = min(start+self.dt_sec, self.mission.horizon_sec,
                      self._events[self._cursor]["time_sec"] if self._cursor < len(self._events) else self.mission.horizon_sec)
            self.mission.step(end-start, actions, products=selections, processing=processing, processing_jobs=cpu_jobs)
            self._due()
            report = self.mission.report()
            terminated = self._cursor == len(self._events) and all(p["status"] != "incomplete" for p in report["products"])
            truncated = self.mission.clock >= self.mission.horizon_sec-1e-9 and not terminated
            self._done = terminated or truncated
            info = self._info()
            info.update(invalid_actors=invalid, elapsed_sec=self.mission.clock-start,
                        invalid_cpu_actors=invalid_cpu,
                        success=terminated and all(p["status"] == "delivered_on_time" for p in report["products"]))
            return self._observe(), (report["on_time_bits"]-before)/1e6, terminated, truncated, info
        except Exception:
            self.mission, self._cursor, self._done = checkpoint
            raise
