"""Causal downlink operations over fixed GCRS ephemerides.

An explicit simplified resource profile, not a drop-in replacement of MissionWorld.
No acquisition, maneuver, thermal, attitude or calibrated RF-link model is implied.
"""
from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from space_env.core.constants import AU, SOLAR_CONSTANT
from space_env.core.environment import shadow_state
from space_env.core.ground import GroundStation
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.orbit import _ephemerides
from space_env.simulation.snapshot import canonical_json

PROFILE = "reference-fixed-downlink/1"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class VehicleResources(_Config):
    memory_bits: float = Field(gt=0)
    battery_wh: float = Field(gt=0)
    initial_soc: float = Field(default=1., ge=0, le=1)
    min_tx_soc: float = Field(default=.1, ge=0, lt=1)
    base_power_w: float = Field(default=10., ge=0)
    tx_power_w: float = Field(default=20., ge=0)
    rate_bps: float = Field(default=1e6, gt=0)
    solar_area_m2: float = Field(default=0., ge=0)
    solar_efficiency: float = Field(default=.3, ge=0, le=1)
    solar_mode: Literal["ideal_sun_tracking"] = "ideal_sun_tracking"


class OnboardProduct(_Config):
    id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    size_bits: float = Field(ge=1)
    ready_sec: float = Field(default=0., ge=0)
    deadline_sec: float = Field(gt=0)
    priority: int = Field(default=0, ge=0)
    destination: str = ""


class StationOutage(_Config):
    station: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)


class ReferenceDownlinkMission:
    """All products exist onboard at t=0; ready_sec unlocks transmission only.

    Actions map actors to one station or None. Arbitration is alphabetical by actor
    among feasible transmitters; FIFO is replaced by priority/deadline/product ID.
    Resource integration holds absolute-grid midpoint solar samples, with exact boundaries
    for product readiness/completion, contacts, outages and battery thresholds.
    """

    def __init__(self, spec: dict):
        required = {"schema", "ephemerides", "vehicles", "stations", "products"}
        if not required <= spec.keys() or spec.keys() - required - {"outages", "resource_step_sec", "horizon_sec"}:
            raise ValueError("invalid reference operations spec fields")
        if spec["schema"] != "space-env.reference-downlink-spec/1":
            raise ValueError("unsupported reference operations spec")
        canonical_json(spec)
        self.spec = copy.deepcopy(spec)
        self.ephemerides = {name: ReferenceEphemeris(row) for name, row in spec["ephemerides"].items()}
        if not self.ephemerides or set(self.ephemerides) != set(spec["vehicles"]):
            raise ValueError("vehicle resources must match ephemerides")
        if len(self.ephemerides) > 64:
            raise ValueError("this profile is limited to 64 actors")
        self.actors = list(self.ephemerides)
        epoch = self.ephemerides[self.actors[0]].epoch
        if any(abs(row.epoch.seconds_since(epoch)) > 1e-8 for row in self.ephemerides.values()):
            raise ValueError("reference operations require common SI epoch")
        self.vehicles = {name: VehicleResources.model_validate(row) for name, row in spec["vehicles"].items()}
        self.stations = {}
        for row in spec["stations"]:
            if set(row) - set(GroundStation.model_fields):
                raise ValueError("unknown station fields")
            station = GroundStation.model_validate(row)
            if not station.name or station.name in self.stations:
                raise ValueError("station names must be nonempty and unique")
            if station.share_bandwidth or station.band_hz:
                raise ValueError("fixed-rate operations do not support shared bandwidth or RF-band filtering")
            self.stations[station.name] = station
        if not 1 <= len(self.stations) <= 128:
            raise ValueError("this profile requires 1-128 stations")
        self.horizon_sec = float(spec.get("horizon_sec", min(e.duration_sec for e in self.ephemerides.values())))
        if not math.isfinite(self.horizon_sec) or not 1e-9 < self.horizon_sec <= min(e.duration_sec for e in self.ephemerides.values()):
            raise ValueError("horizon must fit every ephemeris")
        self.resource_step_sec = float(spec.get("resource_step_sec", 5.))
        if not math.isfinite(self.resource_step_sec) or not .01 <= self.resource_step_sec <= 30:
            raise ValueError("resource step must be in [.01,30] seconds")
        self.products = {}
        for row in spec["products"]:
            product = OnboardProduct.model_validate(row)
            if product.id in self.products or product.actor not in self.vehicles:
                raise ValueError("duplicate product or unknown actor")
            if product.ready_sec >= product.deadline_sec or product.deadline_sec > self.horizon_sec:
                raise ValueError("product requires ready < deadline <= horizon")
            if product.destination and product.destination not in self.stations:
                raise ValueError("unknown product destination")
            self.products[product.id] = product
        if len(self.products) > 10000:
            raise ValueError("this profile is limited to 10000 products")
        for actor, config in self.vehicles.items():
            if sum(p.size_bits for p in self.products.values() if p.actor == actor) > config.memory_bits:
                raise ValueError("initial onboard inventory exceeds memory")
        self.outages = [StationOutage.model_validate(row) for row in spec.get("outages", [])]
        if any(o.station not in self.stations or o.end_sec <= o.start_sec for o in self.outages):
            raise ValueError("invalid station outage")
        self.windows = {}
        boundaries = {0., self.horizon_sec}
        for actor, ephemeris in self.ephemerides.items():
            for name, station in self.stations.items():
                rows = ephemeris.access(station)["windows"]
                self.windows[actor, name] = [(row["aos_sec"], row["los_sec"]) for row in rows]
                boundaries.update(t for span in self.windows[actor, name] for t in span if t <= self.horizon_sec)
        for product in self.products.values():
            boundaries.update((product.ready_sec, product.deadline_sec))
        for outage in self.outages:
            boundaries.update((outage.start_sec, outage.end_sec))
        self._boundaries = sorted(boundaries)
        self.clock = 0.
        self.remaining = {name: p.size_bits for name, p in self.products.items()}
        self.energy_wh = {name: v.initial_soc*v.battery_wh for name, v in self.vehicles.items()}
        self.unserved_base_wh = dict.fromkeys(self.actors, 0.)
        self.generated_wh = dict.fromkeys(self.actors, 0.)
        self.consumed_wh = dict.fromkeys(self.actors, 0.)
        self.curtailed_wh = dict.fromkeys(self.actors, 0.)
        self.receipts: list[dict] = []
        self.intervals: list[dict] = []

    def observation(self) -> dict:
        return {"time_sec": self.clock, "actors": {actor: {
            "soc": self.energy_wh[actor]/self.vehicles[actor].battery_wh,
            "stored_bits": sum(self.remaining[p.id] for p in self.products.values() if p.actor == actor),
            "contacts": [name for name in self.stations if self._visible(actor, name)],
        } for actor in self.actors}}

    def _visible(self, actor: str, station: str) -> bool:
        return any(lo <= self.clock < hi for lo, hi in self.windows[actor, station]) and not any(
            o.station == station and o.start_sec <= self.clock < o.end_sec for o in self.outages)

    def _product(self, actor: str, station: str) -> OnboardProduct | None:
        ready = [p for p in self.products.values() if p.actor == actor and self.remaining[p.id] > 0
                 and p.ready_sec <= self.clock + 1e-9 and (not p.destination or p.destination == station)]
        return min(ready, key=lambda p: (-p.priority, p.deadline_sec, p.id)) if ready else None

    def _generation(self, actor: str, midpoint: float) -> float:
        config = self.vehicles[actor]
        if config.solar_area_m2 == 0:
            return 0.
        ephemeris = self.ephemerides[actor]
        sun, _moon = _ephemerides(ephemeris.epoch.shifted(midpoint))
        r = ephemeris.state(midpoint)[:3]
        # Explicit ideal independent Sun tracking; no attitude/slew/RF pointing claim.
        return config.solar_area_m2*config.solar_efficiency*SOLAR_CONSTANT*(AU/float((sum(x*x for x in sun))**.5))**2*shadow_state(r, sun).flux_fraction

    def step(self, dt_sec: float, actions: dict[str, str | None]) -> dict:
        if isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec <= 1e-9 or self.clock+dt_sec > self.horizon_sec+1e-9:
            raise ValueError("step must exceed 1ns, be finite and within horizon")
        if actions.keys() - self.vehicles.keys() or any(s is not None and s not in self.stations for s in actions.values()):
            raise ValueError("unknown actor or station action")
        checkpoint = (self.clock, dict(self.remaining), dict(self.energy_wh), dict(self.unserved_base_wh),
                      dict(self.generated_wh), dict(self.consumed_wh), dict(self.curtailed_wh),
                      len(self.receipts), len(self.intervals))
        try:
            end = min(self.clock+dt_sec, self.horizon_sec)
            while self.clock < end-1e-9:
                right = min(end, (math.floor((self.clock+1e-9)/self.resource_step_sec)+1)*self.resource_step_sec,
                            *[t for t in self._boundaries if t > self.clock+1e-9])
                span = right-self.clock
                # Fixed absolute resource grid, independent of caller's dt or
                # subsequent completion/threshold subdivision of this interval.
                sample = min((math.floor((self.clock+1e-9)/self.resource_step_sec)+.5)*self.resource_step_sec,
                             self.horizon_sec)
                generation = {a: self._generation(a, sample) for a in self.actors}
                chosen, used = {}, dict.fromkeys(self.stations, 0)
                for actor in sorted(actions):
                    station = actions[actor]
                    config = self.vehicles[actor]
                    if station is None or not self._visible(actor, station):
                        continue
                    product = self._product(actor, station)
                    tx_net = generation[actor]-config.base_power_w-config.tx_power_w
                    floor = config.min_tx_soc*config.battery_wh
                    if product is None or used[station] >= self.stations[station].terminals:
                        continue
                    if self.energy_wh[actor] <= floor+1e-12 and tx_net < 0:
                        continue
                    if self.energy_wh[actor] < floor-1e-12:
                        continue
                    chosen[actor] = (station, product)
                    used[station] += 1
                    span = min(span, self.remaining[product.id]/config.rate_bps)
                    if tx_net < 0:
                        span = min(span, max(0., (self.energy_wh[actor]-floor)*3600/-tx_net))
                for actor in self.actors:
                    config = self.vehicles[actor]
                    idle_net = generation[actor]-config.base_power_w
                    floor = config.min_tx_soc*config.battery_wh
                    if actor not in chosen and self.energy_wh[actor] < floor-1e-12 and idle_net > 0:
                        span = min(span, (floor-self.energy_wh[actor])*3600/idle_net)
                if span <= 1e-10:
                    raise RuntimeError("resource boundary made no progress")
                for actor in self.actors:
                    config = self.vehicles[actor]
                    net = generation[actor]-config.base_power_w-(config.tx_power_w if actor in chosen else 0)
                    generated = generation[actor]*span/3600
                    demand = (config.base_power_w+(config.tx_power_w if actor in chosen else 0))*span/3600
                    candidate = self.energy_wh[actor]+net*span/3600
                    self.generated_wh[actor] += generated
                    self.consumed_wh[actor] += demand-max(0., -candidate)
                    self.curtailed_wh[actor] += max(0., candidate-config.battery_wh)
                    if candidate < 0:
                        self.unserved_base_wh[actor] += -candidate
                    self.energy_wh[actor] = min(config.battery_wh, max(0., candidate))
                for actor, (station, product) in chosen.items():
                    rate = self.vehicles[actor].rate_bps
                    bits = self.remaining[product.id] if span >= self.remaining[product.id]/rate else rate*span
                    self.remaining[product.id] = max(0., self.remaining[product.id]-bits)
                    self.receipts.append({"product_id": product.id, "actor": actor, "station": station,
                                          "start_sec": self.clock, "end_sec": self.clock+span, "bits": bits})
                self.intervals.append({"start_sec": self.clock, "end_sec": self.clock+span,
                                       "transmitters": {a: pair[0] for a, pair in chosen.items()}})
                self.clock += span
            return self.report()
        except Exception:
            (self.clock, self.remaining, self.energy_wh, self.unserved_base_wh,
             self.generated_wh, self.consumed_wh, self.curtailed_wh, receipts, intervals) = checkpoint
            del self.receipts[receipts:]
            del self.intervals[intervals:]
            raise

    def report(self) -> dict:
        outcomes = []
        for product in self.products.values():
            slips = [r for r in self.receipts if r["product_id"] == product.id]
            complete = self.remaining[product.id] == 0
            completed_at = max((r["end_sec"] for r in slips), default=None) if complete else None
            status = ("delivered_on_time" if completed_at <= product.deadline_sec+1e-9 else "delivered_late") if complete else "incomplete"
            outcomes.append({"id": product.id, "status": status, "completed_sec": completed_at,
                             "delivered_bits": product.size_bits-self.remaining[product.id],
                             "remaining_bits": self.remaining[product.id]})
        return {"profile": PROFILE, "time_sec": self.clock, "observation": self.observation(),
                "products": outcomes, "receipts": copy.deepcopy(self.receipts),
                "energy_wh": dict(self.energy_wh), "unserved_base_wh": dict(self.unserved_base_wh),
                "energy_ledger": {actor: {"generated_wh": self.generated_wh[actor],
                                         "consumed_wh": self.consumed_wh[actor], "curtailed_wh": self.curtailed_wh[actor],
                                         "balance_error_wh": self.vehicles[actor].initial_soc*self.vehicles[actor].battery_wh
                                         + self.generated_wh[actor]-self.consumed_wh[actor]-self.curtailed_wh[actor]-self.energy_wh[actor]}
                                  for actor in self.actors},
                "on_time_bits": sum(self.products[row["id"]].size_bits for row in outcomes if row["status"] == "delivered_on_time")}
