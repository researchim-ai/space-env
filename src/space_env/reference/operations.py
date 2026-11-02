"""Causal downlink operations over fixed GCRS ephemerides.

An explicit simplified resource profile, not a drop-in replacement of MissionWorld.
No acquisition, maneuver, thermal, attitude or calibrated RF-link model is implied.
"""
from __future__ import annotations

import copy
import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from space_env.core.constants import AU, SOLAR_CONSTANT
from space_env.core.environment import shadow_state
from space_env.core.ground import GroundStation
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.dependencies import dependency_children, dependency_order
from space_env.reference.orbit import _ephemerides
from space_env.reference.thermal import TransmitterThermal
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
    solar_mode: Literal["ideal_sun_tracking", "body_fixed_bus"] = "ideal_sun_tracking"
    max_tx_sec: float | None = Field(default=None, ge=0)
    thermal: TransmitterThermal | None = None
    processing_power_w: float = Field(default=0., ge=0)
    min_processing_soc: float = Field(default=.1, ge=0, lt=1)

    @model_validator(mode="after")
    def heat_budget(self):
        if self.thermal is not None and self.thermal.heat_tx_w > self.tx_power_w:
            raise ValueError("transmitter heat cannot exceed electrical TX power")
        return self


class OnboardProduct(_Config):
    id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    size_bits: float = Field(ge=1)
    ready_sec: float = Field(default=0., ge=0)
    deadline_sec: float = Field(gt=0)
    priority: int = Field(default=0, ge=0)
    destination: str = ""
    processing_sec: float = Field(default=0., ge=0)
    workspace_bits: float = Field(default=0., ge=0)
    depends_on: tuple[str, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def dependency_work(self):
        if self.depends_on and self.processing_sec <= 0:
            raise ValueError("dependent products require positive processing work")
        return self


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
    Optional per-step products selections pin an actor to one owned product;
    completion waits rather than automatically advancing its queue.
    """

    def __init__(self, spec: dict):
        required = {"schema", "ephemerides", "vehicles", "stations", "products"}
        if not required <= spec.keys() or spec.keys() - required - {"outages", "resource_step_sec", "horizon_sec", "radio_network", "dtn_network"}:
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
        if any(v.solar_mode == "body_fixed_bus" for v in self.vehicles.values()) and not getattr(self, "supports_body_fixed_solar", False):
            raise ValueError("body_fixed_bus solar requires the coupled spacecraft profile")
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
            if product.workspace_bits > self.vehicles[product.actor].memory_bits:
                raise ValueError("processing workspace exceeds memory")
            self.products[product.id] = product
        if len(self.products) > 10000:
            raise ValueError("this profile is limited to 10000 products")
        self._dependency_order = dependency_order(self.products)
        self._dependency_children = dependency_children(self.products)
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
        self.initial_bits = {a: sum(p.size_bits for p in self.products.values() if p.actor == a)
                             for a in self.actors}
        self.energy_wh = {name: v.initial_soc*v.battery_wh for name, v in self.vehicles.items()}
        self.unserved_base_wh = dict.fromkeys(self.actors, 0.)
        self.generated_wh = dict.fromkeys(self.actors, 0.)
        self.consumed_wh = dict.fromkeys(self.actors, 0.)
        self.curtailed_wh = dict.fromkeys(self.actors, 0.)
        self.receipts: list[dict] = []
        self.intervals: list[dict] = []
        self.events: list[dict] = []
        self.cancelled: dict[str, float] = {}
        self.discarded = dict.fromkeys(self.products, 0.)
        self.transmitter_rates = {a: v.rate_bps for a, v in self.vehicles.items()}
        self.station_rates: dict[str, float | None] = dict.fromkeys(self.stations)
        self.bookings = {}
        self.external_reservations = {}
        self.booking_cancelled = {}
        self.reservation_required = dict.fromkeys(self.stations, False)
        self.tx_seconds = dict.fromkeys(self.actors, 0.)
        self.temperature = {a: v.thermal.initial_k if v.thermal else 0. for a, v in self.vehicles.items()}
        self.thermal_locked = {a: bool(v.thermal and v.thermal.initial_k >= v.thermal.trip_k-1e-9) for a, v in self.vehicles.items()}
        self.heat_j = dict.fromkeys(self.actors, 0.)
        self.rejected_heat_j = dict.fromkeys(self.actors, 0.)
        self.processing_remaining = {p.id: p.processing_sec for p in self.products.values()}
        self.processing_seconds = dict.fromkeys(self.actors, 0.)
        self.processing_completed = {p.id: None for p in self.products.values()}
        self.processing_intervals: list[dict] = []
        self.radio_network = None
        if spec.get("radio_network") is not None:
            from space_env.reference.radio_network import RadioGroundNetwork
            self.radio_network = RadioGroundNetwork(spec["radio_network"], self)
        self.dtn = None
        self.rx_seconds = dict.fromkeys(self.actors, 0.)
        self.dtn_lane_bandwidth = {}
        if spec.get("dtn_network") is not None:
            from space_env.reference.dtn import CustodyNetwork
            self.dtn = CustodyNetwork(spec["dtn_network"], self)

    def transport_memory(self, actor):
        return 0. if self.dtn is None else self.dtn.reserved(actor)

    def processed_products(self) -> set[str]:
        completed = set()
        for name in self._dependency_order:
            p = self.products[name]
            if (name not in self.cancelled and self.processing_remaining[name] == 0
                    and p.ready_sec <= self.clock+1e-9):
                completed.add(name)
        return completed

    def retained_for(self, product_id: str) -> list[str]:
        return [name for name in self._dependency_children[product_id]
                if self.remaining[name] > 0 and self.processing_remaining[name] > 0]

    def processing_status(self, product_id: str, generation_w: float = 0.) -> dict:
        """Current CPU admission only; generation is an explicit supplied sample."""
        if isinstance(generation_w, bool) or not math.isfinite(generation_w) or generation_w < 0:
            raise ValueError("generation sample must be finite and nonnegative")
        p = self.products[product_id]
        config = self.vehicles[p.actor]
        reason = "eligible"
        if p.id in self.cancelled:
            reason = "cancelled"
        elif self.processing_remaining[p.id] == 0:
            reason = "completed"
        elif p.ready_sec > self.clock+1e-9:
            reason = "not_ready"
        elif p.depends_on and not set(p.depends_on) <= self.processed_products():
            reason = "dependency"
        elif sum(self.remaining[q.id] for q in self.products.values() if q.actor == p.actor)+self.transport_memory(p.actor)+p.workspace_bits > config.memory_bits:
            reason = "memory"
        else:
            floor = config.min_processing_soc*config.battery_wh
            if self.energy_wh[p.actor] < floor-1e-12 or (self.energy_wh[p.actor] <= floor+1e-12
                    and generation_w-config.base_power_w-config.processing_power_w < 0):
                reason = "power"
        return {"eligible": reason == "eligible", "reason": reason, "generation_sample_w": generation_w}

    def processing_job(self, actor: str, generation_w: float = 0., product_id: str | None = None) -> OnboardProduct | None:
        config = self.vehicles[actor]
        floor = config.min_processing_soc*config.battery_wh
        if self.energy_wh[actor] < floor-1e-12 or (self.energy_wh[actor] <= floor+1e-12
                and generation_w-config.base_power_w-config.processing_power_w < 0):
            return None
        occupied = sum(self.remaining[p.id] for p in self.products.values() if p.actor == actor)+self.transport_memory(actor)
        processed = self.processed_products()
        jobs = [p for p in self.products.values() if p.actor == actor and self.remaining[p.id] > 0
                and self.processing_remaining[p.id] > 0 and p.ready_sec <= self.clock+1e-9
                and occupied+p.workspace_bits <= config.memory_bits]
        jobs = [p for p in jobs if all(parent in processed for parent in p.depends_on)]
        if product_id is not None:
            jobs = [p for p in jobs if p.id == product_id]
        return min(jobs, key=lambda p: (-p.priority, p.deadline_sec, p.id)) if jobs else None

    def transmitter_ready(self, actor: str) -> bool:
        quota = self.vehicles[actor].max_tx_sec
        return not self.thermal_locked[actor] and (quota is None or self.tx_seconds[actor] < quota-1e-10)

    def transmitter_resume_sec(self, actor: str) -> float | None:
        v = self.vehicles[actor]
        if v.max_tx_sec is not None and self.tx_seconds[actor] >= v.max_tx_sec-1e-10:
            return None
        if self.thermal_locked[actor]:
            return self.clock+v.thermal.crossing(self.temperature[actor], v.thermal.resume_k, False)
        return self.clock

    def booking_capacity(self, actor: str, station: str) -> tuple[bool, int]:
        return self._booking_capacity_at(actor, station, self.clock)

    def _booking_capacity_at(self, actor: str, station: str, time_sec: float) -> tuple[bool, int]:
        active = [b for b in self.bookings.values() if b.station == station
                  and b.start_sec <= time_sec < min(b.end_sec, self.booking_cancelled.get(b.id, b.end_sec))]
        return any(b.actor == actor for b in active), self.stations[station].terminals-len(active)

    def station_permitted(self, actor: str, station: str) -> bool:
        reserved, free = self.booking_capacity(actor, station)
        return reserved or (free > 0 and not self.reservation_required[station])

    def service_windows(self, actor: str, station: str) -> list[dict]:
        """Known geometry/outages/bookings; current policies/rates held constant.

        Does not forecast energy, traffic, or future unobserved operator events.
        """
        if actor not in self.vehicles or station not in self.stations:
            raise ValueError("unknown service-window actor or station")
        if self.radio_network is None and self.effective_rate(actor, station) == 0:
            return []
        edges = sorted({self.clock, self.horizon_sec,
                        *[t for t in self._boundaries if self.clock < t < self.horizon_sec]})
        result = []
        for lo, hi in pairwise(edges):
            middle = (lo+hi)/2
            reserved, free = self._booking_capacity_at(actor, station, middle)
            visible = any(a <= middle < b for a, b in self.windows[actor, station])
            outage = any(o.station == station and o.start_sec <= middle < o.end_sec for o in self.outages)
            if not visible or outage or not (reserved or (free > 0 and not self.reservation_required[station])):
                continue
            mode = "reserved" if reserved else "open"
            if result and result[-1]["end_sec"] == lo and result[-1]["mode"] == mode:
                result[-1]["end_sec"] = hi
            else:
                result.append({"start_sec": lo, "end_sec": hi, "mode": mode})
        return result

    def effective_rate(self, actor: str, station: str) -> float:
        """Configured per-terminal throughput ceiling, not a calibrated RF model."""
        cap = self.station_rates[station]
        baseline = min(self.transmitter_rates[actor], cap) if cap is not None else self.transmitter_rates[actor]
        return baseline if self.radio_network is None else min(baseline, self.radio_network.rate(actor, station))

    def apply_events(self, events: list[dict]) -> dict:
        from space_env.reference.events import apply_events

        return apply_events(self, events)

    def observation(self) -> dict:
        return {"time_sec": self.clock, "actors": {actor: {
            "soc": self.energy_wh[actor]/self.vehicles[actor].battery_wh,
            "stored_bits": sum(self.remaining[p.id] for p in self.products.values() if p.actor == actor),
            "contacts": [name for name in self.stations if self._visible(actor, name)],
            "transmitter_rate_bps": self.transmitter_rates[actor],
            **({"dtn":self.dtn.observation(actor)} if self.dtn is not None else {}),
            **({"radio_network":self.radio_network.observation(self, actor)} if self.radio_network is not None else {}),
        } for actor in self.actors}}

    def _visible(self, actor: str, station: str) -> bool:
        return any(lo <= self.clock < hi for lo, hi in self.windows[actor, station]) and not any(
            o.station == station and o.start_sec <= self.clock < o.end_sec for o in self.outages)

    def _product(self, actor: str, station: str) -> OnboardProduct | None:
        ready = [p for p in self.products.values() if p.actor == actor and self.remaining[p.id] > 0
                 and self.processing_remaining[p.id] == 0
                 and not self.retained_for(p.id)
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

    def select_transmissions(self, actions, selections, *, processors=(), generation, vehicles=None, ignore_radio=False):
        """Readonly native station/booking/rate/SOC admission shared by executors."""
        vehicles = self.vehicles if vehicles is None else vehicles
        chosen, used = {}, dict.fromkeys(self.stations, 0)
        open_used = dict.fromkeys(self.stations, 0)
        for actor in sorted(actions):
            station = actions[actor]
            config = vehicles[actor]
            if station is None or not self._visible(actor, station) or not self.transmitter_ready(actor) or actor in processors or (self.dtn is not None and self.dtn.receiving(actor)):
                continue
            reserved, free = self.booking_capacity(actor, station)
            if not reserved and (self.reservation_required[station] or open_used[station] >= free):
                continue
            product = self.products[selections[actor]] if actor in selections else self._product(actor, station)
            if product is not None and self.dtn is not None and self.dtn.pinned(product.id):
                product = None
            if product is not None and (self.remaining[product.id] <= 0 or product.ready_sec > self.clock+1e-9
                                        or self.processing_remaining[product.id] > 0 or self.retained_for(product.id)):
                product = None
            net = generation[actor]-config.base_power_w-config.tx_power_w
            floor = config.min_tx_soc*config.battery_wh
            rate = self.transmitter_rates[actor] if ignore_radio else self.effective_rate(actor, station)
            if ignore_radio and self.radio_network is not None:
                rate = self.radio_network.budget(self, actor, station)["rate_bps"]
            if product is None or used[station] >= self.stations[station].terminals or rate == 0:
                continue
            if self.energy_wh[actor] < floor-1e-12 or (self.energy_wh[actor] <= floor+1e-12 and net < 0):
                continue
            chosen[actor] = (station, product)
            used[station] += 1
            if not reserved:
                open_used[station] += 1
        return chosen

    def step(self, dt_sec: float, actions: dict[str, str | None], *,
             products: dict[str, str] | None = None, processing: dict[str, bool] | None = None,
             processing_jobs: dict[str, str] | None = None, relay_controls=(), relay_blocked=()) -> dict:
        if isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec <= 1e-9 or self.clock+dt_sec > self.horizon_sec+1e-9:
            raise ValueError("step must exceed 1ns, be finite and within horizon")
        if actions.keys() - self.vehicles.keys() or any(s is not None and s not in self.stations for s in actions.values()):
            raise ValueError("unknown actor or station action")
        if relay_controls and (self.dtn is None or any(t not in self.dtn.transfers for t in relay_controls)):
            raise ValueError("relay controls require configured admitted transfer IDs")
        if set(relay_blocked)-set(self.actors):
            raise ValueError("unknown blocked relay endpoint")
        selections = {} if products is None else dict(products)
        processing = {} if processing is None else dict(processing)
        jobs = {} if processing_jobs is None else dict(processing_jobs)
        if processing.keys() - self.vehicles.keys() or any(not isinstance(v, bool) for v in processing.values()):
            raise ValueError("processing controls require known actors and boolean values")
        for actor, name in jobs.items():
            if (not isinstance(name, str) or name not in self.products or self.products[name].actor != actor
                    or not processing.get(actor, True)):
                raise ValueError("selected CPU job requires its owner and enabled processing")
        for actor, product_id in selections.items():
            product = self.products.get(product_id)
            station = actions.get(actor)
            if (product is None or product.actor != actor or station is None
                    or (product.destination and product.destination != station)):
                raise ValueError("selected product requires its owner and a compatible station action")
        checkpoint = (self.clock, dict(self.remaining), dict(self.energy_wh), dict(self.unserved_base_wh),
                      dict(self.generated_wh), dict(self.consumed_wh), dict(self.curtailed_wh),
                      dict(self.tx_seconds), dict(self.temperature), dict(self.thermal_locked), dict(self.heat_j), dict(self.rejected_heat_j),
                      dict(self.processing_remaining), dict(self.processing_seconds), dict(self.processing_completed), len(self.processing_intervals),
                      len(self.receipts), len(self.intervals))
        radio_checkpoint = None if self.radio_network is None else copy.deepcopy(self.radio_network.__dict__)
        dtn_checkpoint = None if self.dtn is None else copy.deepcopy(self.dtn.__dict__)
        product_checkpoint, rx_checkpoint = dict(self.products), dict(self.rx_seconds)
        custody_inventory = None if self.dtn is None else (dict(self.cancelled),dict(self.discarded))
        try:
            end = min(self.clock+dt_sec, self.horizon_sec)
            while self.clock < end-1e-9:
                if self.dtn is not None:
                    self.dtn.settle(self)
                right = min(end, (math.floor((self.clock+1e-9)/self.resource_step_sec)+1)*self.resource_step_sec,
                            *[t for t in self._boundaries if t > self.clock+1e-9])
                span = right-self.clock
                # Fixed absolute resource grid, independent of caller's dt or
                # subsequent completion/threshold subdivision of this interval.
                sample = min((math.floor((self.clock+1e-9)/self.resource_step_sec)+.5)*self.resource_step_sec,
                             self.horizon_sec)
                generation = {a: self._generation(a, sample) for a in self.actors}
                processors = {a: job for a in self.actors if processing.get(a, True)
                              and not (self.dtn is not None and self.dtn.receiving(a))
                              and (job := self.processing_job(a, generation[a], jobs.get(a))) is not None}
                for actor, job in processors.items():
                    config = self.vehicles[actor]
                    span = min(span, self.processing_remaining[job.id])
                    net = generation[actor]-config.base_power_w-config.processing_power_w
                    if net < 0:
                        span = min(span, max(0., (self.energy_wh[actor]-config.min_processing_soc*config.battery_wh)*3600/-net))
                if self.radio_network is not None:
                    candidates = self.select_transmissions(actions, selections, processors=processors, generation=generation, ignore_radio=True)
                    self.radio_network.prepare(self, candidates)
                    right = self.radio_network.boundary(self.clock, right)
                    span = min(span, right-self.clock)
                chosen = self.select_transmissions(actions, selections, processors=processors, generation=generation)
                relay_active = {}
                if self.dtn is not None:
                    relay_active = self.dtn.prepare(self, relay_controls, chosen, processors, generation, relay_blocked)
                    span = min(span,self.dtn.boundary(self,self.clock+span)-self.clock)
                    for a,item in relay_active.items():
                        bus = getattr(self,'bus',None)
                        if bus is not None:
                            item['floor'] = max(item['floor'],bus.network.heater_floor(a))
                        net = generation[a]-self.vehicles[a].base_power_w-item['power_w']
                        if net < 0:
                            span = min(span,max(0.,(self.energy_wh[a]-item['floor']*self.vehicles[a].battery_wh)*3600/-net))
                        if item['role']=='tx':
                            v=self.vehicles[a]
                            if v.max_tx_sec is not None:
                                span=min(span,v.max_tx_sec-self.tx_seconds[a])
                            if v.thermal is not None:
                                span=min(span,v.thermal.crossing(self.temperature[a],v.thermal.trip_k,True))
                for actor, (station, product) in chosen.items():
                    config = self.vehicles[actor]
                    rate = self.effective_rate(actor, station)
                    tx_net = generation[actor]-config.base_power_w-config.tx_power_w
                    floor = config.min_tx_soc*config.battery_wh
                    span = min(span, self.remaining[product.id]/rate)
                    if config.max_tx_sec is not None:
                        span = min(span, config.max_tx_sec-self.tx_seconds[actor])
                    if config.thermal:
                        span = min(span, config.thermal.crossing(self.temperature[actor], config.thermal.trip_k, True))
                    if tx_net < 0:
                        span = min(span, max(0., (self.energy_wh[actor]-floor)*3600/-tx_net))
                for actor in self.actors:
                    config = self.vehicles[actor]
                    idle_net = generation[actor]-config.base_power_w
                    floor = config.min_tx_soc*config.battery_wh
                    if actor not in chosen and self.thermal_locked[actor]:
                        span = min(span, config.thermal.crossing(self.temperature[actor], config.thermal.resume_k, False))
                    for threshold in (floor, config.min_processing_soc*config.battery_wh):
                        if actor not in chosen and actor not in processors and self.energy_wh[actor] < threshold-1e-12 and idle_net > 0:
                            span = min(span, (threshold-self.energy_wh[actor])*3600/idle_net)
                if span <= 1e-10:
                    raise RuntimeError("resource boundary made no progress")
                for actor in self.actors:
                    config = self.vehicles[actor]
                    relay = relay_active.get(actor)
                    transmitting = actor in chosen or (relay is not None and relay['role']=='tx')
                    if transmitting:
                        self.tx_seconds[actor] += span
                    if relay is not None and relay['role']=='rx':
                        self.rx_seconds[actor] += span
                    if config.thermal:
                        old = self.temperature[actor]
                        new = config.thermal.advance(old, span, transmitting)
                        heat = config.thermal.heat_tx_w*span if transmitting else 0.
                        self.heat_j[actor] += heat
                        self.rejected_heat_j[actor] += heat-config.thermal.capacity_j_per_k*(new-old)
                        self.temperature[actor] = new
                        if new >= config.thermal.trip_k-1e-9:
                            self.thermal_locked[actor] = True
                        elif new <= config.thermal.resume_k+1e-9:
                            self.thermal_locked[actor] = False
                    load = config.base_power_w+(config.tx_power_w if actor in chosen else 0)+(config.processing_power_w if actor in processors else 0)+(relay["power_w"] if relay is not None else 0)
                    net = generation[actor]-load
                    generated = generation[actor]*span/3600
                    demand = load*span/3600
                    candidate = self.energy_wh[actor]+net*span/3600
                    self.generated_wh[actor] += generated
                    self.consumed_wh[actor] += demand-max(0., -candidate)
                    self.curtailed_wh[actor] += max(0., candidate-config.battery_wh)
                    if candidate < 0:
                        self.unserved_base_wh[actor] += -candidate
                    self.energy_wh[actor] = min(config.battery_wh, max(0., candidate))
                for actor, job in processors.items():
                    needed = self.processing_remaining[job.id]
                    ulp = max(math.ulp(self.clock), math.ulp(self.clock+span), math.ulp(needed))
                    complete = span >= needed or abs(span-needed) <= 4*ulp
                    self.processing_remaining[job.id] = 0. if complete else needed-span
                    self.processing_seconds[actor] += span
                    if complete:
                        self.processing_completed[job.id] = self.clock+span
                    occupied = sum(self.remaining[p.id] for p in self.products.values() if p.actor == actor)+self.transport_memory(actor)
                    self.processing_intervals.append({"actor": actor, "product_id": job.id, "start_sec": self.clock,
                                                      "end_sec": self.clock+span, "workspace_bits": job.workspace_bits,
                                                      "inventory_bits": occupied, "total_memory_bits": occupied+job.workspace_bits})
                receipt_start = len(self.receipts)
                for actor, (station, product) in chosen.items():
                    rate = self.effective_rate(actor, station)
                    duration = self.remaining[product.id]/rate
                    # Subtracting absolute endpoints may lose a few ulps. Snap
                    # only at the completion time's representational precision,
                    # never using a fixed remaining-bits success threshold.
                    endpoint_ulp = max(math.ulp(self.clock), math.ulp(self.clock+span), math.ulp(duration))
                    complete = span >= duration or abs(span-duration) <= 4*endpoint_ulp
                    bits = self.remaining[product.id] if complete else rate*span
                    self.remaining[product.id] = max(0., self.remaining[product.id]-bits)
                    self.receipts.append({"product_id": product.id, "actor": actor, "station": station,
                                          "start_sec": self.clock, "end_sec": self.clock+span, "bits": bits})
                self.intervals.append({"start_sec": self.clock, "end_sec": self.clock+span,
                                       "transmitters": {a: pair[0] for a, pair in chosen.items()}})
                if self.radio_network is not None:
                    self.radio_network.integrate(self, self.clock, span, chosen, receipt_start)
                if self.dtn is not None:
                    self.dtn.integrate(self,self.clock,span,relay_active)
                self.clock += span
            if self.dtn is not None:
                self.dtn.settle(self)
            return self.report()
        except Exception:
            self.products, self.rx_seconds = product_checkpoint, rx_checkpoint
            if custody_inventory is not None:
                self.cancelled,self.discarded = custody_inventory
            if dtn_checkpoint is not None:
                self.dtn.__dict__ = dtn_checkpoint
            if radio_checkpoint is not None:
                self.radio_network.__dict__ = radio_checkpoint
            (self.clock, self.remaining, self.energy_wh, self.unserved_base_wh,
             self.generated_wh, self.consumed_wh, self.curtailed_wh,
             self.tx_seconds, self.temperature, self.thermal_locked, self.heat_j, self.rejected_heat_j,
             self.processing_remaining, self.processing_seconds, self.processing_completed, processing_intervals, receipts, intervals) = checkpoint
            del self.processing_intervals[processing_intervals:]
            del self.receipts[receipts:]
            del self.intervals[intervals:]
            raise

    def acknowledged_time(self, product_id):
        if self.remaining[product_id] != 0 or product_id in self.cancelled:
            return None
        slips = [r for r in self.receipts if r["product_id"] == product_id]
        if not slips:
            return None
        due = max(r.get("ack_due_sec", r["end_sec"]) for r in slips)
        return due if due <= self.clock+4*math.ulp(self.clock) else None

    def report(self) -> dict:
        outcomes = []
        processed = self.processed_products()
        for product in self.products.values():
            slips = [r for r in self.receipts if r["product_id"] == product.id]
            completed_at = self.acknowledged_time(product.id)
            complete = completed_at is not None
            status = ("delivered_on_time" if completed_at <= product.deadline_sec+1e-9 else "delivered_late") if complete else "incomplete"
            if product.id in self.cancelled:
                status = "cancelled"
            outcomes.append({"id": product.id, "status": status, "completed_sec": completed_at,
                             "cancelled_sec": self.cancelled.get(product.id),
                             "discarded_bits": self.discarded[product.id],
                             "processing_remaining_sec": self.processing_remaining[product.id],
                             "processing_completed_sec": self.processing_completed[product.id],
                             "processing_admission": self.processing_status(product.id),
                             "depends_on": list(product.depends_on),
                             "dependency_blockers": [p for p in product.depends_on if p not in processed]
                                 if self.processing_remaining[product.id] > 0 and self.remaining[product.id] > 0 else [],
                             "retained_for": self.retained_for(product.id),
                             "downlink_ready": self.remaining[product.id] > 0 and self.processing_remaining[product.id] == 0
                                               and product.ready_sec <= self.clock+1e-9 and not self.retained_for(product.id),
                             "delivered_bits": sum(r["bits"] for r in slips),
                             "remaining_bits": self.remaining[product.id]})
        inventory = {}
        for actor in self.actors:
            initial = self.initial_bits[actor]
            injected = sum(self.products[e["payload"]["id"]].size_bits for e in self.events
                           if e["type"] == "onboard_product_added" and e["payload"]["actor"] == actor)
            delivered = sum(r["bits"] for r in self.receipts if r["actor"] == actor)
            remaining = sum(self.remaining[p.id] for p in self.products.values() if p.actor == actor)
            discarded = sum(self.discarded[p.id] for p in self.products.values() if p.actor == actor)
            custody_in,custody_out = (0.,0.) if self.dtn is None else self.dtn.custody(actor)
            inventory[actor] = {**({"custody_in_bits":custody_in,"custody_out_bits":custody_out} if self.dtn is not None else {}),"initial_bits": initial, "injected_bits": injected,
                                "delivered_bits": delivered, "remaining_bits": remaining,
                                "discarded_bits": discarded,
                                "balance_error_bits": initial+injected+custody_in-custody_out-delivered-remaining-discarded}
        utilization = []
        for b in self.bookings.values():
            end = min(self.clock, b.end_sec, self.booking_cancelled.get(b.id, b.end_sec))
            elapsed = max(0., end-b.start_sec)
            transmitting = sum(max(0., min(end, r["end_sec"])-max(b.start_sec, r["start_sec"]))
                               for r in self.receipts if r["actor"] == b.actor and r["station"] == b.station)
            utilization.append({"booking_id": b.id, "elapsed_reserved_sec": elapsed,
                                "transmitting_sec": transmitting, "idle_reserved_sec": elapsed-transmitting,
                                "utilization": transmitting/elapsed if elapsed else None})
        return {"profile": PROFILE, "time_sec": self.clock, "observation": self.observation(),
                **({"dtn":self.dtn.report(self)} if self.dtn is not None else {}),
                **({"radio_network":self.radio_network.report(self)} if self.radio_network is not None else {}),
                "products": outcomes, "receipts": copy.deepcopy(self.receipts),
                "events": copy.deepcopy(self.events),
                "inventory_ledger": inventory,
                "rate_limits": {"transmitters_bps": dict(self.transmitter_rates),
                                "station_per_terminal_bps": dict(self.station_rates)},
                "bookings": [dict(b.model_dump(), cancelled_sec=self.booking_cancelled.get(b.id)) for b in self.bookings.values()],
                "reservation_required": dict(self.reservation_required),
                "external_reservations": copy.deepcopy(self.external_reservations),
                "booking_utilization": utilization,
                "processing": {"seconds": dict(self.processing_seconds),
                               "energy_wh": {a: self.processing_seconds[a]*v.processing_power_w/3600 for a, v in self.vehicles.items()},
                               "intervals": copy.deepcopy(self.processing_intervals)},
                "transmitter_health": {a: {"tx_seconds": self.tx_seconds[a],
                    "quota_remaining_sec": None if v.max_tx_sec is None else max(0., v.max_tx_sec-self.tx_seconds[a]),
                    "temperature_k": self.temperature[a] if v.thermal else None,
                    "thermal_locked": self.thermal_locked[a], "ready": self.transmitter_ready(a),
                    "heat_input_j": self.heat_j[a], "rejected_heat_j": self.rejected_heat_j[a],
                    "thermal_balance_error_j": (self.heat_j[a]-self.rejected_heat_j[a]
                        -v.thermal.capacity_j_per_k*(self.temperature[a]-v.thermal.initial_k)) if v.thermal else 0.}
                    for a, v in self.vehicles.items()},
                "energy_wh": dict(self.energy_wh), "unserved_base_wh": dict(self.unserved_base_wh),
                "energy_ledger": {actor: {"generated_wh": self.generated_wh[actor],
                                         "consumed_wh": self.consumed_wh[actor], "curtailed_wh": self.curtailed_wh[actor],
                                         "balance_error_wh": self.vehicles[actor].initial_soc*self.vehicles[actor].battery_wh
                                         + self.generated_wh[actor]-self.consumed_wh[actor]-self.curtailed_wh[actor]-self.energy_wh[actor]}
                                  for actor in self.actors},
                "on_time_bits": sum(self.products[row["id"]].size_bits for row in outcomes if row["status"] == "delivered_on_time")}
