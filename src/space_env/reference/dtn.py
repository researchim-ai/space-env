"""Bounded store-and-forward custody transport over the reference mission clock.

Engineering packet model, not BPv7/BIBE/LTP wire interoperability. Stop-and-wait
fragments retain source bytes until whole-product custody confirmation.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import R_EARTH_EQUATOR
from space_env.core.link import earth_occults
from space_env.reference.radio import C_M_S, RFReceiver, rf_budget
from space_env.runtime.schema import StrictModel

TERMINAL = {"completed", "failed", "cancelled", "expired"}


class ReceiverResources(StrictModel):
    power_w: float = Field(gt=0, le=1e6, strict=True)
    min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)
    memory_bandwidth_bps: float = Field(default=1e9, gt=0, le=1e15, strict=True)


class ISLLink(StrictModel):
    source: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    receiver: RFReceiver
    max_range_m: float = Field(default=1e7, gt=1.0, le=1e10, strict=True)
    acquisition_sec: float = Field(default=0.1, ge=0, le=3600, strict=True)
    rx_beamwidth_deg: float | None = Field(default=None, gt=0, le=180, strict=True)
    loss_probability: float = Field(default=0.0, ge=0, le=1, strict=True)
    ack_loss_probability: float = Field(default=0.0, ge=0, le=1, strict=True)
    outages: tuple[tuple[float, float], ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def applicable(self):
        if self.source == self.destination:
            raise ValueError("ISL requires distinct endpoints")
        rx = self.receiver
        if (
            rx.acquisition_sec
            or rx.ack_processing_sec
            or rx.setup_power_w
            or rx.tracking_power_w
            or rx.aggregate_rate_bps is not None
            or rx.aggregate_bandwidth_hz is not None
        ):
            raise ValueError(
                "ISL receiver cannot silently use ground-terminal policies"
            )
        if any(not math.isfinite(a + b) or not 0 <= a < b for a, b in self.outages):
            raise ValueError("ISL outage intervals must be finite and ordered")
        object.__setattr__(self, "outages", tuple(sorted(self.outages)))
        return self


class DTNSettings(StrictModel):
    model: Literal["reference-custody/1"] = "reference-custody/1"
    source: str = Field(min_length=1)
    parameter_status: Literal["synthetic", "published", "calibrated"]
    endpoints: dict[str, ReceiverResources] = Field(min_length=1, max_length=64)
    links: tuple[ISLLink, ...] = Field(min_length=1, max_length=512)
    sample_sec: float = Field(default=0.1, ge=0.01, le=5.0, strict=True)
    fragment_bits: int = Field(default=250000, ge=1, le=1000000000, strict=True)
    max_fragments: int = Field(default=10000, ge=1, le=100000, strict=True)
    max_retries: int = Field(default=3, ge=0, le=100, strict=True)
    retry_delay_sec: float = Field(default=0.1, ge=1e-6, le=3600, strict=True)
    ack_timeout_sec: float = Field(default=0.1, ge=1e-6, le=3600, strict=True)
    ack_processing_sec: float = Field(default=0.01, ge=1e-6, le=3600, strict=True)
    limb_altitude_m: float = Field(default=80000.0, ge=0, le=1e6, strict=True)
    seed: int = Field(default=0, ge=0, le=2**63 - 1, strict=True)
    max_transfers: int = Field(default=512, ge=1, le=10000, strict=True)
    max_events: int = Field(default=100000, ge=1, le=1000000, strict=True)

    @model_validator(mode="after")
    def unique_links(self):
        if len({(r.source, r.destination) for r in self.links}) != len(self.links):
            raise ValueError("duplicate directed ISL link")
        object.__setattr__(
            self,
            "links",
            tuple(sorted(self.links, key=lambda r: (r.source, r.destination))),
        )
        return self


class CustodyNetwork:
    def __init__(self, config, mission):
        self.config = DTNSettings.model_validate(config)
        if mission.radio_network is None:
            raise ValueError("DTN requires an explicit radio_network")
        if set(self.config.endpoints) != set(mission.actors):
            raise ValueError("DTN endpoints must cover every actor")
        if any(
            r.source not in mission.actors or r.destination not in mission.actors
            for r in self.config.links
        ):
            raise ValueError("unknown ISL endpoint")
        if any(
            b > mission.horizon_sec for r in self.config.links for _, b in r.outages
        ):
            raise ValueError("ISL outage exceeds mission horizon")
        if any(
            r.rx_beamwidth_deg is not None for r in self.config.links
        ) and not getattr(mission, "supports_body_fixed_solar", False):
            raise ValueError("receiver BODY pattern requires coupled spacecraft")
        self.transfers = {}
        self.bundle_expiry = {}
        self.expired_products = {}
        self.queue = []
        self.sequence = 0
        self.event_count = 0
        self.history = []
        self.intervals = []
        self.custody_history = []
        self.plans = {}
        self.tx_bits = dict.fromkeys(mission.actors, 0.0)
        self.rx_unique_bits = dict.fromkeys(mission.actors, 0.0)
        self.rx_duplicate_bits = dict.fromkeys(mission.actors, 0.0)
        self.rx_lost_bits = dict.fromkeys(mission.actors, 0.0)
        self.rx_discarded_bits = dict.fromkeys(mission.actors, 0.0)
        self.wasted_partial_bits = dict.fromkeys(mission.actors, 0.0)
        self.abandoned_inflight_bits = dict.fromkeys(mission.actors, 0.0)
        self.rx_energy_j = dict.fromkeys(mission.actors, 0.0)
        self.managed_blocked = set()

    def link(self, source, destination):
        return next(
            (
                r
                for r in self.config.links
                if (r.source, r.destination) == (source, destination)
            ),
            None,
        )

    def pinned(self, pid):
        return any(
            r["product_id"] == pid and r["status"] not in TERMINAL
            for r in self.transfers.values()
        )

    def reserved(self, actor):
        return sum(
            r["size_bits"]
            for r in self.transfers.values()
            if r["destination"] == actor and r["reserved"]
        )

    def receiving(self, actor):
        return any(
            r["destination"] == actor
            and (
                r["state"] == "inflight"
                or (r["state"] == "ready" and r["progress_bits"] > 0)
            )
            for r in self.transfers.values()
            if r["status"] not in TERMINAL
        )

    def log(self, kind, now, **data):
        if len(self.history) >= self.config.max_events:
            raise ValueError("DTN history budget exhausted")
        self.history.append({"time_sec": now, "kind": kind, **data})

    def schedule(self, time, kind, tid, attempt):
        if self.event_count >= self.config.max_events:
            raise ValueError("DTN event budget exhausted")
        self.sequence += 1
        self.event_count += 1
        heapq.heappush(self.queue, [time, self.sequence, kind, tid, attempt])

    def validate(self, c, mission):
        if c.command_id in self.transfers:
            raise ValueError("DTN transfer ID is immutable and cannot be reused")
        p = mission.products.get(c.product_id)
        if self.link(c.actor_id, c.destination_id) is None:
            raise ValueError("relay requires a configured directed ISL link")
        if (
            p is None
            or p.actor != c.actor_id
            or mission.remaining[p.id] != p.size_bits
            or p.id in mission.cancelled
        ):
            raise ValueError("relay requires an intact locally owned product")
        if not float(p.size_bits).is_integer() or p.size_bits > 2**53:
            raise ValueError(
                "DTN payload requires exact integer bit count within 2**53"
            )
        if (
            mission.processing_remaining[p.id]
            or mission.retained_for(p.id)
            or p.ready_sec > mission.clock
        ):
            raise ValueError("relay product must be ready, processed and unretained")
        if self.pinned(p.id):
            raise ValueError("product already has an unsettled custody transfer")
        if (
            math.ceil(p.size_bits / self.config.fragment_bits)
            > self.config.max_fragments
        ):
            raise ValueError("DTN fragment budget exceeded")
        if len(self.transfers) >= self.config.max_transfers:
            raise ValueError("DTN transfer budget exhausted")

    def admit(self, c, mission, expiry=None):
        self.validate(c, mission)
        size = mission.products[c.product_id].size_bits
        expires = mission.clock + c.relay.ttl_sec
        if expiry is not None:
            expires = min(expires, expiry)
        expires = min(expires, self.bundle_expiry.get(c.product_id, expires))
        self.bundle_expiry[c.product_id] = expires
        self.transfers[c.command_id] = {
            "id": c.command_id,
            "source": c.actor_id,
            "destination": c.destination_id,
            "product_id": c.product_id,
            "size_bits": size,
            "status": "queued",
            "state": "ready",
            "reason": None,
            "expires_sec": expires,
            "until_sec": mission.clock + c.duration_sec,
            "reserved": False,
            "fragment": 0,
            "offset_bits": 0.0,
            "progress_bits": 0.0,
            "retries": 0,
            "attempt": 0,
            "received": [],
            "received_bits": 0.0,
            "next_sec": mission.clock,
            "ready_sec": None,
            "arrival_sec": None,
            "rx_failed": False,
            "cancel_requested": False,
        }
        self.log(
            "submitted",
            mission.clock,
            transfer_id=c.command_id,
            product_id=c.product_id,
        )

    def finish(self, row, status, now, reason):
        if status != "completed":
            self.rx_discarded_bits[row["destination"]] += row["received_bits"]
        if row["state"] == "ready":
            self.wasted_partial_bits[row["source"]] += row["progress_bits"]
        elif row["state"] == "inflight":
            self.abandoned_inflight_bits[row["source"]] += self.fragment_size(row)
        row.update(
            status=status,
            state="terminal",
            reason=reason,
            reserved=False,
            arrival_sec=None,
        )
        self.log(
            status,
            now,
            transfer_id=row["id"],
            reason=reason,
            released_staging_bits=row["received_bits"],
            discarded_staging_bits=0.0
            if status == "completed"
            else row["received_bits"],
        )
        row["received_bits"] = 0.0

    def cancel(self, tid, now):
        row = self.transfers.get(tid)
        if row is None or row["status"] in TERMINAL:
            return
        row["cancel_requested"] = True
        if row["state"] in {"inflight", "await_ack"}:
            self.log("cancel_draining", now, transfer_id=tid)
        else:
            self.finish(row, "cancelled", now, "command_cancelled")

    def draw(self, row, label):
        key = (
            f"{self.config.seed}:{row['id']}:{row['fragment']}:{row['attempt']}:{label}"
        )
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64

    def faulted(self, mission, actor):
        bus = getattr(mission, "bus", None)
        return bus is not None and (
            bus.states[actor]["safe"]
            or bool({"radio", "bus"} & set(bus.states[actor]["faults"]))
        )

    def settle(self, mission):
        now = mission.clock
        for pid, expiry in self.bundle_expiry.items():
            if now < expiry or pid in self.expired_products or pid in mission.cancelled:
                continue
            ack = mission.acknowledged_time(pid)
            if ack is not None and ack < expiry:
                continue
            bits = mission.remaining[pid]
            self.expired_products[pid] = {
                "time_sec": expiry,
                "bits": bits,
                "actor": mission.products[pid].actor,
            }
            mission.discarded[pid] += bits
            mission.remaining[pid] = 0.0
            mission.cancelled[pid] = expiry
            mission.processing_remaining[pid] = 0.0
            self.log("bundle_expired", expiry, product_id=pid, bits=bits)
        for row in self.transfers.values():
            if row["status"] in TERMINAL:
                continue
            if now >= row["expires_sec"]:
                self.finish(row, "expired", now, "bundle_lifetime")
            elif now >= row["until_sec"] and not row["cancel_requested"]:
                self.cancel(row["id"], now)
        while self.queue and self.queue[0][0] <= now + 4 * math.ulp(now):
            time, _, kind, tid, attempt = heapq.heappop(self.queue)
            row = self.transfers[tid]
            if row["status"] in TERMINAL or attempt != row["attempt"]:
                continue
            link = self.link(row["source"], row["destination"])
            if kind == "arrival":
                row["state"] = "await_ack"
                row["arrival_sec"] = None
                failed = (
                    row["rx_failed"]
                    or any(a <= time < b for a, b in link.outages)
                    or self.faulted(mission, row["destination"])
                    or self.draw(row, "data") < link.loss_probability
                )
                if failed:
                    self.rx_lost_bits[row["destination"]] += self.fragment_size(row)
                    self.log(
                        "fragment_lost", time, transfer_id=tid, fragment=row["fragment"]
                    )
                else:
                    fragment = row["fragment"]
                    size = self.fragment_size(row)
                    if fragment in row["received"]:
                        self.rx_duplicate_bits[row["destination"]] += size
                        self.log(
                            "duplicate_fragment",
                            time,
                            transfer_id=tid,
                            fragment=fragment,
                        )
                    else:
                        row["received"].append(fragment)
                        row["received_bits"] += size
                        self.rx_unique_bits[row["destination"]] += size
                        self.log(
                            "fragment_received",
                            time,
                            transfer_id=tid,
                            fragment=fragment,
                            bits=size,
                        )
                    self.schedule(
                        time + row["propagation_sec"] + self.config.ack_processing_sec,
                        "ack",
                        tid,
                        attempt,
                    )
            elif kind == "ack":
                if (
                    self.draw(row, "ack") < link.ack_loss_probability
                    or self.faulted(mission, row["destination"])
                    or self.faulted(mission, row["source"])
                ):
                    self.log(
                        "ack_lost", time, transfer_id=tid, fragment=row["fragment"]
                    )
                    continue
                if row["cancel_requested"]:
                    self.finish(row, "cancelled", time, "command_cancelled")
                    continue
                row["offset_bits"] += self.fragment_size(row)
                row.update(
                    fragment=row["fragment"] + 1,
                    progress_bits=0.0,
                    retries=0,
                    state="ready",
                    ready_sec=None,
                    next_sec=time,
                )
                self.log(
                    "fragment_acknowledged",
                    time,
                    transfer_id=tid,
                    offset_bits=row["offset_bits"],
                )
                if row["offset_bits"] >= row["size_bits"]:
                    p = mission.products[row["product_id"]]
                    if (
                        p.actor != row["source"]
                        or mission.remaining[p.id] != p.size_bits
                    ):
                        raise ValueError(
                            "custody source inventory changed while pinned"
                        )
                    mission.products[p.id] = p.model_copy(
                        update={"actor": row["destination"]}
                    )
                    self.custody_history.append(
                        {
                            "product_id": p.id,
                            "source": row["source"],
                            "destination": row["destination"],
                            "bits": p.size_bits,
                            "time_sec": time,
                        }
                    )
                    self.finish(row, "completed", time, "custody_confirmed")
            elif kind == "timeout" and row["state"] in {"inflight", "await_ack"}:
                if row["cancel_requested"]:
                    self.finish(row, "cancelled", time, "command_cancelled")
                else:
                    self.retry(row, time, "ack_timeout")

    def retry(self, row, now, reason):
        if row["state"] == "ready":
            self.wasted_partial_bits[row["source"]] += row["progress_bits"]
            row["progress_bits"] = 0.0
        row["retries"] += 1
        if row["retries"] > self.config.max_retries:
            self.finish(row, "failed", now, "retry_budget")
            return
        row.update(
            state="ready",
            progress_bits=0.0,
            ready_sec=None,
            next_sec=now + self.config.retry_delay_sec,
            reason=reason,
        )
        self.log(
            "retry", now, transfer_id=row["id"], fragment=row["fragment"], reason=reason
        )

    def fragment_size(self, row):
        return min(self.config.fragment_bits, row["size_bits"] - row["offset_bits"])

    def budget(self, mission, row):
        link = self.link(row["source"], row["destination"])
        sample = (
            math.floor((mission.clock + 1e-9) / self.config.sample_sec)
            * self.config.sample_sec
        )
        left = mission.ephemerides[row["source"]].state(sample)
        right = mission.ephemerides[row["destination"]].state(sample)
        delta = right[:3] - left[:3]
        distance = float(np.linalg.norm(delta))
        if (
            distance < 1.0
            or distance > link.max_range_m
            or earth_occults(
                left[:3], right[:3], R_EARTH_EQUATOR + self.config.limb_altitude_m
            )
        ):
            return {"rate_bps": 0.0, "reason": "geometry", "range_m": distance}
        if any(a <= mission.clock < b for a, b in link.outages):
            return {"rate_bps": 0.0, "reason": "outage", "range_m": distance}
        tx = mission.radio_network.config.transmitters[row["source"]]
        direction = delta / distance
        tx_error = rx_loss = 0.0
        bus = getattr(mission, "bus", None)
        if tx.beamwidth_deg is not None or link.rx_beamwidth_deg is not None:
            from space_env.reference.spacecraft_bus import rotate_body

            if bus is None:
                raise ValueError("BODY ISL pattern requires coupled bus")
            if tx.beamwidth_deg is not None:
                bore = rotate_body(
                    bus.states[row["source"]]["quaternion_wxyz"], (0.0, 0.0, 1.0)
                )
                tx_error = math.degrees(
                    math.acos(float(np.clip(bore @ direction, -1, 1)))
                )
            if link.rx_beamwidth_deg is not None:
                bore = rotate_body(
                    bus.states[row["destination"]]["quaternion_wxyz"], (0.0, 0.0, 1.0)
                )
                error = math.degrees(
                    math.acos(float(np.clip(bore @ -direction, -1, 1)))
                )
                rx_loss = min(60.0, 12 * (error / link.rx_beamwidth_deg) ** 2)
        cap = min(
            mission.transmitter_rates[row["source"]],
            *(
                self.config.endpoints[a].memory_bandwidth_bps
                for a in (row["source"], row["destination"])
            ),
        )
        if bus is not None and getattr(mission, "dtn_lane_bandwidth", None):
            cap = min(
                cap,
                *(
                    mission.dtn_lane_bandwidth[a]
                    for a in (row["source"], row["destination"])
                ),
            )
        result = rf_budget(
            tx,
            link.receiver,
            distance,
            fade_db=rx_loss,
            pointing_error_deg=tx_error,
            rate_cap_bps=cap,
            doppler_hz=-tx.frequency_hz
            * float(direction @ (right[3:] - left[3:]))
            / C_M_S,
        )
        return {
            **result,
            "range_m": distance,
            "reason": "eligible" if result["rate_bps"] else "rf_margin",
        }

    def boundary(self, mission, right):
        now = mission.clock
        step = self.config.sample_sec
        right = min(right, (math.floor((now + 1e-9) / step) + 1) * step)
        times = [e[0] for e in self.queue] + list(self.bundle_expiry.values())
        for row in self.transfers.values():
            if row["status"] in TERMINAL:
                continue
            times.extend([row["expires_sec"], row["until_sec"], row["next_sec"]])
            link = self.link(row["source"], row["destination"])
            times.extend(t for interval in link.outages for t in interval)
            if row["ready_sec"] is not None:
                times.append(row["ready_sec"])
            if row["state"] == "ready":
                b = self.budget(mission, row)
                if row["ready_sec"] is None and link.acquisition_sec:
                    times.append(now + link.acquisition_sec)
                elif b["rate_bps"]:
                    times.append(
                        now
                        + (self.fragment_size(row) - row["progress_bits"])
                        / b["rate_bps"]
                    )
        return min([right, *[t for t in times if now + 1e-9 < t < right]])

    def allowed(self, mission, actor, power, floor, generation):
        v = mission.vehicles[actor]
        bus = getattr(mission, "bus", None)
        if self.faulted(mission, actor):
            return False
        if bus is not None:
            if (v.base_power_w + power) * bus.models[
                actor
            ].power.discharge_efficiency > bus.models[actor].power.max_load_w + 1e-9:
                return False
            floor = max(floor, bus.network.heater_floor(actor))
        threshold = floor * v.battery_wh
        return mission.energy_wh[actor] >= threshold - 1e-12 and not (
            mission.energy_wh[actor] <= threshold + 1e-12
            and generation[actor] < v.base_power_w + power
        )

    def rx_power(self, mission, actor):
        watts = self.config.endpoints[actor].power_w
        bus = getattr(mission, "bus", None)
        return (
            watts
            if bus is None
            else watts / bus.models[actor].power.discharge_efficiency
        )

    def prepare(self, mission, controls, chosen, processors, generation, blocked):
        self.settle(mission)
        now = mission.clock
        active = {}
        self.plans = {}
        busy = set(chosen) | set(processors) | set(blocked) | set(self.managed_blocked)
        # A receiver listens through the propagation tail. It cannot be reused
        # before packet arrival, even though transmitter serialization has ended.
        for row in self.transfers.values():
            if row["status"] not in TERMINAL and row["state"] == "inflight":
                a = row["destination"]
                if any(
                    lo <= now < hi for lo, hi in self.link(row["source"], a).outages
                ):
                    row["rx_failed"] = True
                power = self.rx_power(mission, a)
                busy.add(a)
                if self.allowed(
                    mission, a, power, self.config.endpoints[a].min_soc, generation
                ):
                    active[a] = {
                        "role": "rx",
                        "power_w": power,
                        "floor": self.config.endpoints[a].min_soc,
                        "transfer_id": row["id"],
                    }
                else:
                    row["rx_failed"] = True
        for tid in sorted(
            controls,
            key=lambda t: (
                self.transfers[t]["progress_bits"] <= 0,
                self.transfers[t]["ready_sec"] is None,
                t,
            ),
        ):
            row = self.transfers.get(tid)
            if (
                row is None
                or row["status"] in TERMINAL
                or row["state"] != "ready"
                or row["cancel_requested"]
                or row["next_sec"] > now + 1e-9
            ):
                continue
            src, dst = row["source"], row["destination"]
            link = self.link(src, dst)
            if src in busy or dst in busy:
                if row["progress_bits"]:
                    self.retry(row, now, "endpoint_interrupted")
                row["ready_sec"] = None
                row["reason"] = "endpoint_busy"
                continue
            b = self.budget(mission, row)
            if not b["rate_bps"]:
                if row["progress_bits"]:
                    self.retry(row, now, "link_interrupted")
                row["ready_sec"] = None
                row["reason"] = b["reason"]
                continue
            if not row["reserved"]:
                occupied = (
                    sum(
                        mission.remaining[p.id]
                        for p in mission.products.values()
                        if p.actor == dst
                    )
                    + self.reserved(dst)
                    + getattr(mission, "dtn_managed_memory", {}).get(dst, 0.0)
                )
                if occupied + row["size_bits"] > mission.vehicles[dst].memory_bits:
                    row["reason"] = "receiver_memory"
                    continue
            tx = mission.vehicles[src].tx_power_w
            rx = self.rx_power(mission, dst)
            if (
                not mission.transmitter_ready(src)
                or not self.allowed(
                    mission, src, tx, mission.vehicles[src].min_tx_soc, generation
                )
                or not self.allowed(
                    mission, dst, rx, self.config.endpoints[dst].min_soc, generation
                )
            ):
                if row["progress_bits"]:
                    self.retry(row, now, "power_interrupted")
                row["ready_sec"] = None
                row["reason"] = "endpoint_power_or_health"
                continue
            row["reserved"] = True
            row["status"] = "running"
            row["reason"] = None
            if row["ready_sec"] is None:
                row["ready_sec"] = now + link.acquisition_sec
                self.log("acquiring", now, transfer_id=tid, ready_sec=row["ready_sec"])
            payload = row["ready_sec"] <= now + 1e-9
            active[src] = {
                "role": "tx",
                "power_w": tx,
                "floor": mission.vehicles[src].min_tx_soc,
                "transfer_id": tid,
            }
            active[dst] = {
                "role": "rx",
                "power_w": rx,
                "floor": self.config.endpoints[dst].min_soc,
                "transfer_id": tid,
            }
            self.plans[tid] = {
                "rate_bps": b["rate_bps"] if payload else 0.0,
                "range_m": b["range_m"],
            }
            busy.update((src, dst))
        return active

    def integrate(self, mission, start, span, active):
        if not active and not any(
            r["status"] not in TERMINAL for r in self.transfers.values()
        ):
            return
        if len(self.intervals) >= self.config.max_events:
            raise ValueError("DTN interval budget exhausted")
        self.intervals.append(
            {
                "start_sec": start,
                "end_sec": start + span,
                "endpoints": copy.deepcopy(active),
                "transmissions": copy.deepcopy(self.plans),
            }
        )
        end = start + span
        for a, item in active.items():
            if item["role"] == "rx":
                self.rx_energy_j[a] += self.config.endpoints[a].power_w * span
        for tid, plan in self.plans.items():
            row = self.transfers[tid]
            if not plan["rate_bps"]:
                continue
            needed = self.fragment_size(row) - row["progress_bits"]
            duration = needed / plan["rate_bps"]
            complete = span >= duration or abs(span - duration) <= 4 * max(
                math.ulp(start), math.ulp(end), math.ulp(duration)
            )
            bits = needed if complete else plan["rate_bps"] * span
            row["progress_bits"] += bits
            self.tx_bits[row["source"]] += bits
            if complete:
                row["attempt"] += 1
                row["rx_failed"] = False
                row["state"] = "inflight"
                row["propagation_sec"] = plan["range_m"] / C_M_S
                arrival = end + row["propagation_sec"]
                row["arrival_sec"] = arrival
                self.schedule(arrival, "arrival", tid, row["attempt"])
                timeout = (
                    arrival
                    + row["propagation_sec"]
                    + self.config.ack_processing_sec
                    + self.config.ack_timeout_sec
                )
                self.schedule(timeout, "timeout", tid, row["attempt"])
                self.log(
                    "serialized",
                    end,
                    transfer_id=tid,
                    fragment=row["fragment"],
                    attempt=row["attempt"],
                    bits=self.fragment_size(row),
                    arrival_sec=arrival,
                )

    def sync_artifacts(self, mission, artifacts):
        for pid, row in artifacts.items():
            owner = mission.products[pid].actor
            if row["actor"] != owner:
                row.setdefault("origin_actor", row["actor"])
                row["actor"] = owner

    def custody(self, actor):
        incoming = sum(
            r["bits"] for r in self.custody_history if r["destination"] == actor
        )
        outgoing = sum(r["bits"] for r in self.custody_history if r["source"] == actor)
        return incoming, outgoing

    def observation(self, actor):
        return {
            "model": self.config.model,
            "reserved_bits": self.reserved(actor),
            "transfers": [
                copy.deepcopy(r)
                for r in self.transfers.values()
                if actor in (r["source"], r["destination"])
            ],
        }

    def report(self, mission):
        staging = {
            a: sum(
                r["received_bits"]
                for r in self.transfers.values()
                if r["destination"] == a
            )
            for a in mission.actors
        }
        rx_balance = {
            a: self.rx_unique_bits[a]
            - self.custody(a)[0]
            - self.rx_discarded_bits[a]
            - staging[a]
            for a in mission.actors
        }
        inflight = sum(
            self.fragment_size(r)
            for r in self.transfers.values()
            if r["status"] not in TERMINAL and r["state"] == "inflight"
        )
        partial = sum(
            r["progress_bits"]
            for r in self.transfers.values()
            if r["status"] not in TERMINAL and r["state"] == "ready"
        )
        wire_balance = math.fsum(self.tx_bits.values()) - math.fsum(
            [
                *self.rx_unique_bits.values(),
                *self.rx_duplicate_bits.values(),
                *self.rx_lost_bits.values(),
                *self.wasted_partial_bits.values(),
                *self.abandoned_inflight_bits.values(),
                inflight,
                partial,
            ]
        )
        return {
            "schema": "space-env.reference-custody/1",
            "transfers": copy.deepcopy(self.transfers),
            "bundle_expiry_sec": dict(self.bundle_expiry),
            "expired_products": copy.deepcopy(self.expired_products),
            "pending_events": copy.deepcopy(sorted(self.queue)),
            "history": copy.deepcopy(self.history),
            "intervals": copy.deepcopy(self.intervals),
            "custody_history": copy.deepcopy(self.custody_history),
            "tx_serialized_bits": dict(self.tx_bits),
            "rx_unique_bits": dict(self.rx_unique_bits),
            "rx_duplicate_bits": dict(self.rx_duplicate_bits),
            "rx_discarded_staging_bits": dict(self.rx_discarded_bits),
            "rx_lost_fragment_bits": dict(self.rx_lost_bits),
            "wasted_partial_bits": dict(self.wasted_partial_bits),
            "abandoned_inflight_bits": dict(self.abandoned_inflight_bits),
            "receiver_balance_error_bits": rx_balance,
            "wire_balance_error_bits": wire_balance,
            "inflight_payload_bits": inflight,
            "partial_serialization_bits": partial,
            "rx_dc_energy_j": dict(self.rx_energy_j),
            "reserved_bits": {a: self.reserved(a) for a in mission.actors},
            "scope": "whole-product store-and-forward; retained source copy, stop-and-wait fragments; ideal out-of-band lossy ACK, not BPv7/BIBE/LTP or wire protocol",
        }
