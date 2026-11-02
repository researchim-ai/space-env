"""Deterministic bounded resource lanes over the existing coupled executor.

One managed CPU, payload and radio lane per actor. No native in-place CPU,
multicore scheduling, RF link-budget improvement or optimal allocation claim.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel


class ActorLanes(StrictModel):
    cpu: bool = Field(default=True, strict=True)
    payload: bool = Field(default=True, strict=True)
    radio: bool = Field(default=True, strict=True)
    memory_bandwidth_bps: float = Field(gt=0, le=1e15, strict=True)
    priority: tuple[Literal["payload", "cpu", "radio"], ...] = (
        "payload",
        "cpu",
        "radio",
    )

    @model_validator(mode="after")
    def permutation(self):
        if len(self.priority) != 3 or set(self.priority) != {"payload", "cpu", "radio"}:
            raise ValueError("lane priority must be a permutation of payload/cpu/radio")
        return self


class ParallelExecution(StrictModel):
    mode: Literal["resource-lanes/1"] = "resource-lanes/1"
    actors: dict[str, ActorLanes] = Field(min_length=1, max_length=64)


def command_claims(command, config, bus_models):
    kind, actor = command.kind, command.actor_id
    lane = {"transform": "cpu", "imaging": "payload", "downlink": "radio"}.get(kind)
    if lane is not None and not getattr(config.actors[actor], lane):
        raise ValueError(f"actor {lane} lane is disabled")
    if kind == "relay":
        return ("cpu", "payload", "radio", "attitude", "propulsion")
    if kind == "transform":
        return ("cpu",)
    if kind == "imaging":
        return ("payload", "attitude")
    if kind == "downlink":
        return (
            ("radio", "attitude")
            if bus_models[actor].attitude.radio_tolerance_deg < 180
            else ("radio",)
        )
    if kind in {"body_burn", "momentum_unload"}:
        return ("propulsion", "attitude")
    if kind in {"slew", "wheel_torque"}:
        return ("attitude",)
    if kind == "wait":
        return ("cpu", "payload", "radio", "attitude", "propulsion")
    raise ValueError("command is unsupported by parallel resource lanes")


class ResourceLanes:
    def __init__(self, config):
        self.config = config
        self.intervals = []
        self.blocked = []

    def _pause(self, lifecycle, op, reason, backend):
        op["resource_pause_reason"] = reason
        if op["kind"] == "transform":
            lifecycle.job_state[op["job_id"]].update(status="paused", reason=reason)
        elif op["progress_sec"] > 0:
            lifecycle.fail_acquisition(op, backend, "resource_preempted:" + reason)

    @staticmethod
    def job_io(lifecycle, job):
        sizes = {r.raw_product_id: r.raw_size_bits for r in lifecycle.requests.values()}
        sizes.update(
            {p.id: p.size_bits for row in lifecycle.jobs.values() for p in row.outputs}
        )
        return (
            sum(sizes[p] for p in job.inputs) + sum(p.size_bits for p in job.outputs)
        ) / job.work_sec

    def allocate(self, backend, operations, ordinary, right):
        life, mission, bus = backend.lifecycle, backend.mission, backend.bus
        now = backend.now()
        radio = {
            row["command"]["actor_id"]: row
            for row in ordinary
            if row["command"]["kind"] == "downlink"
        }
        actions = {
            actor: row["command"]["destination_id"] for actor, row in radio.items()
        }
        products = {actor: row["command"]["product_id"] for actor, row in radio.items()}
        # Use the same native station/booking/SOC selector, in stored-battery units.
        vehicles = {
            a: v.model_copy(
                update={
                    "base_power_w": bus.battery_load(
                        a, bus.base_power(a, v.base_power_w) + bus.extra_power(a) + life.actuation_power(a)
                    ),
                    "tx_power_w": bus.battery_load(a, v.tx_power_w),
                    "min_tx_soc": max(v.min_tx_soc, life.thermal_heater_floor(a)),
                }
            )
            for a, v in mission.vehicles.items()
        }
        if mission.radio_network is not None:
            candidates = mission.select_transmissions(actions, products, generation=bus.generation_w, vehicles=vehicles, ignore_radio=True)
            mission.radio_network.prepare(mission, candidates)
            right = mission.radio_network.boundary(now, right)
        selected = mission.select_transmissions(
            actions, products, generation=bus.generation_w, vehicles=vehicles
        )
        by_actor = {a: {} for a in backend.actors}
        for op in operations:
            by_actor[op["actor"]]["payload" if op["kind"] == "imaging" else "cpu"] = op
        for a in selected:
            by_actor[a]["radio"] = radio[a]
        grants, rates, radio_rates, loads = [], {}, {}, {}
        denied = []
        for a in backend.actors:
            config, vehicle = self.config.actors[a], mission.vehicles[a]
            base = bus.base_power(a, vehicle.base_power_w) + bus.extra_power(a) + life.actuation_power(a)
            average, peak, used_bw = 0.0, 0.0, 0.0
            floor = (
                bus.models[a].attitude.slew_min_soc if bus.extra_power(a) > 0 else 0.0
            )
            floor = max(floor, life.actuation_floor(a))
            for lane in config.priority:
                item = by_actor[a].get(lane)
                if item is None:
                    continue
                speed = 1.0
                bandwidth_left = max(0.0, config.memory_bandwidth_bps - used_bw)
                if lane == "payload":
                    request = life.requests[item["request_id"]]
                    demand = request.raw_size_bits / request.acquisition_sec
                    watts = life.imaging_power(a)
                    peak_w = (
                        life.spec.payloads[a].power_w
                        if life.quality is None
                        else life.quality.admission_power(
                            a, life.spec.payloads[a].power_w
                        )
                    )
                    reserve = life.spec.payloads[a].min_soc
                elif lane == "cpu":
                    job = life.jobs[item["job_id"]]
                    demand = (
                        0.0
                        if item["switch_remaining_sec"] > 0
                        else self.job_io(life, job)
                    )
                    speed = min(1.0, bandwidth_left / demand) if demand > 0 else 1.0
                    watts, peak_w, reserve = job.power_w, job.power_w, job.min_soc
                else:
                    station, _ = selected[a]
                    demand = min(mission.effective_rate(a, station), bandwidth_left)
                    watts, peak_w, reserve = (
                        vehicle.tx_power_w,
                        vehicle.tx_power_w,
                        vehicle.min_tx_soc,
                    )
                reason = None
                if (
                    (lane == "payload" and demand > bandwidth_left)
                    or (lane == "cpu" and speed <= 0)
                    or (lane == "radio" and demand <= 0)
                ):
                    reason = "memory_bandwidth"
                elif base + peak + peak_w > bus.models[a].power.max_load_w:
                    reason = "shared_peak_power"
                else:
                    next_floor = max(floor, reserve)
                    net = bus.generation_w[a] - bus.battery_load(
                        a, base + average + watts
                    )
                    seconds = (
                        (mission.energy_wh[a] - next_floor * vehicle.battery_wh)
                        * 3600
                        / -net
                        if net < 0
                        else math.inf
                    )
                    if (
                        mission.energy_wh[a] < next_floor * vehicle.battery_wh - 1e-12
                        or seconds <= 1e-9
                    ):
                        reason = "shared_soc"
                cid = (
                    item["command"]["command_id"]
                    if lane == "radio"
                    else next(key for key, op in life.operations.items() if op is item)
                )
                if reason is not None:
                    denied.append(
                        {"actor": a, "command_id": cid, "lane": lane, "reason": reason}
                    )
                    if lane != "radio":
                        self._pause(life, item, reason, backend)
                    continue
                average += watts
                peak += peak_w
                floor = max(floor, reserve)
                used_bw += demand * speed
                if lane == "radio":
                    radio_rates[a] = demand
                    product = selected[a][1]
                    right = min(right, now + mission.remaining[product.id] / demand)
                    if vehicle.max_tx_sec is not None:
                        right = min(
                            right, now + vehicle.max_tx_sec - mission.tx_seconds[a]
                        )
                else:
                    item["resource_pause_reason"] = None
                    grants.append(item)
                    rates[cid] = speed
                    if lane == "cpu" and item["switch_remaining_sec"] <= 0:
                        right = min(
                            right,
                            now
                            + life.job_state[item["job_id"]]["remaining_sec"] / speed,
                        )
            net = bus.generation_w[a] - bus.battery_load(a, base + average)
            if net < 0 and average > 0:
                right = min(
                    right,
                    now
                    + (mission.energy_wh[a] - floor * vehicle.battery_wh) * 3600 / -net,
                )
            loads[a] = {
                "reserved_peak_w": base + peak,
                "reserved_memory_bps": used_bw,
                "memory_bandwidth_bps": config.memory_bandwidth_bps,
            }
        ordinary = [
            row
            for row in ordinary
            if row["command"]["kind"] != "downlink"
            or row["command"]["actor_id"] in radio_rates
            or (mission.radio_network is not None and mission.radio_network.links.get(row["command"]["actor_id"], {}).get("state") == "acquiring")
        ]
        # Native boundaries include geometry/outages/bookings and rate transitions.
        right = min(
            [right, *[t for t in mission._boundaries if now + 1e-9 < t < right]]
        )
        return grants, ordinary, rates, radio_rates, right, loads, denied

    def record(
        self, backend, now, right, valid, powers, work_rates, loads, denied, before
    ):
        mission = backend.mission
        actors = {}
        for a in backend.actors:
            bytes_sent = sum(
                row["bits"]
                for row in mission.receipts[before["receipt_index"] :]
                if row["actor"] == a
            )
            work = []
            io_bits = bytes_sent
            for op in valid:
                if op["actor"] != a:
                    continue
                cid = next(
                    key
                    for key, value in backend.lifecycle.operations.items()
                    if value is op
                )
                factor = work_rates.get(cid, 1.0)
                if op["kind"] == "imaging":
                    io_bits += (
                        backend.lifecycle.requests[op["request_id"]].raw_size_bits
                        / backend.lifecycle.requests[op["request_id"]].acquisition_sec
                        * (right - now)
                    )
                elif op["switch_remaining_sec"] <= 0:
                    io_bits += (
                        self.job_io(
                            backend.lifecycle, backend.lifecycle.jobs[op["job_id"]]
                        )
                        * factor
                        * (right - now)
                    )
                work.append(
                    {
                        "command_id": cid,
                        "kind": op["kind"],
                        "work_rate": factor,
                        "phase": "switching"
                        if op["kind"] == "transform" and op["switch_remaining_sec"] > 0
                        else "work",
                    }
                )
            consumed = mission.consumed_wh[a] - before["consumed"][a]
            actors[a] = {
                **loads[a],
                "managed": work,
                "radio_bits": bytes_sent,
                "memory_io_bits": io_bits,
                "useful_load_wh": consumed
                * backend.bus.models[a].power.discharge_efficiency,
                "managed_energy_wh": powers.get(a, 0.0) * (right - now) / 3600,
                "inventory_bits": backend.lifecycle.occupied(backend, a),
                "reserved_bits": backend.lifecycle.reservations(a),
                "memory_capacity_bits": mission.vehicles[a].memory_bits,
            }
        self.intervals.append(
            {
                "start_sec": now,
                "end_sec": right,
                "actors": actors,
                "blocked": copy.deepcopy(denied),
            }
        )
        self.blocked.extend({**row, "time_sec": now} for row in denied)

    def observation(self, actor):
        latest = self.intervals[-1] if self.intervals else None
        return {
            "configuration": self.config.actors[actor].model_dump(mode="json"),
            "last_interval": None
            if latest is None
            else {
                "start_sec": latest["start_sec"],
                "end_sec": latest["end_sec"],
                **copy.deepcopy(latest["actors"][actor]),
                "blocked": [
                    copy.deepcopy(row)
                    for row in latest["blocked"]
                    if row["actor"] == actor
                ],
            },
        }

    def report(self):
        return {
            "schema": "space-env.resource-lanes/1",
            "configuration": self.config.model_dump(mode="json"),
            "intervals": copy.deepcopy(self.intervals),
            "blocked": copy.deepcopy(self.blocked),
            "scope": "one managed CPU/payload/radio lane, bounded grid arbitration; no native in-place CPU or optimal scheduler",
        }
