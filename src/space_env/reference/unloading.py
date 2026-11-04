"""Paid wheel/RCS momentum unloading, ideal balanced couples and shared tank."""

from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.propulsion import G0_M_S2
from space_env.reference.rigid_bus import error_vector
from space_env.reference.thrust import BalancedCoupleThrust
from space_env.runtime.schema import FiniteScalar, StrictModel


class RcsSettings(StrictModel):
    parameter_status: Literal["synthetic", "published", "calibrated"]
    propellant_pool: Literal["shared-lumped"]
    max_total_nozzle_thrust_n: float = Field(gt=0, le=1e6, strict=True)
    model_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    max_torque_nm: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    lever_arm_m: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    isp_s: float = Field(gt=0, le=10000, strict=True)
    startup_sec: float = Field(default=0.2, ge=0, le=100, strict=True)
    startup_power_w: float = Field(default=3.0, gt=0, le=1e6, strict=True)
    firing_power_w: float = Field(default=4.0, gt=0, le=1e6, strict=True)
    electrical_heat_fraction: float = Field(default=1.0, ge=0, le=1, strict=True)
    wall_heat_w_per_n: float = Field(default=0.0, ge=0, le=1e6, strict=True)
    min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)
    max_mass_loss_fraction: float = Field(default=0.001, gt=0, le=0.05, strict=True)

    @model_validator(mode="after")
    def physical(self):
        if not self.model_id.strip() or not self.source.strip():
            raise ValueError("RCS requires model ID and provenance")
        if min(self.max_torque_nm) <= 0 or max(self.max_torque_nm) > 1000:
            raise ValueError("RCS torque limits must be in (0,1000] Nm")
        if min(self.lever_arm_m) < 0.001 or max(self.lever_arm_m) > 100:
            raise ValueError("effective RCS pair arms must be in [0.001,100] m")
        return self


class UnloadingSettings(StrictModel):
    actors: dict[str, RcsSettings] = Field(min_length=1, max_length=64)
    max_segments: int = Field(default=256, ge=1, le=10000, strict=True)


class MomentumUnloader:
    def __init__(self, config, backend):
        self.config = config
        for a, c in config.actors.items():
            if (
                a not in backend.actors
                or backend.config.spacecraft_bus[a].attitude.rigid is None
            ):
                raise ValueError("RCS unloading requires a known rigid actor")
            rigid = backend.config.spacecraft_bus[a].attitude.rigid
            if (
                rigid.max_applicable_rate_rad_s
                * min(rigid.control_step_sec, rigid.model.max_step_sec)
                > 0.25
            ):
                raise ValueError(
                    "RCS unloading angular sampling exceeds 0.25 rad applicability"
                )
            if not math.isfinite(
                (
                    c.startup_power_w
                    + c.firing_power_w
                    + c.wall_heat_w_per_n
                    * sum(np.asarray(c.max_torque_nm) / c.lever_arm_m)
                )
                * backend.horizon()
            ):
                raise ValueError("RCS energy outside finite applicability")
        self.initial_mass = {
            a: backend.mission.ephemerides[a].mass(0.0) for a in config.actors
        }
        self.fuel_used = dict.fromkeys(config.actors, 0.0)
        self.operations, self.plans, self.pending = {}, {}, {}
        self.intervals = []
        self.control_latches = {}
        self.control_samples = dict.fromkeys(config.actors, 0)
        self.fixed_periods = {
            a: backend.config.spacecraft_bus[a].attitude.rigid.control_step_sec
            for a in config.actors
            if backend.config.spacecraft_bus[a].attitude.rigid.clock_mode == "fixed-tick/1"
        }

    def validate(self, c, backend):
        if c.actor_id not in self.config.actors:
            raise ValueError("actor has no declared unloading RCS")
        limit = backend.bus.rigid.models[c.actor_id].model.max_momentum_nms
        if np.any(np.abs(c.momentum_unload.target_momentum_nms) > np.asarray(limit)):
            raise ValueError("target momentum exceeds wheel capacity")
        if c.momentum_unload.tolerance_nms >= min(limit):
            raise ValueError("momentum tolerance must be smaller than wheel capacity")

    def admit(self, c, backend):
        self.validate(c, backend)
        s = backend.bus.rigid.states[c.actor_id]
        self.operations[c.command_id] = {
            "actor": c.actor_id,
            "status": "accepted",
            "reason": None,
            "phase": "startup",
            "startup_remaining_sec": self.config.actors[c.actor_id].startup_sec,
            "target_momentum_nms": list(c.momentum_unload.target_momentum_nms),
            "tolerance_nms": c.momentum_unload.tolerance_nms,
            "hold_quaternion_wxyz": list(s.quaternion_wxyz),
            "initial_momentum_nms": list(s.wheel_momentum_nms),
            "until_sec": backend.now() + c.duration_sec,
            "settle_remaining_sec": backend.bus.models[
                c.actor_id
            ].attitude.settling_sec,
            "dc_energy_j": 0.0,
            "rcs_heat_j": 0.0,
            "wall_heat_j": 0.0,
            "electrical_nonthermal_j": 0.0,
            "consumed_propellant_kg": 0.0,
            "startup_sec": 0.0,
            "control_sec": 0.0,
        }

    def cancel(self, cid):
        op = self.operations.get(cid)
        if op is None or op["status"] in {"completed", "failed", "cancelled"}:
            return False
        op.update(status="cancelled", reason="operator_cancelled")
        self.control_latches.pop(op["actor"], None)
        return True

    def stop(self, a, backend, reason):
        self.control_latches.pop(a, None)
        plan = self.plans.pop(a, None)
        if plan is not None:
            self.operations[plan["cid"]].update(status="failed", reason=reason)
        backend.bus.rigid.disable(a, reason)
        backend.bus.rigid.set_external_torque(a, [0.0, 0.0, 0.0])

    def prepare(self, backend, active):
        self.plans, self.pending = {}, {}
        live = {row["command"]["actor_id"] for row in active
                if row["command"]["kind"] == "momentum_unload"
                and self.operations[row["command"]["command_id"]]["status"]
                not in {"completed", "failed", "cancelled"}}
        self.control_latches = {a: p for a, p in self.control_latches.items() if a in live}
        total = sum(len(p.maneuvers) for p in backend.mission.ephemerides.values())
        main = backend.lifecycle.actuation
        reserved = (
            sum(p["phase"] == "firing" for p in main.plans.values())
            if main is not None
            else 0
        )
        slots = min(
            self.config.max_segments - len(self.intervals),
            backend.config.max_maneuvers - total - reserved,
        )
        for row in sorted(
            active,
            key=lambda r: (
                backend.actors.index(r["command"]["actor_id"]),
                r["command"]["command_id"],
            ),
        ):
            c = row["command"]
            if c["kind"] != "momentum_unload":
                continue
            op = self.operations[c["command_id"]]
            if op["status"] in {"completed", "failed", "cancelled"}:
                continue
            a = op["actor"]
            cfg = self.config.actors[a]
            bus = backend.bus
            rigid = bus.rigid
            s = rigid.states[a]
            r = rigid.models[a]
            vehicle = backend.mission.vehicles[a]
            phase = "startup" if op["startup_remaining_sec"] > 1e-9 else "control"
            u, external = np.zeros(3), np.zeros(3)
            period = self.fixed_periods.get(a)
            key = (c["command_id"], math.floor((backend.now()+1e-9)/r.control_step_sec),
                   bus.states[a]["rate_scale"])
            latch = self.control_latches.get(a)
            held = period is not None and latch is not None and latch["key"] == key
            if phase == "startup":
                self.control_latches.pop(a, None)
            if phase == "control":
                if held:
                    u, external = latch["motor"].copy(), latch["external"].copy()
                else:
                    w = np.asarray(s.omega_body_rad_s)
                    h = np.asarray(s.wheel_momentum_nms)
                    error = error_vector(s.quaternion_wxyz, op["hold_quaternion_wxyz"])
                    body = (
                        r.proportional_nm_per_rad * error
                        - r.damping_nm_per_rad_s * w
                        + np.cross(w, np.asarray(r.model.inertia_kg_m2) @ w + h)
                    )
                    cap = np.asarray(cfg.max_torque_nm) * bus.states[a]["rate_scale"]
                    body = np.clip(body, -cap, cap)
                    available = np.maximum(0.0, cap - np.abs(body))
                    wheel = np.minimum(
                        available,
                        np.asarray(r.model.max_torque_nm) * bus.states[a]["rate_scale"],
                    )
                    u = np.clip(
                        (np.asarray(op["target_momentum_nms"]) - h) / r.control_step_sec,
                        -wheel,
                        wheel,
                    )
                    external = body + u
            scalar = float(np.sum(np.abs(external) / cfg.lever_arm_m))
            if scalar > cfg.max_total_nozzle_thrust_n:
                scale = cfg.max_total_nozzle_thrust_n / scalar
                u *= scale
                external *= scale
                scalar = cfg.max_total_nozzle_thrust_n
            if phase == "control" and period is not None and not held:
                self.control_samples[a] += 1
                self.control_latches[a] = {"key":key,"motor":u.copy(),"external":external.copy(),
                                           "sample_sec":backend.now()}
            power = (
                cfg.startup_power_w
                if phase == "startup"
                else cfg.firing_power_w
                if scalar > 1e-15
                else 0.0
            )
            heat = power * cfg.electrical_heat_fraction + scalar * cfg.wall_heat_w_per_n
            wheel_power = r.supply_w if phase == "control" else 0.0
            reason = bus.allowed(
                a, "body_burn", c["command_id"], backend, extra_w=power + wheel_power
            )
            floor = max(cfg.min_soc, bus.models[a].attitude.slew_min_soc)
            if (
                reason is None
                and backend.mission.energy_wh[a] <= floor * vehicle.battery_wh + 1e-12
            ):
                reason = "unloading_soc"
            remaining = (
                op["startup_remaining_sec"]
                if phase == "startup"
                else r.control_step_sec
            )
            if scalar > 1e-15:
                provider = backend.mission.ephemerides[a]
                mass = provider.mass(backend.now())
                mass_floor = max(
                    provider.dry_mass_kg,
                    self.initial_mass[a] * (1 - cfg.max_mass_loss_fraction),
                )
                seconds = (mass - mass_floor) * cfg.isp_s * G0_M_S2 / scalar
                if seconds <= 1e-9:
                    reason = (
                        "propellant_exhausted"
                        if mass <= provider.dry_mass_kg + 1e-12
                        else "constant_inertia_mass_limit"
                    )
                elif slots <= 0:
                    reason = "unloading_segment_budget"
                else:
                    remaining = min(remaining, seconds)
                    slots -= 1
            op.update(status="running", phase=phase)
            self.plans[a] = {
                "cid": c["command_id"],
                "phase": phase,
                "power_w": power,
                "heat_w": heat,
                "wall_w": scalar * cfg.wall_heat_w_per_n,
                "nonthermal_w": power * (1 - cfg.electrical_heat_fraction),
                "remaining_sec": remaining,
                "scalar_nozzle_thrust_n": scalar,
                "motor_torque_nm": u,
                "torque_body_nm": external,
                "floor": floor,
                "qualified_at_start": (
                    np.max(
                        np.abs(
                            np.asarray(s.wheel_momentum_nms) - op["target_momentum_nms"]
                        )
                    )
                    <= op["tolerance_nms"]
                    and np.linalg.norm(s.omega_body_rad_s) <= r.rate_tolerance_rad_s
                    and np.linalg.norm(
                        error_vector(s.quaternion_wxyz, op["hold_quaternion_wxyz"])
                    )
                    <= math.radians(bus.models[a].attitude.rigid_pointing_tolerance_deg)
                ),
            }
            if reason:
                self.stop(a, backend, reason)
            else:
                self.sync(a, backend)

    def sync(self, a, backend):
        p = self.plans[a]
        rigid = backend.bus.rigid
        rigid.plans[a].update(
            enabled=p["phase"] == "control", torque=p["motor_torque_nm"], reason=None
        )
        rigid.set_external_torque(a, p["torque_body_nm"])

    def boundary(self, backend, right):
        for a, p in list(self.plans.items()):
            self.sync(a, backend)
            v = backend.mission.vehicles[a]
            bus = backend.bus
            power = (
                bus.base_power(a, v.base_power_w) + bus.extra_power(a) + p["power_w"]
            )
            net = bus.generation_w[a] - bus.battery_load(a, power)
            remaining = p["remaining_sec"]
            if net < 0:
                remaining = min(
                    remaining,
                    (backend.mission.energy_wh[a] - p["floor"] * v.battery_wh)
                    * 3600
                    / -net,
                )
            if remaining <= 1e-9:
                self.stop(a, backend, "unloading_soc")
            else:
                right = min(right, backend.now() + remaining)
        return right

    def power(self, a):
        return self.plans.get(a, {}).get("power_w", 0.0)

    def floor(self, a):
        return self.plans.get(a, {}).get("floor", 0.0)

    def heat_adjustment(self, a, fraction):
        p = self.plans.get(a)
        return 0.0 if p is None else p["heat_w"] - fraction * p["power_w"]

    def apply(self, backend, span):
        for a, p in self.plans.items():
            if p["scalar_nozzle_thrust_n"] <= 1e-15:
                continue
            provider = backend.mission.ephemerides[a]
            cfg = self.config.actors[a]
            before = provider.mass(backend.now())
            law = BalancedCoupleThrust(
                p["scalar_nozzle_thrust_n"],
                cfg.isp_s,
                provider.dry_mass_kg,
                (0.0, 0.0, 0.0),
                tuple(p["torque_body_nm"]),
                cfg.lever_arm_m,
            )
            geometry_changed = provider.burn(backend.now(), span, law)
            self.pending[a] = {
                "mass_before": before,
                "mass_after": provider.mass(backend.now() + span),
            }
            if geometry_changed:
                backend.geometry_revision += 1
                backend._refresh_geometry(a)

    def finish(self, backend, start, span):
        for a, p in self.plans.items():
            op = self.operations[p["cid"]]
            s = backend.bus.rigid.states[a]
            r = backend.bus.rigid.models[a]
            op["dc_energy_j"] += p["power_w"] * span
            op["rcs_heat_j"] += p["heat_w"] * span
            op["wall_heat_j"] += p["wall_w"] * span
            op["electrical_nonthermal_j"] += p["nonthermal_w"] * span
            if a in self.pending:
                mass = self.pending[a]
                used = mass["mass_before"] - mass["mass_after"]
                self.fuel_used[a] += used
                op["consumed_propellant_kg"] += used
                self.intervals.append(
                    {
                        "actor": a,
                        "command_id": p["cid"],
                        "start_sec": start,
                        "end_sec": backend.now(),
                        "scalar_nozzle_thrust_n": p["scalar_nozzle_thrust_n"],
                        "torque_body_nm": p["torque_body_nm"].tolist(),
                        "motor_torque_nm": p["motor_torque_nm"].tolist(),
                        "net_force_gcrs_n": [0.0, 0.0, 0.0],
                        **mass,
                    }
                )
            if p["phase"] == "startup":
                op["startup_sec"] += span
                op["startup_remaining_sec"] = max(
                    0.0, op["startup_remaining_sec"] - span
                )
            else:
                op["control_sec"] += span
                residual = np.asarray(s.wheel_momentum_nms) - op["target_momentum_nms"]
                pointing = np.linalg.norm(
                    error_vector(s.quaternion_wxyz, op["hold_quaternion_wxyz"])
                )
                qualified = (
                    np.max(np.abs(residual)) <= op["tolerance_nms"]
                    and np.linalg.norm(s.omega_body_rad_s) <= r.rate_tolerance_rad_s
                    and pointing
                    <= math.radians(
                        backend.bus.models[a].attitude.rigid_pointing_tolerance_deg
                    )
                )
                op["settle_remaining_sec"] = (
                    max(0.0, op["settle_remaining_sec"] - span)
                    if qualified and p["qualified_at_start"]
                    else backend.bus.models[a].attitude.settling_sec
                )
                if qualified and op["settle_remaining_sec"] <= 1e-9:
                    op.update(status="completed", completed_sec=backend.now())
            op["remaining_momentum_nms"] = (
                np.asarray(s.wheel_momentum_nms) - op["target_momentum_nms"]
            ).tolist()
            if op["status"] != "completed" and backend.now() >= op["until_sec"] - 1e-9:
                op.update(status="failed", reason="unload_window_ended")

    def clock_observation(self, actor):
        return {"mode":"fixed-joint-wheel-rcs/1", "period_sec":self.fixed_periods[actor],
                "samples":self.control_samples[actor],
                "last_sample_sec":self.control_latches[actor]["sample_sec"]
                if actor in self.control_latches else None}

    def observation(self, actor):
        if actor not in self.config.actors:
            return None
        return {
            **({"control_clock":self.clock_observation(actor)} if actor in self.fixed_periods else {}),
            "model": self.config.actors[actor].model_dump(mode="json"),
            "consumed_propellant_kg": self.fuel_used[actor],
            "operations": {
                k: copy.deepcopy(v)
                for k, v in self.operations.items()
                if v["actor"] == actor
            },
        }

    def report(self, backend):
        return {
            "schema": "space-env.momentum-unloading/1",
            **({"control_clocks": {a:self.clock_observation(a) for a in self.fixed_periods}}
               if self.fixed_periods else {}),
            "models": {
                a: c.model_dump(mode="json") for a, c in self.config.actors.items()
            },
            "operations": copy.deepcopy(self.operations),
            "intervals": copy.deepcopy(self.intervals),
            "fuel_used_kg": dict(self.fuel_used),
            "fuel_balance_error_kg": {
                a: self.initial_mass[a]
                - backend.mission.ephemerides[a].mass(backend.now())
                - used
                - (
                    backend.lifecycle.actuation.states.get(a, {}).get(
                        "consumed_propellant_kg", 0.0
                    )
                    if backend.lifecycle.actuation is not None
                    else 0.0
                )
                for a, used in self.fuel_used.items()
            },
            "energy_balance_error_j": {
                k: o["dc_energy_j"]
                + o["wall_heat_j"]
                - o["rcs_heat_j"]
                - o["electrical_nonthermal_j"]
                for k, o in self.operations.items()
            },
            "scope": "ideal continuous balanced nozzle pairs, shared lumped propellant, fixed COM/inertia small-mass-loss model; no flight calibration",
        }
