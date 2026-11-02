"""Versioned coupled spacecraft bus with optional bounded rigid wheel attitude.

Legacy rate-limited or opt-in rigid control, body-fixed PV and battery converters;
optional network conduction/radiation/heaters or legacy RC, and causal health masks.
No flight calibration.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.attitude import align_z, boresight, slew_step
from space_env.core.constants import AU, SOLAR_CONSTANT
from space_env.core.environment import shadow_state
from space_env.core.frames import geodetic_to_ecef
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.orbit import _ephemerides
from space_env.reference.rigid_bus import RigidBus, RigidControllerSettings
from space_env.reference.thermal import TransmitterThermal
from space_env.reference.thermal_bus import CoupledThermalNetwork
from space_env.reference.thermal_network import ThermalNetworkSettings
from space_env.runtime.schema import FiniteScalar, StrictModel


class AttitudeSettings(StrictModel):
    rigid: RigidControllerSettings | None = None
    rigid_pointing_tolerance_deg: float = Field(default=0.1, gt=0, le=10, strict=True)
    initial_quaternion_wxyz: tuple[
        FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar
    ] = (1.0, 0.0, 0.0, 0.0)
    max_rate_deg_s: float = Field(default=2.0, gt=0, le=180, strict=True)
    settling_sec: float = Field(default=1.0, ge=0, strict=True)
    slew_power_w: float = Field(default=3.0, ge=0, strict=True)
    slew_min_soc: float = Field(default=0.02, ge=0, lt=1, strict=True)
    payload_tolerance_deg: float = Field(default=5.0, gt=0, le=90, strict=True)
    radio_tolerance_deg: float = Field(default=180.0, gt=0, le=180, strict=True)
    sun_keep_out_deg: float = Field(default=0.0, ge=0, le=90, strict=True)
    panel_normal_body: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (0.0, 0.0, 1.0)

    @model_validator(mode="after")
    def units(self):
        if not math.isclose(
            math.hypot(*self.initial_quaternion_wxyz), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("attitude quaternion must be unit body-to-GCRS wxyz")
        if not math.isclose(
            math.hypot(*self.panel_normal_body), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("panel body normal must be a unit vector")
        return self


class BusPower(StrictModel):
    max_load_w: float = Field(default=100.0, gt=0, strict=True)
    charge_efficiency: float = Field(default=0.95, gt=0, le=1, strict=True)
    discharge_efficiency: float = Field(default=0.9, gt=0, le=1, strict=True)
    safe_enter_soc: float = Field(default=0.05, ge=0, lt=1, strict=True)
    recovery_soc: float = Field(default=0.1, gt=0, le=1, strict=True)
    safe_base_power_w: float | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def hysteresis(self):
        if self.safe_enter_soc >= self.recovery_soc:
            raise ValueError("recovery SOC must exceed safe-entry SOC")
        if (
            self.safe_base_power_w is not None
            and self.safe_base_power_w > self.max_load_w
        ):
            raise ValueError("safe base power exceeds bus capacity")
        return self


class SpacecraftBusModel(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    parameter_status: Literal["synthetic", "published", "calibrated"]
    sources: tuple[str, ...] = Field(min_length=1)
    attitude: AttitudeSettings = Field(default_factory=AttitudeSettings)
    power: BusPower = Field(default_factory=BusPower)
    thermal: TransmitterThermal
    thermal_network: ThermalNetworkSettings | None = None
    electrical_heat_fraction: float = Field(default=0.8, ge=0, le=1, strict=True)
    safe_slew_allowed: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def scope(self):
        if (
            not self.model_id.strip()
            or not self.revision.strip()
            or any(not s.strip() for s in self.sources)
        ):
            raise ValueError("bus model requires nonblank ID/revision/provenance")
        if self.thermal.heat_tx_w != 0:
            raise ValueError(
                "bus thermal heat is computed from actual load; static heat_tx_w must be zero"
            )
        return self


def bus_preset(name: str) -> dict:
    if name not in {"synthetic-agile/1", "synthetic-constrained/1"}:
        raise ValueError("unknown spacecraft bus preset")
    constrained = name == "synthetic-constrained/1"
    return SpacecraftBusModel(
        model_id=name,
        revision="1",
        parameter_status="synthetic",
        sources=("SpaceEnv benchmark assumptions; no spacecraft calibration",),
        attitude=AttitudeSettings(
            max_rate_deg_s=0.5 if constrained else 2.0,
            slew_power_w=6.0 if constrained else 3.0,
        ),
        power=BusPower(max_load_w=18.0 if constrained else 100.0),
        thermal=TransmitterThermal(
            capacity_j_per_k=50.0 if constrained else 200.0,
            conductance_w_per_k=1.0,
            heat_tx_w=0.0,
            trip_k=310.0,
            resume_k=300.0,
            initial_k=285.0,
        ),
    ).model_dump(mode="json")


def rotate_body(quaternion, vector):
    q = np.asarray(quaternion, dtype=float)
    w, xyz = q[0], q[1:]
    v = np.asarray(vector, dtype=float)
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + w * v)


def pointing_quaternion(provider, target, clock):
    r = provider.state(clock)[:3]
    ground = provider.context.matrix(
        provider.epoch.shifted(clock)
    ).T @ geodetic_to_ecef(target.lat_deg, target.lon_deg, target.alt_m)
    return align_z(ground - r)


def angle_deg(a, b):
    return math.degrees(
        math.acos(
            float(
                np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
            )
        )
    )


def quaternion_angle(a, b):
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    left, right = left / np.linalg.norm(left), right / np.linalg.norm(right)
    if min(np.linalg.norm(left - right), np.linalg.norm(left + right)) < 1e-12:
        return 0.0
    return math.degrees(2 * math.acos(float(np.clip(abs(np.dot(left, right)), -1, 1))))


class CoupledResourceMission(ReferenceDownlinkMission):
    supports_body_fixed_solar = True
    bus = None

    def _generation(self, actor, midpoint):
        if self.bus is None:
            return 0.0
        return self.bus.generation_w[actor]

    def step(self, dt_sec, actions, **kwargs):
        original = self.vehicles
        self.vehicles = {
            a: v.model_copy(
                update={
                    "base_power_w": v.base_power_w
                    / self.bus.models[a].power.discharge_efficiency,
                    "tx_power_w": v.tx_power_w
                    / self.bus.models[a].power.discharge_efficiency,
                    "processing_power_w": v.processing_power_w
                    / self.bus.models[a].power.discharge_efficiency,
                }
            )
            for a, v in original.items()
        }
        try:
            return super().step(dt_sec, actions, **kwargs)
        finally:
            self.vehicles = original


class SpacecraftBus:
    def __init__(self, models, backend):
        self.models = models
        self.states = {
            a: {
                "quaternion_wxyz": list(m.attitude.initial_quaternion_wxyz),
                "temperature_k": m.thermal.initial_k,
                "safe": False,
                "safe_reasons": [],
                "faults": [],
                "solar_scale": 1.0,
                "rate_scale": 1.0,
                "settle_remaining_sec": 0.0,
                "angular_rate_deg_s": 0.0,
                "mean_slew_rate_deg_s": 0.0,
                "electrical_load_wh": 0.0,
                "raw_pv_wh": 0.0,
                "charge_loss_wh": 0.0,
                "discharge_loss_wh": 0.0,
                "heat_j": 0.0,
                "payload_heat_adjustment_j": 0.0,
                "rejected_heat_j": 0.0,
                "slew_sec": 0.0,
                "slew_energy_wh": 0.0,
            }
            for a, m in models.items()
        }
        self.initial_energy_wh = dict(backend.mission.energy_wh)
        for state in self.states.values():
            q = np.asarray(state["quaternion_wxyz"])
            state["quaternion_wxyz"] = (q / np.linalg.norm(q)).tolist()
        self.rigid = RigidBus(models)
        self.network = CoupledThermalNetwork(models, backend)
        for a in self.network.models:
            self.states[a]["temperature_k"] = max(self.network.temperatures[a].values())
        self.operations = {}
        self.events = []
        self.denials = []
        self.plans = {}
        self.generation_w = dict.fromkeys(models, 0.0)
        self.raw_generation_w = dict.fromkeys(models, 0.0)
        for actor, config in backend.mission.vehicles.items():
            if config.solar_mode != "body_fixed_bus" or config.thermal is not None:
                raise ValueError(
                    "coupled bus requires body_fixed_bus solar and one bus thermal node (no independent TX thermal)"
                )
            if (
                self.base_power(actor, config.base_power_w)
                > models[actor].power.max_load_w
            ):
                raise ValueError("base electrical load exceeds bus power rating")
        self.apply_protection(backend)

    def apply_protection(self, backend):
        for a, state in self.states.items():
            model = self.models[a]
            reasons = set(state["safe_reasons"])
            if a in self.network.models:
                reasons.update(self.network.violations(a))
            elif state["temperature_k"] >= model.thermal.trip_k:
                reasons.add("overtemperature")
            if (
                model.power.safe_enter_soc > 0
                and backend.mission.energy_wh[a]
                / backend.mission.vehicles[a].battery_wh
                <= model.power.safe_enter_soc
            ):
                reasons.add("low_soc")
            state["safe_reasons"] = sorted(reasons)
            state["safe"] = bool(reasons)

    def base_power(self, actor, nominal):
        state, model = self.states[actor], self.models[actor]
        raw = (
            model.power.safe_base_power_w
            if state["safe"] and model.power.safe_base_power_w is not None
            else nominal
        )

        return raw + self.network.heater_power(actor)

    def thermal_recovered(self, actor):
        return (
            self.network.recovered(actor)
            if actor in self.network.models
            else self.states[actor]["temperature_k"]
            <= self.models[actor].thermal.resume_k
        )

    def battery_load(self, actor, watts):
        return watts / self.models[actor].power.discharge_efficiency

    def _deny(self, actor, cid, reason, now):
        row = {"actor": actor, "command_id": cid, "time_sec": now, "reason": reason}
        if not self.denials or self.denials[-1] != row:
            self.denials.append(row)
        return reason

    def allowed(self, actor, kind, cid, backend, *, extra_w=0.0):
        state, model = self.states[actor], self.models[actor]
        faults = set(state["faults"])
        subsystem = (
            "propulsion"
            if kind == "body_burn"
            else "payload"
            if kind == "imaging"
            else "radio"
            if kind == "downlink"
            else "adcs"
            if kind in {"slew", "wheel_torque"}
            else "cpu"
        )
        reason = None
        if "bus" in faults or subsystem in faults:
            reason = f"{subsystem}_fault"
        elif kind in {"imaging", "body_burn"} and "adcs" in faults:
            reason = "adcs_fault"
        elif state["safe"] and not (
            kind == "slew" and model.safe_slew_allowed and self.thermal_recovered(actor)
        ):
            reason = "safe_mode"
        elif (
            self.base_power(actor, backend.mission.vehicles[actor].base_power_w)
            + extra_w
            + self.extra_power(actor)
            > model.power.max_load_w
        ):
            reason = "bus_power_capacity"
        if reason:
            self._deny(actor, cid, reason, backend.now())
        return reason

    def admit(self, command, backend):
        if command.kind == "wheel_torque":
            self.operations[command.command_id] = {
                "actor": command.actor_id,
                "goal": None,
                "status": "accepted",
                "motor_torque_nm": list(command.wheel_torque.motor_torque_nm),
                "until_sec": backend.now() + command.duration_sec,
                "reason": None,
                "settle_remaining_sec": 0.0,
            }
            return
        self.operations[command.command_id] = {
            "actor": command.actor_id,
            "pointing_tolerance_deg": self.models[
                command.actor_id
            ].attitude.rigid_pointing_tolerance_deg,
            "goal": (
                np.asarray(command.slew.quaternion_wxyz)
                / np.linalg.norm(command.slew.quaternion_wxyz)
            ).tolist(),
            "status": "accepted",
            "until_sec": backend.now() + command.duration_sec,
            "reason": None,
            "settle_remaining_sec": self.models[command.actor_id].attitude.settling_sec,
        }

    def cancel(self, cid):
        if cid not in self.operations or self.operations[cid]["status"] in {
            "completed",
            "failed",
            "cancelled",
        }:
            return False
        operation = self.operations[cid]
        if operation["status"] == "running":
            state = self.states[operation["actor"]]
            state["settle_remaining_sec"] = max(
                state["settle_remaining_sec"],
                self.models[operation["actor"]].attitude.settling_sec,
            )
            if operation["actor"] not in self.rigid.models:
                state["angular_rate_deg_s"] = 0.0
        if (
            operation["actor"] in self.rigid.models
            and self.plans.get(operation["actor"], {}).get("cid") == cid
        ):
            self.rigid.disable(operation["actor"], "operator_cancelled")
        operation.update(status="cancelled", reason="operator_cancelled")
        return True

    def prepare(self, backend, active):
        self.apply_protection(backend)
        self.rigid.plans = {}
        self.plans = {
            a: {
                "start": np.array(s["quaternion_wxyz"]),
                "goal": np.array(s["quaternion_wxyz"]),
                "kind": "hold",
                "cid": None,
            }
            for a, s in self.states.items()
        }
        for row in active:
            c = row["command"]
            a, cid = c["actor_id"], c["command_id"]
            model, state = self.models[a], self.states[a]
            if c["kind"] in {"slew", "wheel_torque"}:
                operation = self.operations[cid]
                if operation["status"] in {"completed", "failed", "cancelled"}:
                    continue
                self.plans[a].update(
                    kind=c["kind"],
                    goal=np.array(operation["goal"])
                    if operation["goal"] is not None
                    else self.plans[a]["start"],
                    cid=cid,
                )
                reason = self.allowed(a, "slew", cid, backend)
                floor = (
                    model.attitude.slew_min_soc * backend.mission.vehicles[a].battery_wh
                )
                if reason is None and backend.mission.energy_wh[a] < floor:
                    reason = "slew_power"
                if reason:
                    operation.update(status="failed", reason=reason)
                    self.plans[a].update(kind="hold", goal=self.plans[a]["start"])
                else:
                    operation["status"] = "running"
            elif c["kind"] == "imaging":
                operation = backend.lifecycle.operations.get(cid)
                if operation is None or operation["status"] in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    continue
                target = backend.acquisition_targets[
                    backend.lifecycle.requests[c["request_id"]].target_id
                ]
                goal = pointing_quaternion(
                    backend.mission.ephemerides[a], target, backend.now()
                )
                self.plans[a].update(kind="tracking", goal=goal, cid=cid)
        for a, state in self.states.items():
            provider = backend.mission.ephemerides[a]
            epoch = provider.epoch.shifted(backend.now())
            r = provider.state(backend.now())[:3]
            sun, _ = _ephemerides(epoch)
            to_sun = sun - r
            normal = rotate_body(
                state["quaternion_wxyz"], self.models[a].attitude.panel_normal_body
            )
            if a in self.network.models:
                normal = normal / np.linalg.norm(normal)
            cosine = max(0.0, float(np.dot(normal, to_sun) / np.linalg.norm(to_sun)))
            vehicle = backend.mission.vehicles[a]
            raw = (
                vehicle.solar_area_m2
                * vehicle.solar_efficiency
                * SOLAR_CONSTANT
                * (AU / np.linalg.norm(to_sun)) ** 2
                * shadow_state(r, sun).flux_fraction
                * cosine
                * state["solar_scale"]
            )
            self.raw_generation_w[a] = float(raw)
            self.generation_w[a] = float(raw * self.models[a].power.charge_efficiency)
        self.network.prepare(backend, self)
        self.rigid.prepare(backend, self)

    def stop_tracking(self, actor):
        if self.plans.get(actor, {}).get("kind") == "tracking":
            self.plans[actor].update(kind="hold", goal=self.plans[actor]["start"])
            if actor in self.rigid.models:
                self.rigid.disable(actor, "tracking_stopped")

    def extra_power(self, actor):
        if actor in self.rigid.models:
            return self.rigid.power(actor)
        plan = self.plans.get(actor)
        if plan is None or plan["kind"] == "hold":
            return 0.0
        moving = quaternion_angle(plan["start"], plan["goal"]) > 1e-6
        settling = (
            plan["kind"] == "slew"
            and self.operations[plan["cid"]]["settle_remaining_sec"] > 0
        )
        return self.models[actor].attitude.slew_power_w if moving or settling else 0.0

    def predicted_quaternion(self, actor, elapsed):
        if actor in self.rigid.models:
            return np.asarray(self.rigid.predict(actor, elapsed)[0].quaternion_wxyz)
        plan = self.plans[actor]
        if plan["kind"] == "hold" or elapsed <= 0:
            return plan["start"].copy()
        return slew_step(
            plan["start"],
            plan["goal"],
            self.models[actor].attitude.max_rate_deg_s
            * self.states[actor]["rate_scale"],
            elapsed,
        )[0]

    def boundary(self, backend, right):
        right = self.rigid.boundary(backend, self, right)
        right = self.network.boundary(backend, self, right)
        for a, plan in self.plans.items():
            if a in self.rigid.models or plan["kind"] != "slew":
                continue
            vehicle, model = backend.mission.vehicles[a], self.models[a]
            power = self.base_power(a, vehicle.base_power_w) + self.extra_power(a)
            net = self.generation_w[a] - self.battery_load(a, power)
            if net < 0:
                available = (
                    (
                        backend.mission.energy_wh[a]
                        - model.attitude.slew_min_soc * vehicle.battery_wh
                    )
                    * 3600
                    / -net
                )
                if available <= 1e-9:
                    self.operations[plan["cid"]].update(
                        status="failed", reason="slew_power"
                    )
                    plan.update(kind="hold", goal=plan["start"])
                    continue
                right = min(right, backend.now() + available)
            angle = quaternion_angle(plan["start"], plan["goal"])
            remaining = (
                angle
                / (
                    self.models[a].attitude.max_rate_deg_s
                    * self.states[a]["rate_scale"]
                )
                if angle > 1e-6
                else self.operations[plan["cid"]]["settle_remaining_sec"]
            )
            if remaining > 1e-9:
                right = min(right, backend.now() + remaining)
        return right

    def imaging_reason(self, actor, target, times, backend, cid):
        state, model = self.states[actor], self.models[actor]
        if actor in self.rigid.models and not self.rigid.power(actor):
            return self._deny(actor, cid, "rigid_controller_unpowered", backend.now())
        if state["settle_remaining_sec"] > 1e-9:
            return self._deny(actor, cid, "settling", backend.now())
        for t in times:
            provider = backend.mission.ephemerides[actor]
            goal = pointing_quaternion(provider, target, t)
            q = self.predicted_quaternion(actor, t - backend.now())
            if actor in self.rigid.models:
                predicted = self.rigid.predict(actor, t - backend.now())[0]
                if (
                    np.linalg.norm(predicted.omega_body_rad_s)
                    > self.rigid.models[actor].payload_max_body_rate_rad_s
                ):
                    return self._deny(actor, cid, "rigid_payload_rate", backend.now())
            if (
                angle_deg(boresight(q), boresight(goal))
                > model.attitude.payload_tolerance_deg
            ):
                return self._deny(actor, cid, "pointing", backend.now())
            r = provider.state(t)[:3]
            sun, _ = _ephemerides(provider.epoch.shifted(t))
            if angle_deg(boresight(q), sun - r) < model.attitude.sun_keep_out_deg:
                return self._deny(actor, cid, "sun_keep_out", backend.now())
        return None

    def ordinary_allowed(self, row, backend, *, span=0.0):
        c = row["command"]
        actor, kind = c["actor_id"], c["kind"]
        if kind not in {"downlink", "processing"}:
            return True
        vehicle = backend.mission.vehicles[actor]
        watts = vehicle.tx_power_w if kind == "downlink" else vehicle.processing_power_w
        if self.allowed(actor, kind, c["command_id"], backend, extra_w=watts):
            return False
        if kind == "downlink" and self.models[actor].attitude.radio_tolerance_deg < 180:
            station = backend.mission.stations[c["destination_id"]]
            provider = backend.mission.ephemerides[actor]
            samples = (0.0, span / 2, span) if actor in self.rigid.models else (0.0,)
            for elapsed in samples:
                t = backend.now() + elapsed
                target = provider.context.matrix(
                    provider.epoch.shifted(t)
                ).T @ geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
                q = (
                    self.predicted_quaternion(actor, elapsed)
                    if actor in self.rigid.models
                    else self.states[actor]["quaternion_wxyz"]
                )
                if (
                    angle_deg(boresight(q), target - provider.state(t)[:3])
                    > self.models[actor].attitude.radio_tolerance_deg
                ):
                    self._deny(actor, c["command_id"], "radio_pointing", backend.now())
                    return False
        return True

    def integrate(self, backend, now, span, before):
        mission = backend.mission
        for a, state in self.states.items():
            model, plan = self.models[a], self.plans[a]
            qnew = self.predicted_quaternion(a, span)
            moved = quaternion_angle(plan["start"], qnew)
            if a in self.rigid.models:
                self.rigid.integrate(a, self, backend, span)
                state["mean_slew_rate_deg_s"] = moved / span
            else:
                state["quaternion_wxyz"] = qnew.tolist()
                state["mean_slew_rate_deg_s"] = moved / span
                state["angular_rate_deg_s"] = (
                    model.attitude.max_rate_deg_s * state["rate_scale"]
                    if quaternion_angle(qnew, plan["goal"]) > 1e-6
                    else 0.0
                )
                if plan["kind"] == "slew":
                    operation = self.operations[plan["cid"]]
                    if quaternion_angle(qnew, plan["goal"]) <= 1e-6:
                        travel = quaternion_angle(plan["start"], plan["goal"]) / (
                            model.attitude.max_rate_deg_s * state["rate_scale"]
                        )
                        operation["settle_remaining_sec"] = max(
                            0.0,
                            operation["settle_remaining_sec"] - max(0.0, span - travel),
                        )
                        state["settle_remaining_sec"] = operation[
                            "settle_remaining_sec"
                        ]
                        if operation["settle_remaining_sec"] <= 1e-9:
                            operation["status"] = "completed"
                            state["angular_rate_deg_s"] = 0.0
                    if (
                        operation["status"] != "completed"
                        and mission.clock >= operation["until_sec"] - 1e-9
                    ):
                        operation.update(status="failed", reason="slew_window_ended")
                        state["settle_remaining_sec"] = max(
                            state["settle_remaining_sec"], model.attitude.settling_sec
                        )
                        state["angular_rate_deg_s"] = 0.0
                else:
                    state["settle_remaining_sec"] = max(
                        0.0, state["settle_remaining_sec"] - span
                    )
            raw_pv = self.raw_generation_w[a] * span / 3600
            generated = mission.generated_wh[a] - before[a][0]
            withdrawn = mission.consumed_wh[a] - before[a][1]
            electrical = withdrawn * model.power.discharge_efficiency
            charge_loss, discharge_loss = raw_pv - generated, withdrawn - electrical
            state["raw_pv_wh"] += raw_pv
            state["electrical_load_wh"] += electrical
            state["charge_loss_wh"] += charge_loss
            state["discharge_loss_wh"] += discharge_loss
            adcs = before[a][2] * span / 3600
            state["slew_energy_wh"] += adcs
            if adcs > 0:
                state["slew_sec"] += span
            heat = (
                electrical * model.electrical_heat_fraction
                + charge_loss
                + discharge_loss
            ) * 3600
            adjustment = before[a][3] * span
            heat += adjustment
            state["payload_heat_adjustment_j"] += adjustment
            if a in self.network.models:
                heat, rejected = self.network.integrate(
                    backend,
                    self,
                    a,
                    now,
                    span,
                    before[a],
                    heat,
                    (charge_loss + discharge_loss) * 3600,
                    raw_pv * 3600,
                )
                state["temperature_k"] = max(self.network.temperatures[a].values())
                state["heat_j"] += heat
                state["rejected_heat_j"] += rejected
            else:
                thermal = model.thermal.model_copy(update={"heat_tx_w": heat / span})
                old = state["temperature_k"]
                new = thermal.advance(old, span, True)
                state["temperature_k"] = new
                state["heat_j"] += heat
                state["rejected_heat_j"] += heat - model.thermal.capacity_j_per_k * (
                    new - old
                )

    def inject(self, backend, event):
        payload = event["payload"]
        kind = event["type"]
        actor = payload.get("actor")
        if actor not in self.states:
            raise ValueError("unknown spacecraft actor")
        state, model = self.states[actor], self.models[actor]
        if kind in {"spacecraft_safe_mode", "spacecraft_recover"}:
            if set(payload) != {"actor"}:
                raise ValueError("safe/recover requires exactly actor")
            if kind == "spacecraft_safe_mode":
                state["safe"] = True
                state["safe_reasons"] = sorted(
                    set(state["safe_reasons"]) | {"operator"}
                )
            else:
                soc = (
                    backend.mission.energy_wh[actor]
                    / backend.mission.vehicles[actor].battery_wh
                )
                if (
                    state["faults"]
                    or not self.thermal_recovered(actor)
                    or soc < model.power.recovery_soc
                ):
                    raise ValueError(
                        "recovery requires cleared faults, thermal hysteresis and recovery SOC"
                    )
                state["safe"] = False
                state["safe_reasons"] = []
        elif kind in {"spacecraft_fault", "spacecraft_fault_cleared"}:
            if set(payload) != {"actor", "subsystem"} or payload["subsystem"] not in {
                "bus",
                "adcs",
                "payload",
                "cpu",
                "radio",
                "propulsion",
            }:
                raise ValueError("fault event requires known subsystem")
            faults = set(state["faults"])
            if kind == "spacecraft_fault":
                faults.add(payload["subsystem"])
            else:
                if payload["subsystem"] not in faults:
                    raise ValueError("subsystem has no fault to clear")
                faults.remove(payload["subsystem"])
            state["faults"] = sorted(faults)
        elif kind == "spacecraft_degraded":
            if set(payload) != {"actor", "solar_scale", "rate_scale"}:
                raise ValueError("degradation requires solar/rate scales")
            for key in ("solar_scale", "rate_scale"):
                value = payload[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not (0 <= value <= state[key])
                    or (key == "rate_scale" and value == 0)
                ):
                    raise ValueError(
                        "degradation scales must be finite, monotonic, rate positive"
                    )
                state[key] = float(value)
        else:
            raise ValueError("unknown spacecraft event")
        self.events.append(copy.deepcopy(event))

    def observation(self, actor):
        return {
            **copy.deepcopy(self.states[actor]),
            **(
                {"thermal_network": self.network.observation(actor)}
                if actor in self.network.models
                else {}
            ),
            **(
                {"rigid_attitude": self.rigid.observation(actor)}
                if actor in self.rigid.models
                else {}
            ),
        }

    def report(self, backend):
        ledgers = {}
        for a, state in self.states.items():
            model = self.models[a]
            mission = backend.mission
            stored = (
                self.network.stored(a)
                if a in self.network.models
                else model.thermal.capacity_j_per_k
                * (state["temperature_k"] - model.thermal.initial_k)
            )
            ledgers[a] = {
                "battery_balance_error_wh": self.initial_energy_wh[a]
                + mission.generated_wh[a]
                - mission.consumed_wh[a]
                - mission.curtailed_wh[a]
                - mission.energy_wh[a],
                "thermal_stored_j": stored,
                "thermal_balance_error_j": state["heat_j"]
                - state["rejected_heat_j"]
                - stored,
                "pv_balance_error_wh": state["raw_pv_wh"]
                - state["charge_loss_wh"]
                - mission.generated_wh[a],
                "discharge_balance_error_wh": mission.consumed_wh[a]
                - state["discharge_loss_wh"]
                - state["electrical_load_wh"],
            }
        return {
            "schema": "space-env.spacecraft-bus/1",
            "models": {a: m.model_dump(mode="json") for a, m in self.models.items()},
            "states": {a: self.observation(a) for a in self.states},
            "operations": copy.deepcopy(self.operations),
            "events": copy.deepcopy(self.events),
            "denials": copy.deepcopy(self.denials),
            "ledgers": ledgers,
            "scope": "opt-in rigid wheel attitude or legacy rate-limited kinematics; body-fixed PV/two-port battery, optional multi-node thermal or legacy RC; no flight qualification",
        }
