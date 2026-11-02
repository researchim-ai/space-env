"""Bounded body thrust, finite gimbal and paid engine phases.

Legacy held-body or explicit rigid rotating/off-center coupling with declared
constant-inertia mass applicability. Only applied intervals modify the orbit.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.propulsion import G0_M_S2
from space_env.reference.spacecraft_bus import angle_deg, rotate_body
from space_env.reference.thrust import FiniteThrust, RotatingThrust
from space_env.runtime.schema import FiniteScalar, StrictModel


class RigidThrustSettings(StrictModel):
    lever_arm_body_m: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    max_mass_loss_fraction: float = Field(default=0.001, gt=0, le=0.05, strict=True)
    max_attitude_step_rad: float = Field(default=0.25, gt=0, le=1, strict=True)

    @model_validator(mode="after")
    def applicability(self):
        if math.hypot(*self.lever_arm_body_m) > 100:
            raise ValueError("engine lever arm exceeds 100 m applicability")
        return self


class EngineSettings(StrictModel):
    rigid_coupling: RigidThrustSettings | None = None
    kind: Literal["chemical", "electric"]
    model_id: str = Field(min_length=1)
    source_status: Literal["synthetic", "published", "calibrated"]
    source: str = Field(min_length=1)
    mount_body: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (0.0, 0.0, 1.0)
    max_thrust_n: float = Field(gt=0, le=1e6, strict=True)
    max_gimbal_deg: float = Field(default=0.0, ge=0, le=45, strict=True)
    gimbal_rate_deg_s: float = Field(default=1.0, gt=0, le=360, strict=True)
    gimbal_settling_sec: float = Field(default=0.5, ge=0, le=3600, strict=True)
    gimbal_power_w: float = Field(default=2.0, ge=0, le=1e8, strict=True)
    ignition_sec: float = Field(default=0.5, ge=0, le=3600, strict=True)
    ignition_power_w: float = Field(default=3.0, ge=0, le=1e8, strict=True)
    firing_aux_power_w: float = Field(default=2.0, ge=0, le=1e8, strict=True)
    electrical_heat_fraction: float = Field(default=1.0, ge=0, le=1, strict=True)
    electric_efficiency: float | None = Field(default=None, ge=1e-3, le=1, strict=True)
    chemical_wall_heat_w_at_max: float = Field(default=0.0, ge=0, le=1e8, strict=True)
    min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)

    @model_validator(mode="after")
    def applicable(self):
        if not self.model_id.strip() or not self.source.strip():
            raise ValueError("engine model ID/source must be nonblank")
        if not math.isclose(math.hypot(*self.mount_body), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError("engine mount must be a unit body vector")
        if (self.kind == "electric") != (self.electric_efficiency is not None):
            raise ValueError("electric efficiency required only for electric engine")
        if self.max_gimbal_deg > 0 and self.gimbal_power_w <= 0:
            raise ValueError("movable gimbal requires positive actuator power")
        if self.kind == "electric" and self.chemical_wall_heat_w_at_max != 0:
            raise ValueError("electric engine cannot claim chemical wall heat")
        return self


class ActuationSettings(StrictModel):
    engines: dict[str, EngineSettings] = Field(min_length=1, max_length=64)
    max_segments: int = Field(default=256, ge=1, le=10000, strict=True)


class CoupledActuation:
    def __init__(self, config, backend):
        self.config = config
        if set(config.engines) - set(backend.actors):
            raise ValueError("engine references unknown actor")
        for a, model in config.engines.items():
            jet = (
                model.max_thrust_n
                * backend.config.orbits[a].propulsion.isp_s
                * G0_M_S2
                / 2
            )
            power = jet / (model.electric_efficiency or 1.0) + model.firing_aux_power_w
            if not math.isfinite(power * backend.horizon()):
                raise ValueError("engine energy exceeds finite model applicability")
            if (
                model.rigid_coupling is not None
                and backend.config.spacecraft_bus[a].attitude.rigid is None
            ):
                raise ValueError("rigid engine coupling requires a rigid actor")
            if model.rigid_coupling is not None:
                rigid = backend.config.spacecraft_bus[a].attitude.rigid
                if (
                    rigid.max_applicable_rate_rad_s
                    * min(rigid.control_step_sec, rigid.model.max_step_sec)
                    > model.rigid_coupling.max_attitude_step_rad
                ):
                    raise ValueError(
                        "rigid thrust angular step exceeds declared sampling applicability"
                    )
            if model.max_thrust_n > backend.config.orbits[a].propulsion.max_thrust_n:
                raise ValueError("engine thrust exceeds declared orbit hardware limit")
        self.initial_mass = {
            a: backend.mission.ephemerides[a].mass(0.0) for a in config.engines
        }
        self.states = {
            a: {
                "gimbal_body": list(model.mount_body),
                "settle_remaining_sec": 0.0,
                "consumed_propellant_kg": 0.0,
            }
            for a, model in config.engines.items()
        }
        self.operations, self.plans, self.pending = {}, {}, {}
        self.intervals = []

    def validate(self, command, backend):
        if command.actor_id not in self.config.engines:
            raise ValueError("actor has no declared engine")
        model, burn = self.config.engines[command.actor_id], command.body_burn
        if (
            command.actor_id in backend.bus.rigid.models
            and model.rigid_coupling is None
        ):
            raise ValueError(
                "rigid body burn requires explicit rigid_coupling and mass applicability"
            )
        if burn.thrust_n > model.max_thrust_n:
            raise ValueError("body thrust exceeds engine limit")
        if (
            angle_deg(model.mount_body, burn.direction_body)
            > model.max_gimbal_deg + 1e-8
        ):
            raise ValueError("requested direction exceeds gimbal cone")
        if backend.now() >= backend.horizon():
            raise ValueError("body burn requires future coverage")

    def admit(self, command, backend):
        self.validate(command, backend)
        self.operations[command.command_id] = {
            "actor": command.actor_id,
            "status": "accepted",
            "reason": None,
            "phase": "gimbal",
            "direction_body": list(command.body_burn.direction_body),
            "thrust_n": command.body_burn.thrust_n,
            "remaining_firing_sec": command.body_burn.firing_sec,
            "ignition_remaining_sec": self.config.engines[
                command.actor_id
            ].ignition_sec,
            "until_sec": backend.now() + command.duration_sec,
            "gimbal_sec": 0.0,
            "settling_sec": 0.0,
            "ignition_sec": 0.0,
            "firing_sec": 0.0,
            "dc_energy_j": 0.0,
            "engine_heat_j": 0.0,
            "jet_energy_j": 0.0,
            "chemical_wall_heat_j": 0.0,
            "electrical_nonthermal_j": 0.0,
            "consumed_propellant_kg": 0.0,
            "ideal_scalar_delta_v_m_s": 0.0,
            "impulse_vector_gcrs_ns": [0.0, 0.0, 0.0],
        }

    def _fail(self, cid, reason):
        op = self.operations[cid]
        a = op["actor"]
        if op["phase"] == "gimbal" and op["gimbal_sec"] > 0:
            self.states[a]["settle_remaining_sec"] = max(
                self.states[a]["settle_remaining_sec"],
                self.config.engines[a].gimbal_settling_sec,
            )
        op.update(status="failed", reason=reason)

    def cancel(self, cid):
        op = self.operations.get(cid)
        if op is None or op["status"] in {"completed", "failed", "cancelled"}:
            return False
        if op["phase"] == "gimbal" and op["gimbal_sec"] > 0:
            a = op["actor"]
            self.states[a]["settle_remaining_sec"] = max(
                self.states[a]["settle_remaining_sec"],
                self.config.engines[a].gimbal_settling_sec,
            )
        op.update(status="cancelled", reason="operator_cancelled")
        return True

    def prepare(self, backend, active):
        self.plans, self.pending = {}, {}
        segments_left = min(
            self.config.max_segments - len(self.intervals),
            backend.config.max_maneuvers
            - sum(len(p.maneuvers) for p in backend.mission.ephemerides.values()),
        )
        for row in sorted(
            active,
            key=lambda row: (
                backend.actors.index(row["command"]["actor_id"]),
                row["command"]["command_id"],
            ),
        ):
            c = row["command"]
            if c["kind"] != "body_burn":
                continue
            op = self.operations[c["command_id"]]
            if op["status"] in {"completed", "failed", "cancelled"}:
                continue
            a, cid = op["actor"], c["command_id"]
            model, state = self.config.engines[a], self.states[a]
            if (
                op["phase"] == "gimbal"
                and angle_deg(state["gimbal_body"], op["direction_body"]) <= 1e-6
            ):
                op["phase"] = "settling"
            if op["phase"] == "settling" and state["settle_remaining_sec"] <= 1e-9:
                op["phase"] = "ignition"
            if op["phase"] == "ignition" and op["ignition_remaining_sec"] <= 1e-9:
                op["phase"] = "firing"
            phase = op["phase"]
            remaining = (
                angle_deg(state["gimbal_body"], op["direction_body"])
                / model.gimbal_rate_deg_s
                if phase == "gimbal"
                else state["settle_remaining_sec"]
                if phase == "settling"
                else op["ignition_remaining_sec"]
                if phase == "ignition"
                else op["remaining_firing_sec"]
            )
            if phase == "gimbal":
                # Mechanical time is rounded upward to the runtime's 1 ns clock
                # quantum, preventing an acos roundoff from ending just before
                # another lane's exact completion boundary.
                remaining = math.ceil(remaining * 1e9) / 1e9
            jet = (
                op["thrust_n"] * backend.config.orbits[a].propulsion.isp_s * G0_M_S2 / 2
                if phase == "firing"
                else 0.0
            )
            power = (
                model.gimbal_power_w
                if phase in {"gimbal", "settling"}
                else model.ignition_power_w
                if phase == "ignition"
                else model.firing_aux_power_w
            )
            nonthermal = power * (1 - model.electrical_heat_fraction)
            heat = power * model.electrical_heat_fraction
            wall = 0.0
            if phase == "firing":
                if model.kind == "electric":
                    power += jet / model.electric_efficiency
                    heat += jet / model.electric_efficiency - jet
                else:
                    wall = (
                        model.chemical_wall_heat_w_at_max
                        * op["thrust_n"]
                        / model.max_thrust_n
                    )
                    heat += wall
            reason = backend.bus.allowed(a, "body_burn", cid, backend, extra_w=power)
            if (
                reason is None
                and backend.mission.energy_wh[a]
                < model.min_soc * backend.mission.vehicles[a].battery_wh - 1e-12
            ):
                reason = "propulsion_soc"
            if reason is not None:
                self._fail(cid, reason)
                continue
            if backend.bus.states[a]["settle_remaining_sec"] > 1e-9:
                phase, remaining, power, heat, jet, wall, nonthermal = (
                    "attitude_wait",
                    backend.bus.states[a]["settle_remaining_sec"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            vehicle = backend.mission.vehicles[a]
            net = backend.bus.generation_w[a] - backend.bus.battery_load(
                a,
                backend.bus.base_power(a, vehicle.base_power_w)
                + backend.bus.extra_power(a)
                + power,
            )
            if (
                net < 0
                and (backend.mission.energy_wh[a] - model.min_soc * vehicle.battery_wh)
                * 3600
                / -net
                <= 1e-9
            ):
                self._fail(cid, "propulsion_power")
                continue
            if phase == "firing":
                provider = backend.mission.ephemerides[a]
                fuel_sec = (
                    (provider.mass(backend.now()) - provider.dry_mass_kg)
                    * provider.isp_s
                    * G0_M_S2
                    / op["thrust_n"]
                )
                if fuel_sec <= 1e-9:
                    self._fail(cid, "propellant_exhausted")
                    continue
                if segments_left <= 0:
                    self._fail(cid, "burn_segment_budget")
                    continue
                if model.rigid_coupling is not None:
                    mass_floor = self.initial_mass[a] * (
                        1 - model.rigid_coupling.max_mass_loss_fraction
                    )
                    scope_sec = (
                        (provider.mass(backend.now()) - mass_floor)
                        * provider.isp_s
                        * G0_M_S2
                        / op["thrust_n"]
                    )
                    if scope_sec <= 1e-9:
                        self._fail(cid, "constant_inertia_mass_limit")
                        continue
                    remaining = min(remaining, scope_sec)
                remaining = min(remaining, fuel_sec)
                segments_left -= 1
            op["status"] = "running"
            op["active_phase"] = phase
            self.plans[a] = {
                "cid": cid,
                "phase": phase,
                "remaining_sec": remaining,
                "power_w": power,
                "heat_w": heat,
                "jet_w": jet,
                "wall_w": wall,
                "nonthermal_w": nonthermal,
                "min_soc": model.min_soc,
            }

        self.sync_rigid_torques(backend)

    def sync_rigid_torques(self, backend):
        for a in self.config.engines:
            if a not in backend.bus.rigid.models:
                continue
            plan = self.plans.get(a)
            firing = plan is not None and plan["phase"] == "firing"
            force = (
                np.asarray(self.states[a]["gimbal_body"])
                * self.operations[plan["cid"]]["thrust_n"]
                if firing
                else np.zeros(3)
            )
            coupling = self.config.engines[a].rigid_coupling
            torque = (
                np.cross(coupling.lever_arm_body_m, force)
                if coupling is not None
                else np.zeros(3)
            )
            backend.bus.rigid.set_external_torque(a, torque, rotating_burn=firing)

    def boundary(self, backend, right):
        for a, plan in list(self.plans.items()):
            now = backend.now()
            right = min(right, now + plan["remaining_sec"])
            vehicle = backend.mission.vehicles[a]
            base = backend.bus.base_power(
                a, vehicle.base_power_w
            ) + backend.bus.extra_power(a)
            net = backend.bus.generation_w[a] - backend.bus.battery_load(
                a, base + plan["power_w"]
            )
            seconds = (
                (backend.mission.energy_wh[a] - plan["min_soc"] * vehicle.battery_wh)
                * 3600
                / -net
                if net < 0
                else math.inf
            )
            if seconds <= 1e-9:
                self._fail(plan["cid"], "propulsion_power")
                del self.plans[a]
            else:
                right = min(right, now + seconds)
        self.sync_rigid_torques(backend)
        return right

    def power(self, actor):
        return self.plans.get(actor, {}).get("power_w", 0.0)

    def floor(self, actor):
        return self.plans.get(actor, {}).get("min_soc", 0.0)

    def heat_adjustment(self, actor, generic_fraction):
        plan = self.plans.get(actor)
        return (
            0.0 if plan is None else plan["heat_w"] - generic_fraction * plan["power_w"]
        )

    def apply(self, backend, span):
        for a, plan in self.plans.items():
            if plan["phase"] != "firing":
                continue
            op, provider = self.operations[plan["cid"]], backend.mission.ephemerides[a]
            direction = rotate_body(
                backend.bus.states[a]["quaternion_wxyz"], self.states[a]["gimbal_body"]
            )
            direction = direction / np.linalg.norm(direction)
            mass_before = provider.mass(backend.now())
            coupling = self.config.engines[a].rigid_coupling
            thrust = FiniteThrust(
                op["thrust_n"], provider.isp_s, provider.dry_mass_kg, direction
            )
            extra = {}
            if coupling is not None:
                _, path = backend.bus.rigid.predict(a, span)
                thrust = RotatingThrust(
                    op["thrust_n"],
                    provider.isp_s,
                    provider.dry_mass_kg,
                    tuple(direction),
                    tuple(self.states[a]["gimbal_body"]),
                    tuple(path["attitude_times_sec"]),
                    tuple(tuple(q) for q in path["attitude_quaternions_wxyz"]),
                )
                impulse = thrust.impulse_vector()
                extra = {
                    "direction_model": "space-env.rotating-body-thrust/1",
                    "direction_end_gcrs": thrust.direction_at(span).tolist(),
                    "torque_body_nm": backend.bus.rigid.plans[a][
                        "external_torque_body_nm"
                    ].tolist(),
                    "attitude_knots": len(thrust.attitude_times_sec),
                    "external_angular_impulse_gcrs_nms": path[
                        "external_angular_impulse_gcrs_nms"
                    ],
                }
            else:
                impulse = direction * op["thrust_n"] * span
            provider.burn(backend.now(), span, thrust)
            self.pending[a] = {
                "mass_before": mass_before,
                "mass_after": provider.mass(backend.now() + span),
                "direction_gcrs": direction.tolist(),
                "impulse_vector_gcrs_ns": impulse.tolist(),
                **extra,
            }
            backend.geometry_revision += 1
            backend._refresh_geometry(a)

    def finish(self, backend, start, span):
        for a, state in self.states.items():
            plan = self.plans.get(a)
            if plan is None or plan["phase"] == "attitude_wait":
                state["settle_remaining_sec"] = max(
                    0.0, state["settle_remaining_sec"] - span
                )
                continue
            op, model = self.operations[plan["cid"]], self.config.engines[a]
            phase = plan["phase"]
            op["dc_energy_j"] += plan["power_w"] * span
            op["engine_heat_j"] += plan["heat_w"] * span
            op["jet_energy_j"] += plan["jet_w"] * span
            op["chemical_wall_heat_j"] += plan["wall_w"] * span
            op["electrical_nonthermal_j"] += plan["nonthermal_w"] * span
            if phase == "gimbal":
                current, goal = (
                    np.asarray(state["gimbal_body"]),
                    np.asarray(op["direction_body"]),
                )
                theta = math.radians(angle_deg(current, goal))
                fraction = (
                    min(1.0, math.radians(model.gimbal_rate_deg_s * span) / theta)
                    if theta > 1e-12
                    else 1.0
                )
                axis = (
                    goal
                    if fraction >= 1
                    else (
                        math.sin((1 - fraction) * theta) * current
                        + math.sin(fraction * theta) * goal
                    )
                    / math.sin(theta)
                )
                state["gimbal_body"] = (axis / np.linalg.norm(axis)).tolist()
                remaining = angle_deg(state["gimbal_body"], goal)
                op["gimbal_sec"] += span
                if remaining <= 1e-6:
                    state["settle_remaining_sec"] = model.gimbal_settling_sec
            elif phase == "settling":
                state["settle_remaining_sec"] = max(
                    0.0, state["settle_remaining_sec"] - span
                )
                op["settling_sec"] += span
            elif phase == "ignition":
                op["ignition_remaining_sec"] = max(
                    0.0, op["ignition_remaining_sec"] - span
                )
                op["ignition_sec"] += span
            else:
                actual = self.pending[a]
                used = actual["mass_before"] - actual["mass_after"]
                state["consumed_propellant_kg"] += used
                op["consumed_propellant_kg"] += used
                op["firing_sec"] += span
                op["remaining_firing_sec"] = max(0.0, op["remaining_firing_sec"] - span)
                op["ideal_scalar_delta_v_m_s"] += (
                    backend.mission.ephemerides[a].isp_s
                    * G0_M_S2
                    * math.log(actual["mass_before"] / actual["mass_after"])
                )
                op["impulse_vector_gcrs_ns"] = (
                    np.asarray(op["impulse_vector_gcrs_ns"])
                    + np.asarray(actual["impulse_vector_gcrs_ns"])
                ).tolist()
                self.intervals.append(
                    {
                        "actor": a,
                        "command_id": plan["cid"],
                        "start_sec": start,
                        "end_sec": backend.now(),
                        "thrust_n": op["thrust_n"],
                        **actual,
                    }
                )
                if op["remaining_firing_sec"] <= max(1e-9, 4 * math.ulp(backend.now())):
                    op.update(status="completed", phase="completed")
            if op["status"] != "completed" and backend.now() >= op["until_sec"] - 1e-9:
                self._fail(plan["cid"], "burn_window_ended")

        for plan in self.plans.values():
            op = self.operations[plan["cid"]]
            if (
                op["status"] not in {"completed", "failed", "cancelled"}
                and backend.now() >= op["until_sec"] - 1e-9
            ):
                self._fail(plan["cid"], "burn_window_ended")

    def observation(self, actor):
        if actor not in self.states:
            return None
        return {
            "model": self.config.engines[actor].model_dump(mode="json"),
            "state": copy.deepcopy(self.states[actor]),
            "operations": {
                cid: copy.deepcopy(op)
                for cid, op in self.operations.items()
                if op["actor"] == actor
            },
        }

    def report(self, backend):
        return {
            "schema": "space-env.body-thrust/1",
            "models": {
                a: m.model_dump(mode="json") for a, m in self.config.engines.items()
            },
            "states": copy.deepcopy(self.states),
            "operations": copy.deepcopy(self.operations),
            "intervals": copy.deepcopy(self.intervals),
            "fuel_balance_error_kg": {
                a: self.initial_mass[a]
                - backend.mission.ephemerides[a].mass(backend.now())
                - s["consumed_propellant_kg"]
                - (
                    backend.lifecycle.unloader.fuel_used.get(a, 0.0)
                    if backend.lifecycle.unloader is not None
                    else 0.0
                )
                for a, s in self.states.items()
            },
            "electrical_heat_balance_error_j": {
                cid: op["dc_energy_j"]
                + op["chemical_wall_heat_j"]
                - op["engine_heat_j"]
                - op["electrical_nonthermal_j"]
                - (
                    op["jet_energy_j"]
                    if self.config.engines[op["actor"]].kind == "electric"
                    else 0.0
                )
                for cid, op in self.operations.items()
            },
            "forecast_contract": "applied thrust only; coast beyond executed intervals; no prediction of pending phases",
            "scope": "legacy held-body or explicit rigid rotating/off-center thrust with constant inertia mass-loss applicability; no flight or exhaust qualification",
        }
