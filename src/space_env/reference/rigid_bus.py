"""Opt-in mission attitude controller over the bounded rigid-body kernel.

A fixed electrical supply envelope is reserved and charged while enabled; unused
supply power becomes heat. Optional thrust torque is set by engine execution;
no free external momentum unloading actuator is exposed.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from pydantic import Field, model_validator

from space_env.reference.rigid_attitude import (
    RigidAttitudeModel,
    RigidAttitudeState,
    advance_rigid_attitude,
    invariants,
)
from space_env.runtime.schema import FiniteScalar, StrictModel


class RigidControllerSettings(StrictModel):
    model: RigidAttitudeModel
    initial_omega_body_rad_s: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (
        0.0,
        0.0,
        0.0,
    )
    initial_wheel_momentum_nms: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (
        0.0,
        0.0,
        0.0,
    )
    supply_w: float = Field(gt=0, le=1e8, strict=True)
    proportional_nm_per_rad: float = Field(default=1.0, gt=0, strict=True)
    damping_nm_per_rad_s: float = Field(default=2.0, gt=0, strict=True)
    rate_tolerance_rad_s: float = Field(default=1e-3, gt=0, strict=True)
    payload_max_body_rate_rad_s: float = Field(default=0.02, gt=0, strict=True)
    max_applicable_rate_rad_s: float = Field(default=10.0, gt=0, le=100, strict=True)
    control_step_sec: float = Field(default=0.1, ge=0.01, le=0.5, strict=True)
    max_total_substeps: int = Field(default=100000, ge=1, le=1000000, strict=True)

    @model_validator(mode="after")
    def physical(self):
        if (
            np.linalg.norm(self.initial_omega_body_rad_s)
            > self.max_applicable_rate_rad_s
        ):
            raise ValueError("initial spin outside rigid applicability")
        if np.any(
            np.abs(self.initial_wheel_momentum_nms)
            > np.asarray(self.model.max_momentum_nms)
        ):
            raise ValueError("initial wheel momentum exceeds capacity")
        if self.rate_tolerance_rad_s >= self.max_applicable_rate_rad_s:
            raise ValueError("rate tolerance must be below applicability limit")
        u = np.asarray(self.model.max_torque_nm)
        bound = (
            self.model.idle_power_w
            + float(
                u
                @ (
                    np.asarray(self.model.max_momentum_nms)
                    / self.model.wheel_inertia_kg_m2
                    + self.max_applicable_rate_rad_s
                )
            )
            / self.model.efficiency
        )
        bound += self.model.torque_loss_w_per_nm2 * float(u @ u)
        if self.supply_w < bound:
            raise ValueError("supply must cover conservative motor power envelope")
        if (
            math.ceil(self.control_step_sec / self.model.max_step_sec)
            > self.model.max_steps
        ):
            raise ValueError("kernel step budget cannot cover a control interval")
        return self


def error_vector(current, goal):
    """Shortest target rotation expressed in current body axes, radians."""
    a, b = np.asarray(current), np.asarray(goal)
    w = a[0] * b[0] + a[1:] @ b[1:]
    xyz = a[0] * b[1:] - b[0] * a[1:] - np.cross(a[1:], b[1:])
    if w < 0:
        w, xyz = -w, -xyz
    size = np.linalg.norm(xyz)
    return np.zeros(3) if size < 1e-15 else xyz / size * (2 * math.atan2(size, w))


class RigidBus:
    def __init__(self, models):
        self.models = {
            a: m.attitude.rigid
            for a, m in models.items()
            if m.attitude.rigid is not None
        }
        self.states = {
            a: RigidAttitudeState(
                quaternion_wxyz=tuple(
                    np.asarray(models[a].attitude.initial_quaternion_wxyz)
                    / np.linalg.norm(models[a].attitude.initial_quaternion_wxyz)
                ),
                omega_body_rad_s=c.initial_omega_body_rad_s,
                wheel_momentum_nms=c.initial_wheel_momentum_nms,
            )
            for a, c in self.models.items()
        }
        self.initial = {
            a: invariants(c.model, self.states[a]) for a, c in self.models.items()
        }
        self.plans, self.cache = {}, {}
        self.external_impulse = {a: np.zeros(3) for a in self.models}
        self.substeps = dict.fromkeys(self.models, 0)
        self.saturation_sec = {a: np.zeros(3) for a in self.models}

    def power(self, actor):
        return (
            self.models[actor].supply_w
            if self.plans.get(actor, {}).get("enabled")
            else 0.0
        )

    def disable(self, actor, reason):
        self.plans[actor].update(enabled=False, torque=np.zeros(3), reason=reason)
        self.cache = {key: v for key, v in self.cache.items() if key[0] != actor}

    def prepare(self, backend, bus):
        self.plans, self.cache = {}, {}
        for a, c in self.models.items():
            plan, state = bus.plans[a], self.states[a]
            requested = np.zeros(3)
            enabled = plan["kind"] in {"slew", "tracking", "wheel_torque"}
            self.plans[a] = {
                "enabled": False,
                "torque": requested,
                "reason": None,
                "external_torque_body_nm": np.zeros(3),
                "rotating_burn": False,
            }
            if enabled:
                if plan["kind"] == "wheel_torque":
                    requested = np.asarray(
                        bus.operations[plan["cid"]]["motor_torque_nm"]
                    )
                else:
                    error = error_vector(state.quaternion_wxyz, plan["goal"])
                    # PD body torque, with rate limiting via target body rate.
                    max_rate = (
                        math.radians(bus.models[a].attitude.max_rate_deg_s)
                        * bus.states[a]["rate_scale"]
                    )
                    goal_rate = (
                        error * c.proportional_nm_per_rad / c.damping_nm_per_rad_s
                    )
                    norm = np.linalg.norm(goal_rate)
                    if norm > max_rate:
                        goal_rate *= max_rate / norm
                    requested = -c.damping_nm_per_rad_s * (
                        goal_rate - np.asarray(state.omega_body_rad_s)
                    )
                limit = np.asarray(c.model.max_torque_nm) * bus.states[a]["rate_scale"]
                requested = np.clip(requested, -limit, limit)
                reason = None
                flags = bus.states[a]
                if flags["safe"] or {"bus", "adcs"} & set(flags["faults"]):
                    reason = "rigid_protection"
                elif (
                    bus.base_power(a, backend.mission.vehicles[a].base_power_w)
                    + c.supply_w
                    > bus.models[a].power.max_load_w
                ):
                    reason = "rigid_bus_power"
                elif (
                    backend.mission.energy_wh[a]
                    <= bus.models[a].attitude.slew_min_soc
                    * backend.mission.vehicles[a].battery_wh
                    + 1e-12
                ):
                    reason = "rigid_soc"
                if reason:
                    self.disable(a, reason)
                    if plan["cid"] in bus.operations:
                        bus.operations[plan["cid"]].update(
                            status="failed", reason=reason
                        )
                else:
                    self.plans[a].update(enabled=True, torque=requested)

    def set_external_torque(self, actor, torque, *, rotating_burn=False):
        self.plans[actor].update(
            external_torque_body_nm=np.asarray(torque), rotating_burn=rotating_burn
        )
        self.cache = {key: v for key, v in self.cache.items() if key[0] != actor}

    def predict(self, actor, span):
        if span <= 0:
            return self.states[actor], {
                "substeps": 0,
                "capacity_limited_sec": [0.0, 0.0, 0.0],
            }
        c = self.models[actor]
        if span > c.control_step_sec + 1e-8:
            raise ValueError("rigid preview exceeds control interval")
        key = (actor, span)
        if key not in self.cache:
            state = self.states[actor]
            enabled = self.plans[actor]["enabled"]
            m = c.model if enabled else c.model.model_copy(update={"idle_power_w": 0.0})
            result, report = advance_rigid_attitude(
                m,
                state,
                span,
                motor_torque_nm=self.plans[actor]["torque"],
                angular_rate_limit_rad_s=c.max_applicable_rate_rad_s,
                external_torque_body_nm=self.plans[actor]["external_torque_body_nm"],
                return_path=self.plans[actor]["rotating_burn"],
            )
            if self.substeps[actor] + report["substeps"] > c.max_total_substeps:
                raise ValueError("rigid mission substep budget exhausted")
            if np.linalg.norm(result.omega_body_rad_s) > c.max_applicable_rate_rad_s:
                raise ValueError("final angular rate outside declared applicability")
            supplied = self.power(actor) * span
            mechanical = result.motor_work_j - state.motor_work_j
            # Charge the declared constant supply, including dumped unused power.
            if result.dc_energy_j - state.dc_energy_j > supplied + 1e-8:
                raise ValueError("rigid motor exceeded reserved supply")
            result = result.model_copy(
                update={
                    "dc_energy_j": state.dc_energy_j + supplied,
                    "heat_j": state.heat_j + supplied - mechanical,
                }
            )
            self.cache[key] = result, report
        return self.cache[key]

    def boundary(self, backend, bus, right):
        for a, c in self.models.items():
            # Anchor to mission time to avoid repeated addition ending a dwell
            # a few ulps before its exact command boundary.
            tick = (
                math.floor((backend.now() + 1e-9) / c.control_step_sec) + 1
            ) * c.control_step_sec
            right = min(right, tick)
            if not self.power(a):
                continue
            v = backend.mission.vehicles[a]
            net = bus.generation_w[a] - bus.battery_load(
                a, bus.base_power(a, v.base_power_w) + self.power(a)
            )
            available = (
                backend.mission.energy_wh[a]
                - bus.models[a].attitude.slew_min_soc * v.battery_wh
            )
            if net < 0:
                seconds = available * 3600 / -net
                if seconds <= 1e-9:
                    self.disable(a, "rigid_soc")
                    cid = bus.plans[a]["cid"]
                    if cid in bus.operations:
                        bus.operations[cid].update(status="failed", reason="rigid_soc")
                else:
                    right = min(right, backend.now() + seconds)
        return right

    def heat_adjustment(self, actor, span, heat_fraction):
        if actor not in self.models:
            return 0.0
        result, _ = self.predict(actor, span)
        work = result.motor_work_j - self.states[actor].motor_work_j
        return self.power(actor) * (1 - heat_fraction) - work / span

    def integrate(self, actor, bus, backend, span):
        old = self.states[actor]
        new, report = self.predict(actor, span)
        self.states[actor] = new
        self.substeps[actor] += report["substeps"]
        self.external_impulse[actor] += report["external_angular_impulse_gcrs_nms"]
        self.saturation_sec[actor] += report["capacity_limited_sec"]
        state, plan = bus.states[actor], bus.plans[actor]
        state["quaternion_wxyz"] = list(new.quaternion_wxyz)
        state["angular_rate_deg_s"] = math.degrees(np.linalg.norm(new.omega_body_rad_s))
        if plan["cid"] not in bus.operations:
            state["settle_remaining_sec"] = max(
                0.0, state["settle_remaining_sec"] - span
            )
            return
        op = bus.operations[plan["cid"]]
        if op["status"] in {"failed", "cancelled", "completed"}:
            return
        if plan["kind"] == "slew":
            config = self.models[actor]

            def settled(s):
                return (
                    np.linalg.norm(error_vector(s.quaternion_wxyz, op["goal"]))
                    <= math.radians(op["pointing_tolerance_deg"])
                    and np.linalg.norm(s.omega_body_rad_s)
                    <= config.rate_tolerance_rad_s
                )

            remaining = (
                max(0.0, op["settle_remaining_sec"] - span)
                if settled(old) and settled(new)
                else bus.models[actor].attitude.settling_sec
            )
            op["settle_remaining_sec"] = remaining
            state["settle_remaining_sec"] = remaining
            if settled(new) and remaining <= 1e-9:
                op.update(
                    status="completed",
                    completed_sec=backend.now(),
                    final_pointing_error_deg=math.degrees(
                        np.linalg.norm(error_vector(new.quaternion_wxyz, op["goal"]))
                    ),
                    final_rate_rad_s=float(np.linalg.norm(new.omega_body_rad_s)),
                )
            elif backend.now() >= op["until_sec"] - 1e-9:
                op.update(status="failed", reason="rigid_slew_window_ended")
        elif backend.now() >= op["until_sec"] - 1e-9:
            op.update(status="completed")

    def observation(self, actor):
        state = self.states[actor]
        conserved = invariants(self.models[actor].model, state)
        return {
            **state.model_dump(mode="json"),
            **conserved,
            "dc_heat_work_balance_error_j": state.dc_energy_j
            - state.heat_j
            - state.motor_work_j,
            "kinetic_work_balance_error_j": conserved["kinetic_energy_j"]
            - self.initial[actor]["kinetic_energy_j"]
            - state.motor_work_j
            - state.external_work_j,
            "external_angular_impulse_gcrs_nms": self.external_impulse[actor].tolist(),
            "angular_momentum_balance_error_gcrs_nms": (
                np.asarray(conserved["momentum_inertial_nms"])
                - self.initial[actor]["momentum_inertial_nms"]
                - self.external_impulse[actor]
            ).tolist(),
            "substeps": self.substeps[actor],
            "capacity_limited_sec": self.saturation_sec[actor].tolist(),
            "controller": copy.deepcopy(
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self.plans.get(actor, {}).items()
                }
            ),
            "scope": "mission rigid controller, optional rotating body thrust; no dedicated unloading actuator",
        }
