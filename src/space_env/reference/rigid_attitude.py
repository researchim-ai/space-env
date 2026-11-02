"""Standalone rigid body + three orthogonal ideal reaction wheels, SI units.

Reusable kernel; optional mission integration is in rigid_bus. Quaternion is wxyz;
wheel momentum is expressed in fixed body axes. No automatic momentum reset.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar, StrictModel


class RigidAttitudeModel(StrictModel):
    model_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    inertia_kg_m2: tuple[tuple[FiniteScalar, FiniteScalar, FiniteScalar], ...]
    wheel_inertia_kg_m2: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    max_torque_nm: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    max_momentum_nms: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    efficiency: float = Field(default=0.8, gt=0, le=1, strict=True)
    idle_power_w: float = Field(default=0.0, ge=0, strict=True)
    torque_loss_w_per_nm2: float = Field(default=1.0, gt=0, strict=True)
    max_step_sec: float = Field(default=0.01, gt=0, le=1, strict=True)
    max_steps: int = Field(default=100000, ge=1, le=1000000, strict=True)

    @model_validator(mode="after")
    def physical(self):
        if not all(
            math.isfinite(v)
            for v in (
                self.efficiency,
                self.idle_power_w,
                self.torque_loss_w_per_nm2,
                self.max_step_sec,
            )
        ):
            raise ValueError("model scalars must be finite")
        i = np.asarray(self.inertia_kg_m2)
        if not self.model_id.strip() or not self.source.strip():
            raise ValueError("model ID and provenance must be nonblank")
        if i.shape != (3, 3) or not np.allclose(i, i.T, rtol=0, atol=1e-12):
            raise ValueError("inertia must be symmetric 3x3")
        eig = np.linalg.eigvalsh(i)
        if eig[0] <= 0 or eig[-1] > eig[0] + eig[1] + 1e-12:
            raise ValueError(
                "inertia must be positive definite and physically realizable"
            )
        for values in (
            self.wheel_inertia_kg_m2,
            self.max_torque_nm,
            self.max_momentum_nms,
        ):
            if min(values) <= 0:
                raise ValueError(
                    "wheel inertia, torque and momentum limits must be positive"
                )
        return self


class RigidAttitudeState(StrictModel):
    quaternion_wxyz: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    omega_body_rad_s: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (0.0, 0.0, 0.0)
    wheel_momentum_nms: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (
        0.0,
        0.0,
        0.0,
    )
    elapsed_sec: float = Field(default=0.0, ge=0, strict=True)
    dc_energy_j: float = Field(default=0.0, ge=0, strict=True)
    heat_j: float = Field(default=0.0, ge=0, strict=True)
    motor_work_j: FiniteScalar = 0.0
    external_work_j: FiniteScalar = 0.0

    @model_validator(mode="after")
    def quaternion(self):
        if not all(
            math.isfinite(v) for v in (self.elapsed_sec, self.dc_energy_j, self.heat_j)
        ):
            raise ValueError("state counters must be finite")
        if not math.isclose(
            math.hypot(*self.quaternion_wxyz), 1.0, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("quaternion must be unit body-to-inertial wxyz")
        return self


def _vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("expected finite SI 3-vector")
    return value


def thrust_torque_nm(lever_arm_body_m, force_body_n):
    """Moment about COM; caller must also apply force to translation separately."""
    return np.cross(_vector(lever_arm_body_m), _vector(force_body_n))


def _rotate(q, v):
    return v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def invariants(model: RigidAttitudeModel, state: RigidAttitudeState):
    w, h = np.asarray(state.omega_body_rad_s), np.asarray(state.wheel_momentum_nms)
    i = np.asarray(model.inertia_kg_m2)
    return {
        "momentum_inertial_nms": _rotate(
            np.asarray(state.quaternion_wxyz), i @ w + h
        ).tolist(),
        "kinetic_energy_j": float(
            (w @ i @ w + np.sum(h * h / model.wheel_inertia_kg_m2)) / 2
        ),
    }


def advance_rigid_attitude(
    model,
    state,
    dt_sec,
    *,
    motor_torque_nm=(0.0, 0.0, 0.0),
    external_torque_body_nm=(0.0, 0.0, 0.0),
    angular_rate_limit_rad_s=None,
    return_path=False,
):
    """Pure bounded RK4 advance; no state mutation on rejection or failure.

    Inputs are constant body-axis torques during this call. Motor torque is
    clipped per axis, then stops at momentum capacity; no regeneration to DC bus.
    External torque represents a separately supplied physical actuator/disturbance.
    """
    if not math.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("dt_sec must be positive finite")
    if angular_rate_limit_rad_s is not None and (
        not math.isfinite(angular_rate_limit_rad_s) or angular_rate_limit_rad_s <= 0
    ):
        raise ValueError("angular rate limit must be positive finite")
    ratio = dt_sec / model.max_step_sec
    if not math.isfinite(ratio) or ratio > model.max_steps:
        raise ValueError("integration step budget exceeded")
    n = math.ceil(ratio)
    dt = dt_sec / n
    requested, external = _vector(motor_torque_nm), _vector(external_torque_body_nm)
    limits = np.asarray(model.max_momentum_nms)
    h0 = np.asarray(state.wheel_momentum_nms)
    if np.any(np.abs(h0) > limits + 1e-12):
        raise ValueError("initial wheel momentum exceeds capacity")
    i, ji = np.asarray(model.inertia_kg_m2), np.asarray(model.wheel_inertia_kg_m2)
    inverse = np.linalg.inv(i)
    # q, omega, h, DC energy, heat, motor work, external work
    y = np.r_[
        state.quaternion_wxyz,
        state.omega_body_rad_s,
        h0,
        state.dc_energy_j,
        state.heat_j,
        state.motor_work_j,
        state.external_work_j,
        0.0,
        0.0,
        0.0,
    ]
    torque = np.clip(requested, -np.asarray(model.max_torque_nm), model.max_torque_nm)
    path_times, path_quaternions = [0.0], [tuple(state.quaternion_wxyz)]
    saturated_sec = np.zeros(3)
    for index in range(n):
        u = np.clip(torque, (-limits - y[7:10]) / dt, (limits - y[7:10]) / dt)
        saturated_sec += (np.abs(u - torque) > 1e-12) * dt

        def rhs(v, u=u):
            q, w, h = v[:4], v[4:7], v[7:10]
            if (
                angular_rate_limit_rad_s is not None
                and np.linalg.norm(w) > angular_rate_limit_rad_s
            ):
                raise ValueError("angular rate outside declared applicability")
            qdot = 0.5 * np.r_[-np.dot(q[1:], w), q[0] * w + np.cross(q[1:], w)]
            wdot = inverse @ (external - u - np.cross(w, i @ w + h))
            mechanical = float(np.dot(u, h / ji - w))
            dc = (
                model.idle_power_w
                + abs(mechanical) / model.efficiency
                + model.torque_loss_w_per_nm2 * float(u @ u)
            )
            return np.r_[
                qdot,
                wdot,
                u,
                dc,
                dc - mechanical,
                mechanical,
                external @ w,
                _rotate(q, external),
            ]

        k1 = rhs(y)
        k2 = rhs(y + dt * k1 / 2)
        k3 = rhs(y + dt * k2 / 2)
        k4 = rhs(y + dt * k3)
        y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y[:4] /= np.linalg.norm(y[:4])
        if return_path:
            path_times.append(float(dt_sec if index == n - 1 else (index + 1) * dt))
            path_quaternions.append(tuple(float(v) for v in y[:4]))
        if not np.isfinite(y).all():
            raise ValueError("rigid integration left finite applicability")
    result = RigidAttitudeState(
        quaternion_wxyz=tuple(y[:4]),
        omega_body_rad_s=tuple(y[4:7]),
        wheel_momentum_nms=tuple(y[7:10]),
        elapsed_sec=state.elapsed_sec + dt_sec,
        dc_energy_j=y[10],
        heat_j=y[11],
        motor_work_j=y[12],
        external_work_j=y[13],
    )
    return result, {
        "substeps": n,
        "capacity_limited_sec": saturated_sec.tolist(),
        "external_angular_impulse_gcrs_nms": y[14:17].tolist(),
        **(
            {
                "attitude_times_sec": path_times,
                "attitude_quaternions_wxyz": path_quaternions,
            }
            if return_path
            else {}
        ),
        "scope": "standalone ideal 3-wheel rigid dynamics; not mission bus integration",
    }
