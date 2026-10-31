"""Optional Cartesian variational equations along a sealed nominal GCRS arc.

This is covariance prediction, not orbit determination or a calibrated uncertainty
model. Nominal interpolation and finite-difference force partials remain explicit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import solve_ivp

from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.orbit import ForceModel, acceleration
from space_env.reference.thrust import FiniteThrust
from space_env.simulation.snapshot import digest

Scalar = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Matrix = tuple[tuple[Scalar, ...], ...]
_ZERO3 = ((0.0, 0.0, 0.0),) * 3


def psd_matrix(value, dimension):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite {dimension}x{dimension} matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance/noise matrix must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(1.0, float(np.linalg.norm(matrix, 2)))
    if np.linalg.eigvalsh(matrix).min() < -1e-12 * scale:
        raise ValueError("covariance/noise matrix must be positive semidefinite")
    return matrix


class NavigationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    frame: Literal["GCRS"] = "GCRS"
    initial_covariance_si: Matrix
    acceleration_psd_m2_s3: Matrix = _ZERO3
    impulse_velocity_cov_m2_s2: Matrix = _ZERO3
    rtol: float = Field(default=1e-10, ge=1e-13, le=1e-6, strict=True)
    matrix_atol: float = Field(default=1e-11, gt=0, le=1e-4, strict=True)
    max_step_sec: float = Field(default=30.0, gt=0, le=120, strict=True)
    max_evaluations: int = Field(default=200000, ge=100, le=1000000, strict=True)
    position_difference_step_m: float = Field(default=1.0, gt=0, le=100, strict=True)
    velocity_difference_step_m_s: float = Field(default=0.001, gt=0, le=1, strict=True)

    @model_validator(mode="after")
    def matrices(self):
        psd_matrix(self.initial_covariance_si, 6)
        psd_matrix(self.acceleration_psd_m2_s3, 3)
        psd_matrix(self.impulse_velocity_cov_m2_s2, 3)
        return self


def state_jacobian(
    state,
    epoch,
    context,
    model: ForceModel,
    *,
    position_step_m=1.0,
    velocity_step_m_s=0.001,
):
    """df/dx in Cartesian GCRS SI. Mass and inertial pointing are deterministic."""
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("linearization requires finite Cartesian r/v")
    for value in (position_step_m, velocity_step_m_s):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("finite-difference steps must be positive")
    radius = np.linalg.norm(state[:3])
    if radius <= R_EARTH_EQUATOR:
        raise ValueError("Earth intersection is outside the covariance profile")
    result = np.zeros((6, 6))
    result[:3, 3:] = np.eye(3)
    if not any((model.j2, model.drag, model.srp, model.third_body)):
        r = state[:3]
        result[3:, :3] = MU_EARTH * (
            3 * np.outer(r, r) / radius**5 - np.eye(3) / radius**3
        )
    else:
        # Differentiate the same declared forces, including rotating-atmosphere
        # velocity. Two-sided partials; no unannounced automatic differentiation.
        for column in range(6):
            step = position_step_m if column < 3 else velocity_step_m_s
            delta = np.zeros(6)
            delta[column] = step
            result[3:, column] = (
                acceleration(state + delta, epoch, context, model)
                - acceleration(state - delta, epoch, context, model)
            ) / (2 * step)
    return result


class CovarianceArc:
    """Dense Phi and integrated white-acceleration diffusion on one smooth arc.

    P(t)=Phi(t) P0 Phi(t)^T + W(t); Wdot=A W + W A^T + G Q G^T.
    A(t) follows the nominal Hermite state. Integration never crosses an impulse.
    """

    def __init__(
        self,
        ephemeris,
        settings: NavigationSettings,
        *,
        model=None,
        thrust=None,
        initial_covariance=None,
        origin_stm=None,
    ):
        if (
            not isinstance(ephemeris, ReferenceEphemeris)
            or getattr(ephemeris, "_manifest", {}).get("profile") != "gcrs-reference/1"
        ):
            raise ValueError("covariance integration requires a sealed smooth GCRS arc")
        settings = NavigationSettings.model_validate(settings)
        if model is not None and not isinstance(model, ForceModel):
            raise ValueError("covariance integration requires explicit ForceModel")
        if thrust is not None and not isinstance(thrust, FiniteThrust):
            raise ValueError("covariance integration requires explicit FiniteThrust")
        self.duration_sec = ephemeris.duration_sec
        self.settings = settings
        self.initial_covariance = psd_matrix(
            settings.initial_covariance_si
            if initial_covariance is None
            else initial_covariance,
            6,
        ).copy()
        self.origin_stm = (
            np.eye(6)
            if origin_stm is None
            else np.asarray(origin_stm, dtype=float).copy()
        )
        if self.origin_stm.shape != (6, 6) or not np.isfinite(self.origin_stm).all():
            raise ValueError("origin STM must be a finite 6x6 matrix")
        declared_forces = ephemeris._manifest.get("forces")
        if declared_forces is None:
            raise ValueError("nominal arc has no declared force model")
        model = ForceModel(**declared_forces) if model is None else model
        if asdict(model) != declared_forces:
            raise ValueError("covariance force model differs from nominal arc")
        declared_thrust = ephemeris._manifest.get("thrust")
        if (declared_thrust is None) != (thrust is None):
            raise ValueError("covariance thrust model differs from nominal arc")
        if thrust is not None and (
            declared_thrust.get("frame") != "GCRS"
            or any(
                declared_thrust.get(key) != getattr(thrust, key)
                for key in ("thrust_n", "isp_s", "dry_mass_kg")
            )
            or list(declared_thrust.get("direction_gcrs", []))
            != list(thrust.direction_gcrs)
        ):
            raise ValueError("covariance thrust model differs from nominal arc")
        noise = np.zeros((6, 6))
        noise[3:, 3:] = psd_matrix(settings.acceleration_psd_m2_s3, 3)
        max_step = min(settings.max_step_sec, ephemeris.max_interval_sec)
        if math.ceil(self.duration_sec / max_step) > 100000:
            raise ValueError("covariance integration exceeds 100000 step budget")

        evaluations = 0

        def derivative(clock, combined):
            nonlocal evaluations
            evaluations += 1
            if evaluations > settings.max_evaluations:
                raise ValueError("covariance integration evaluation budget exhausted")
            state = ephemeris.state(float(clock))
            current_model = (
                replace(model, mass_kg=thrust.mass(model.mass_kg, clock))
                if thrust
                else model
            )
            jac = state_jacobian(
                state,
                ephemeris.epoch.shifted(float(clock)),
                ephemeris.context,
                current_model,
                position_step_m=settings.position_difference_step_m,
                velocity_step_m_s=settings.velocity_difference_step_m_s,
            )
            phi, diffusion = combined[:36].reshape(6, 6), combined[36:].reshape(6, 6)
            return np.concatenate(
                (
                    (jac @ phi).ravel(),
                    (jac @ diffusion + diffusion @ jac.T + noise).ravel(),
                )
            )

        initial = np.concatenate((np.eye(6).ravel(), np.zeros(36)))
        result = solve_ivp(
            derivative,
            (0.0, self.duration_sec),
            initial,
            method="DOP853",
            rtol=settings.rtol,
            atol=settings.matrix_atol,
            max_step=max_step,
            dense_output=True,
        )
        if not result.success or not np.isfinite(result.y).all():
            raise RuntimeError(f"covariance integration failed: {result.message}")
        self._solution = result.sol
        self.manifest = {
            "schema": "space-env.cartesian-covariance/1",
            "frame": "GCRS",
            "state_order": ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"],
            "settings": settings.model_dump(mode="json"),
            "nominal_ephemeris_sha256": ephemeris.sha256,
            "integrator": "DOP853 dense variational equations on nominal Hermite arc",
            "linearization": "analytic central gravity; two-sided finite-difference full perturbed forces",
            "evaluations": result.nfev,
            "limits": [
                "small-error Cartesian linearization",
                "deterministic mass and perfect inertial pointing",
                "constant GCRS white acceleration PSD; independent additive impulse velocity noise",
                "no navigation measurements, filter update, mass/attitude covariance or cross-actor covariance",
            ],
        }
        self.sha256 = digest(
            {
                "manifest": self.manifest,
                "initial_covariance_si": self.initial_covariance.tolist(),
                "origin_stm": self.origin_stm.tolist(),
            }
        )
        self.covariance(self.duration_sec)

    def _value(self, clock):
        if (
            isinstance(clock, bool)
            or not math.isfinite(clock)
            or not 0 <= clock <= self.duration_sec
        ):
            raise ValueError("covariance query outside coverage")
        value = self._solution(clock)
        if not np.isfinite(value).all():
            raise RuntimeError("nonfinite covariance integration output")
        return value

    def transition(self, clock):
        return self._value(clock)[:36].reshape(6, 6).copy()

    def stm(self, clock):
        return self.transition(clock) @ self.origin_stm

    def diffusion(self, clock):
        return psd_matrix(self._value(clock)[36:].reshape(6, 6), 6).copy()

    def covariance(self, clock):
        transition = self.transition(clock)
        return psd_matrix(
            transition @ self.initial_covariance @ transition.T + self.diffusion(clock),
            6,
        )
