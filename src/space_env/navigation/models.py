"""Synthetic instantaneous GCRS measurements and estimated-state EKF kernels."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp

from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.reference.covariance import psd_matrix, state_jacobian
from space_env.reference.orbit import ForceModel
from space_env.runtime.schema import FiniteScalar, StrictModel

Vec6 = tuple[
    FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar
]


class SensorSpec(StrictModel):
    kind: Literal["cartesian_position", "range_range_rate"]
    frame: Literal["GCRS"] = "GCRS"
    initial_estimate_si: Vec6
    initial_covariance_si: tuple[tuple[FiniteScalar, ...], ...]
    standard_deviation: tuple[FiniteScalar, ...]
    reference_actor: str | None = Field(default=None, min_length=1, max_length=128)
    true_bias: tuple[FiniteScalar, ...] = ()
    cadence_sec: float = Field(default=1.0, ge=0.1, le=3600, strict=True)
    latency_sec: float = Field(default=0.0, ge=0, le=3600, strict=True)
    dropout_probability: float = Field(default=0.0, ge=0, le=1, strict=True)
    acceleration_psd_m2_s3: float = Field(default=0.0, ge=0, le=1, strict=True)
    force_profile: Literal["central", "matched_native"] = "central"
    control_profile: Literal["impulse", "inertial_finite_burn"] = "impulse"
    continuous_power_w: float = Field(default=0.0, ge=0, le=1e6, strict=True)
    minimum_soc: float = Field(default=0.0, ge=0, lt=1, strict=True)
    impulse_velocity_cov_m2_s2: tuple[tuple[FiniteScalar, ...], ...] = (
        (0.0, 0.0, 0.0),
    ) * 3
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)
    max_samples: int = Field(default=512, ge=1, le=512, strict=True)
    nis_gate: float | None = Field(default=None, gt=0, le=1e12, strict=True)

    @model_validator(mode="after")
    def physical(self):
        dimension = 3 if self.kind == "cartesian_position" else 2
        if len(self.standard_deviation) != dimension or any(
            x <= 0 or x > 1e9 for x in self.standard_deviation
        ):
            raise ValueError(
                "positive finite sensor standard deviations required in measurement SI units"
            )
        if self.true_bias and len(self.true_bias) != dimension:
            raise ValueError("bias dimension differs from measurement")
        if (self.kind == "range_range_rate") != (self.reference_actor is not None):
            raise ValueError("range/range-rate requires exactly one known beacon actor")
        psd_matrix(self.initial_covariance_si, 6)
        psd_matrix(self.impulse_velocity_cov_m2_s2, 3)
        if np.linalg.norm(self.initial_estimate_si[:3]) <= R_EARTH_EQUATOR:
            raise ValueError("initial estimate intersects Earth")
        if (
            np.linalg.norm(self.initial_estimate_si[:3]) > 1e9
            or np.linalg.norm(self.initial_estimate_si[3:]) > 1e5
        ):
            raise ValueError("estimate outside bounded Earth-orbit profile")
        return self


def measurement_model(kind, state, reference=None):
    """Return predicted measurement and analytic dh/dx. No clock/light-time terms."""
    x = np.asarray(state, dtype=float)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("finite Cartesian six-state required")
    if kind == "cartesian_position":
        return x[:3].copy(), np.eye(6)[:3]
    if kind != "range_range_rate":
        raise ValueError("unknown measurement kind")
    beacon = np.asarray(reference, dtype=float)
    if beacon.shape != (6,) or not np.all(np.isfinite(beacon)):
        raise ValueError("known finite beacon state required")
    rho, dv = x[:3] - beacon[:3], x[3:] - beacon[3:]
    distance = float(np.linalg.norm(rho))
    if distance < 1.0:
        raise ValueError("range model requires separation >=1 metre")
    u = rho / distance
    rate = float(u @ dv)
    h = np.zeros((2, 6))
    h[0, :3] = u
    h[1, :3] = (dv - u * rate) / distance
    h[1, 3:] = u
    return np.array([distance, rate]), h


def predict(state, covariance, duration, psd=0.0, *, dynamics=None):
    """Propagate estimated mean/covariance; central gravity or trusted declared dynamics."""
    if isinstance(psd, bool) or not np.isfinite(psd) or not 0 <= psd <= 1:
        raise ValueError("finite nonnegative acceleration PSD required")
    if duration < 0 or not np.isfinite(duration) or duration > 3600:
        raise ValueError("prediction duration must be finite and nonnegative")
    x = np.asarray(state, dtype=float)
    p = psd_matrix(covariance, 6)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("finite estimated state required")
    if duration == 0:
        return x.copy(), p.copy()
    q = np.zeros((6, 6))
    q[3:, 3:] = np.eye(3) * psd

    def rhs(t, y):
        current = y[:6]
        r = current[:3]
        radius = np.linalg.norm(r)
        if dynamics is None:
            a = state_jacobian(current, None, None, ForceModel())
            force = -MU_EARTH * r / radius**3
        else:
            force, a = dynamics(t, current)
        covariance = y[6:].reshape(6, 6)
        return np.concatenate(
            (current[3:], force, (a @ covariance + covariance @ a.T + q).ravel())
        )

    result = solve_ivp(
        rhs,
        (0.0, float(duration)),
        np.concatenate((x, p.ravel())),
        rtol=1e-10,
        atol=1e-10,
        max_step=min(5.0, duration),
    )
    if not result.success:
        raise ValueError("navigation propagation failed")
    y = result.y[:, -1]
    return y[:6], psd_matrix(y[6:].reshape(6, 6), 6)


def update(
    state, covariance, value, kind, noise_covariance, reference=None, nis_gate=None
):
    predicted, h = measurement_model(kind, state, reference)
    p = psd_matrix(covariance, 6)
    r = psd_matrix(noise_covariance, len(predicted))
    innovation = np.asarray(value, dtype=float) - predicted
    if innovation.shape != predicted.shape or not np.all(np.isfinite(innovation)):
        raise ValueError("finite measurement vector required")
    s = h @ p @ h.T + r
    try:
        nis = float(innovation @ np.linalg.solve(s, innovation))
        gain = np.linalg.solve(s, h @ p).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation covariance must be nonsingular") from exc
    if not np.isfinite(nis) or not np.all(np.isfinite(gain)):
        raise ValueError("nonfinite Kalman update")
    accepted = nis_gate is None or nis <= nis_gate
    if not accepted:
        return (
            np.array(state, copy=True),
            p.copy(),
            {"nis": nis, "accepted": False, "innovation": innovation.tolist()},
        )
    x = np.asarray(state) + gain @ innovation
    identity = np.eye(6) - gain @ h
    posterior = psd_matrix(identity @ p @ identity.T + gain @ r @ gain.T, 6)
    return (
        x,
        posterior,
        {"nis": nis, "accepted": True, "innovation": innovation.tolist()},
    )
