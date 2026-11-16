"""Noisy CW relative navigation, impulsive inspection/hold and safe retreat tasks.

Circular-chief linearization only; no docking/contact or nonlinear flight claim.
Safety witnesses use simulator truth and never enter actor observations.
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.relative import cw_stm
from space_env.industrial.safety import check_cw_interval
from space_env.industrial.session import DecisionSession
from space_env.reference.covariance import psd_matrix
from space_env.runtime.schema import FiniteScalar, StrictModel

Vec3 = tuple[FiniteScalar, FiniteScalar, FiniteScalar]
Vec6 = tuple[
    FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar
]


class ProximitySpec(StrictModel):
    id: str = Field(default="relative-inspection", min_length=1, max_length=128)
    lineage: str = Field(default="cw-inspection/v1", min_length=1, max_length=128)
    mode: Literal["hold", "inspection", "retreat"] = "inspection"
    mean_motion_rad_s: float = Field(default=0.0011, ge=0, le=0.01, strict=True)
    chief_radius_m: float = Field(
        default=7000000.0, ge=6500000, le=50000000, strict=True
    )
    initial_truth_si: Vec6 = (100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    initial_estimate_si: Vec6 = (102.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    initial_covariance_si: tuple[tuple[FiniteScalar, ...], ...] = tuple(
        tuple(float(4 if i == j and i < 3 else 0.01 if i == j else 0) for j in range(6))
        for i in range(6)
    )
    waypoint_m: Vec3 = (100.0, 0.0, 0.0)
    tolerance_m: float = Field(default=10.0, gt=0, le=1000, strict=True)
    maximum_speed_m_s: float = Field(default=1.0, gt=0, le=10, strict=True)
    required_observations: int = Field(default=3, ge=1, le=128, strict=True)
    keepout_m: float = Field(default=20.0, gt=0, le=10000, strict=True)
    retreat_radius_m: float = Field(default=150.0, gt=0, le=10000, strict=True)
    step_sec: float = Field(default=5.0, ge=0.1, le=30, strict=True)
    horizon_steps: int = Field(default=24, ge=1, le=256, strict=True)
    max_impulse_m_s: float = Field(default=0.5, gt=0, le=10, strict=True)
    delta_v_budget_m_s: float = Field(default=5.0, gt=0, le=100, strict=True)
    position_noise_m: float = Field(default=1.0, gt=0, le=100, strict=True)
    acceleration_psd_m2_s3: float = Field(default=0.000001, ge=0, le=1, strict=True)
    dropout_probability: float = Field(default=0.0, ge=0, le=1, strict=True)
    measurement_latency_steps: int = Field(default=0, ge=0, le=32, strict=True)
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)

    @model_validator(mode="after")
    def physical(self):
        psd_matrix(self.initial_covariance_si, 6)
        if (
            self.step_sec * self.horizon_steps > 3600
            or max(
                np.linalg.norm(self.initial_truth_si[3:]),
                np.linalg.norm(self.initial_estimate_si[3:]),
            )
            > 100
        ):
            raise ValueError("relative profile exceeds one hour or 100 m/s envelope")
        if (
            self.keepout_m >= self.retreat_radius_m
            or np.linalg.norm(self.initial_truth_si[:3]) <= self.keepout_m
        ):
            raise ValueError("initial separation and retreat must exceed keepout")
        if np.linalg.norm(self.waypoint_m) - self.tolerance_m <= self.keepout_m:
            raise ValueError("inspection region intersects keepout")
        if (
            max(
                np.linalg.norm(self.initial_truth_si[:3]),
                np.linalg.norm(self.initial_estimate_si[:3]),
                self.retreat_radius_m,
                np.linalg.norm(self.waypoint_m),
            )
            > 0.01 * self.chief_radius_m
        ):
            raise ValueError(
                "relative configuration exceeds one-percent circular-chief envelope"
            )
        return self


class ProximityAction(StrictModel):
    kind: Literal["coast", "impulse", "inspect", "abort"]
    delta_v_rtn_m_s: Vec3 = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def impulse(self):
        if self.kind not in {"impulse", "abort"} and any(self.delta_v_rtn_m_s):
            raise ValueError("only impulse or abort accepts delta-v")
        if self.kind == "impulse" and not any(self.delta_v_rtn_m_s):
            raise ValueError("nonzero impulse required")
        return self


class ProximitySession(DecisionSession):
    kind = "proximity"

    def __init__(self, configuration=None):
        self.spec = ProximitySpec.model_validate(
            {} if configuration is None else configuration
        )
        self._start(self.spec.model_dump(mode="json"))
        self._truth = np.array(self.spec.initial_truth_si)
        self._estimate = np.array(self.spec.initial_estimate_si)
        self._cov = psd_matrix(self.spec.initial_covariance_si, 6)
        self._rng = np.random.default_rng(self.spec.seed)
        self._used_dv, self._inspections = 0.0, 0
        self._aborting = self.spec.mode == "retreat"
        self._safe = True
        self._retreated = False
        self._success = False
        self._safety = []
        self._packets = []
        self._age_steps = None
        self._last_nis = None
        self._estimates = [(self._estimate.copy(), self._cov.copy())]
        self._controls, self._measurements = [], {}

    def observe(self):
        return {
            "schema": "space-env.proximity-observation/1",
            "revision": self.revision,
            "time_sec": self.revision * self.spec.step_sec,
            "frame": "chief-rotating-RTN",
            "state_source": "linear-relative-Kalman-estimate",
            "estimate_si": self._estimate.tolist(),
            "covariance_si": self._cov.tolist(),
            "measurement_age_steps": self._age_steps,
            "last_nis": self._last_nis,
            "delta_v_remaining_m_s": self.spec.delta_v_budget_m_s - self._used_dv,
            "aborting": self._aborting,
            "task": {
                k: copy.deepcopy(self._configuration[k])
                for k in (
                    "mode",
                    "mean_motion_rad_s",
                    "step_sec",
                    "waypoint_m",
                    "tolerance_m",
                    "maximum_speed_m_s",
                    "required_observations",
                    "keepout_m",
                    "retreat_radius_m",
                    "max_impulse_m_s",
                    "horizon_steps",
                )
            },
            "inspection_acknowledgements": self._inspections,
        }

    def _filter(self, index):
        # Replay only the bounded relative filter window, with frozen acquisition
        # packets. No delayed packet is updated against the wrong epoch.
        start = max(0, index - self.spec.measurement_latency_steps - 1)
        x, p = (v.copy() for v in self._estimates[start])
        f = cw_stm(self.spec.mean_motion_rad_s, self.spec.step_sec)
        t, q = self.spec.step_sec, self.spec.acceleration_psd_m2_s3
        noise = q * np.block(
            [
                [np.eye(3) * t**3 / 3, np.eye(3) * t**2 / 2],
                [np.eye(3) * t**2 / 2, np.eye(3) * t],
            ]
        )
        for k in range(start + 1, index + 1):
            x[3:] += self._controls[k - 1]
            x, p = f @ x, f @ p @ f.T + noise
            if k in self._measurements:
                innovation = self._measurements[k] - x[:3]
                s = p[:3, :3] + np.eye(3) * self.spec.position_noise_m**2
                gain = np.linalg.solve(s, p[:3, :]).T
                self._last_nis = float(innovation @ np.linalg.solve(s, innovation))
                x += gain @ innovation
                a = np.eye(6) - gain @ np.eye(6)[:3]
                p = (
                    a @ p @ a.T
                    + gain @ (np.eye(3) * self.spec.position_noise_m**2) @ gain.T
                )
            p = (p + p.T) / 2
            if k < len(self._estimates):
                self._estimates[k] = (x.copy(), p.copy())
            else:
                self._estimates.append((x.copy(), p.copy()))
        self._estimate, self._cov = x, p

    def _apply(self, value):
        action = ProximityAction.model_validate(value)
        dv = np.array(action.delta_v_rtn_m_s)
        amount = float(np.linalg.norm(dv))
        if (
            amount > self.spec.max_impulse_m_s + 1e-12
            or self._used_dv + amount > self.spec.delta_v_budget_m_s + 1e-12
        ):
            raise ValueError("impulse or total delta-v budget exceeded")
        if action.kind == "inspect" and self._aborting:
            raise ValueError("inspection forbidden after abort")
        self._aborting |= action.kind == "abort"
        self._used_dv += amount
        self._controls.append(dv.copy())
        self._truth[3:] += dv
        witness = check_cw_interval(
            self._truth,
            self.spec.mean_motion_rad_s,
            self.spec.step_sec,
            self.spec.keepout_m,
        )
        self._safety.append(witness)
        self._safe &= witness["status"] == "safe"
        self._truth = (
            cw_stm(self.spec.mean_motion_rad_s, self.spec.step_sec) @ self._truth
        )
        index = self.revision + 1
        if self._rng.random() >= self.spec.dropout_probability:
            self._packets.append(
                (
                    index,
                    index + self.spec.measurement_latency_steps,
                    self._truth[:3]
                    + self._rng.normal(0, self.spec.position_noise_m, 3),
                )
            )
        for acquired, available, measurement in self._packets:
            if available <= index:
                self._measurements[acquired] = measurement
        self._packets = [(a, b, z) for a, b, z in self._packets if b > index]
        self._filter(index)
        self._age_steps = (
            index - max(self._measurements) if self._measurements else None
        )
        valid = (
            np.linalg.norm(self._truth[:3] - self.spec.waypoint_m)
            <= self.spec.tolerance_m
            and np.linalg.norm(self._truth[3:]) <= self.spec.maximum_speed_m_s
        )
        gained = (
            self._safe
            and valid
            and (
                action.kind == "inspect"
                if self.spec.mode == "inspection"
                else not self._aborting
            )
        )
        self._inspections += int(gained)
        self._retreated = (
            self._aborting
            and self._safe
            and np.linalg.norm(self._truth[:3]) >= self.spec.retreat_radius_m
            and float(self._truth[:3] @ self._truth[3:]) >= 0
        )
        self._success = (
            self._safe
            and not self._aborting
            and self._inspections >= self.spec.required_observations
        )
        if self.spec.mode == "retreat":
            self._success = bool(self._retreated)
        envelope = np.linalg.norm(self._truth[:3]) <= 0.01 * self.spec.chief_radius_m
        self._success = bool(self._success and envelope)
        self._done = bool(
            not self._safe
            or not envelope
            or self._success
            or self._retreated
            or index >= self.spec.horizon_steps
        )
        return {
            "reward": float(gained) - amount - float(not self._safe),
            "terminated": self._done,
            "elapsed_sec": self.spec.step_sec,
            "feedback": {"accepted": True, "code": "executed"},
        }

    def metrics(self):
        return {
            "success": bool(self._success),
            "safe_retreat": bool(self._retreated),
            "safe_intervals": bool(self._safe),
            "inspection_acknowledgements": self._inspections,
            "delta_v_used_m_s": self._used_dv,
            "terminal": self._done,
            "full_docking_model": False,
            "mission_mode": self.spec.mode,
        }

    def _witness(self):
        return {
            "truth": self._truth.tolist(),
            "observation": self.observe(),
            "metrics": self.metrics(),
            "intervals": copy.deepcopy(self._safety),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
            "packets": [(a, b, z.tolist()) for a, b, z in self._packets],
            "measurements": {str(k): v.tolist() for k, v in self._measurements.items()},
        }


def proximity_policy(observation):
    """Public estimated-state PD policy; no hidden safety checker or truth query."""
    x = np.array(observation["estimate_si"])
    task = observation["task"]
    # A planning heuristic over the estimated trajectory, not a probability or
    # truth-based safety certificate. Position sigma is only a heuristic margin.
    uncertainty = 3 * np.sqrt(
        max(0.0, np.linalg.eigvalsh(np.array(observation["covariance_si"])[:3, :3])[-1])
    )
    horizon = min(
        30.0, (task["horizon_steps"] - observation["revision"]) * task["step_sec"]
    )
    predicted = check_cw_interval(
        x,
        task["mean_motion_rad_s"],
        horizon,
        task["keepout_m"] + uncertainty,
        max_evaluations=257,
    )
    if not observation["aborting"] and predicted["status"] != "safe":
        axis = np.cross(x[3:], [0.0, 0.0, 1.0])
        if np.linalg.norm(axis) < 1e-9:
            axis = x[:3]
        axis = axis / max(1e-9, np.linalg.norm(axis))
        limit = min(task["max_impulse_m_s"], observation["delta_v_remaining_m_s"])
        return {"kind": "abort", "delta_v_rtn_m_s": (axis * limit).tolist()}
    goal = np.array(task["waypoint_m"])
    if observation["aborting"]:
        direction = x[:3] / max(1e-9, np.linalg.norm(x[:3]))
        goal = direction * (task["retreat_radius_m"] + 20)
    target_v = np.clip((goal - x[:3]) * 0.03, -0.4, 0.4)
    dv = target_v - x[3:]
    limit = min(task["max_impulse_m_s"], observation["delta_v_remaining_m_s"])
    norm = np.linalg.norm(dv)
    if norm > 0.05 and limit > 1e-8:
        dv *= min(1.0, limit / norm)
        return {"kind": "impulse", "delta_v_rtn_m_s": dv.tolist()}
    return {"kind": "coast" if observation["aborting"] else "inspect"}
