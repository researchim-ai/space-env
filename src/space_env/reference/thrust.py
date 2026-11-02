"""Declared inertial, constant-thrust model with exact constant-flow mass loss."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass

import numpy as np

from space_env.core.attitude import _slerp
from space_env.core.propulsion import G0_M_S2, mass_after_thrust
from space_env.reference.frames import _vector


@dataclass(frozen=True)
class FiniteThrust:
    thrust_n: float
    isp_s: float
    dry_mass_kg: float
    direction_gcrs: tuple[float, float, float]

    def __post_init__(self):
        for value in (self.thrust_n, self.isp_s, self.dry_mass_kg):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "thrust, specific impulse and dry mass must be positive"
                )
        direction = _vector(self.direction_gcrs, 3)
        if not math.isclose(
            float(np.linalg.norm(direction)), 1.0, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("thrust direction must be a unit GCRS vector")
        object.__setattr__(self, "direction_gcrs", tuple(float(v) for v in direction))

    def fuel_seconds(self, mass_kg):
        return (mass_kg - self.dry_mass_kg) * self.isp_s * G0_M_S2 / self.thrust_n

    def mass(self, initial_mass_kg, elapsed_sec):
        return mass_after_thrust(
            initial_mass_kg, elapsed_sec, self.thrust_n, self.isp_s, self.dry_mass_kg
        )[0]

    def direction_at(self, elapsed_sec):
        return np.asarray(self.direction_gcrs)

    def manifest(self):
        return {
            **asdict(self),
            "frame": "GCRS",
            "attitude": "perfect inertial pointing; no ADCS dynamics",
            "mass_model": "constant F/(Isp*g0), exact analytic mass; core.propulsion helper",
        }


def _rotate(q, vector):
    q, vector = np.asarray(q), np.asarray(vector)
    return vector + 2 * np.cross(q[1:], np.cross(q[1:], vector) + q[0] * vector)


@dataclass(frozen=True)
class RotatingThrust(FiniteThrust):
    """Sealed piecewise-SLERP body orientation path, not inertial frozen thrust."""

    direction_body: tuple[float, float, float]
    attitude_times_sec: tuple[float, ...]
    attitude_quaternions_wxyz: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self):
        super().__post_init__()
        axis = _vector(self.direction_body, 3)
        if not math.isclose(np.linalg.norm(axis), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("rotating thrust requires unit BODY axis")
        times = np.asarray(self.attitude_times_sec, dtype=float)
        q = np.asarray(self.attitude_quaternions_wxyz, dtype=float)
        if (
            times.ndim != 1
            or not 2 <= len(times) <= 100001
            or times[0] != 0
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or (
                len(times) > 1
                and np.min(np.diff(times)) > 0
                and times[-1] / np.min(np.diff(times)) > 100000.001
            )
            or q.shape != (len(times), 4)
            or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9, rtol=0)
        ):
            raise ValueError("invalid sealed attitude path")
        if not np.allclose(
            _rotate(q[0], axis), self.direction_gcrs, atol=1e-12, rtol=0
        ):
            raise ValueError("initial inertial direction differs from attitude path")
        object.__setattr__(self, "direction_body", tuple(float(v) for v in axis))
        object.__setattr__(self, "attitude_times_sec", tuple(float(v) for v in times))
        object.__setattr__(
            self,
            "attitude_quaternions_wxyz",
            tuple(tuple(float(v) for v in row) for row in q),
        )

    def direction_at(self, elapsed_sec):
        if (
            not math.isfinite(elapsed_sec)
            or not -1e-9 <= elapsed_sec <= self.attitude_times_sec[-1] + 1e-9
        ):
            raise ValueError("time outside sealed thrust attitude path")
        t = float(np.clip(elapsed_sec, 0, self.attitude_times_sec[-1]))
        index = min(
            len(self.attitude_times_sec) - 2,
            max(0, bisect_right(self.attitude_times_sec, t) - 1),
        )
        left, right = self.attitude_times_sec[index : index + 2]
        q = _slerp(
            self.attitude_quaternions_wxyz[index],
            self.attitude_quaternions_wxyz[index + 1],
            (t - left) / (right - left),
        )
        return _rotate(q, self.direction_body)

    def impulse_vector(self):
        # Four-point Gauss quadrature per smooth SLERP interval.
        nodes, weights = np.polynomial.legendre.leggauss(4)
        result = np.zeros(3)
        for left, right in zip(
            self.attitude_times_sec[:-1], self.attitude_times_sec[1:]
        ):
            half = (right - left) / 2
            for node, weight in zip(nodes, weights):
                result += (
                    half
                    * weight
                    * self.direction_at((left + right) / 2 + half * node)
                    * self.thrust_n
                )
        return result

    def manifest(self):
        return {
            **super().manifest(),
            "attitude": "RK4 rigid-body knots with piecewise SLERP; deterministic nominal attitude only",
            "direction_model": "space-env.rotating-body-thrust/1",
        }


@dataclass(frozen=True)
class BalancedCoupleThrust(FiniteThrust):
    """Ideal opposed nozzle pairs: scalar nozzle thrust consumes mass, net force zero."""

    torque_body_nm: tuple[float, float, float]
    lever_arm_m: tuple[float, float, float]

    def __post_init__(self):
        for value in (self.thrust_n, self.isp_s, self.dry_mass_kg):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("nozzle thrust, Isp and dry mass must be positive")
        torque, lever = _vector(self.torque_body_nm, 3), _vector(self.lever_arm_m, 3)
        if np.any(lever <= 0) or not np.array_equal(
            _vector(self.direction_gcrs, 3), np.zeros(3)
        ):
            raise ValueError(
                "balanced couples require positive arms and zero net direction"
            )
        if not math.isclose(
            float(np.sum(np.abs(torque) / lever)),
            self.thrust_n,
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise ValueError("scalar nozzle thrust differs from pair torque/arm")
        for name, value in (
            ("torque_body_nm", torque),
            ("lever_arm_m", lever),
            ("direction_gcrs", np.zeros(3)),
        ):
            object.__setattr__(self, name, tuple(float(v) for v in value))

    def direction_at(self, elapsed_sec):
        return np.zeros(3)

    def manifest(self):
        return {
            **super().manifest(),
            "direction_model": "space-env.balanced-rcs-couples/1",
            "attitude": "body-fixed ideal balanced pairs; external moment, zero net translation force",
            "scalar_thrust_contract": "sum of nozzle thrust magnitudes; not net force",
        }
