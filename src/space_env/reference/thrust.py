"""Declared inertial, constant-thrust model with exact constant-flow mass loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
