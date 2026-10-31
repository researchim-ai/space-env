"""Independent integration-family check of the declared finite-burn equations."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from space_env.core.constants import MU_EARTH
from space_env.core.propulsion import G0_M_S2
from space_env.reference import (
    EOPSeries,
    Epoch,
    FiniteThrust,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)


def finite_burn_report() -> dict:
    duration, mass, thrust, isp = 50.0, 100.0, 2.0, 300.0
    direction = np.array([0.0, 1.0, 0.0])
    context, epoch = (
        FrameContext(EOPSeries.packaged()),
        Epoch.utc("2026-07-18T12:00:00Z"),
    )
    initial = np.array([7e6, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / 7e6), 0.0])
    ours = propagate(
        ReferenceState(epoch, initial[:3], initial[3:]),
        duration,
        context,
        model=ForceModel(mass_kg=mass),
        sample_sec=1.0,
        max_step_sec=5.0,
        thrust=FiniteThrust(thrust, isp, 90.0, tuple(direction)),
    )

    # No production acceleration or mass helper in the comparison RHS.
    def rhs(clock, state):
        current_mass = mass - thrust / (isp * G0_M_S2) * clock
        return np.concatenate(
            (
                state[3:],
                -MU_EARTH * state[:3] / np.linalg.norm(state[:3]) ** 3
                + thrust / current_mass * direction,
            )
        )

    other = solve_ivp(
        rhs,
        (0.0, duration),
        initial,
        method="RK45",
        t_eval=ours["times_sec"],
        rtol=1e-12,
        atol=[1e-6] * 3 + [1e-9] * 3,
        max_step=1.0,
    )
    if not other.success:
        raise RuntimeError(other.message)
    difference = np.asarray(ours["states_si"]) - other.y.T
    position = float(np.max(np.linalg.norm(difference[:, :3], axis=1)))
    velocity = float(np.max(np.linalg.norm(difference[:, 3:], axis=1)))
    mass_error = abs(
        ours["masses_kg"][-1] - (mass - thrust / (isp * G0_M_S2) * duration)
    )
    return {
        "schema": "space-env.finite-burn-comparison/1",
        "duration_sec": duration,
        "position_error_m": position,
        "position_tolerance_m": 0.01,
        "velocity_error_m_s": velocity,
        "velocity_tolerance_m_s": 1e-5,
        "mass_error_kg": mass_error,
        "mass_tolerance_kg": 1e-10,
        "passed": position <= 0.01 and velocity <= 1e-5 and mass_error <= 1e-10,
        "comparison": "independent RHS with RK45 versus production DOP853 and mass helper",
        "shared_inputs": [
            "mu",
            "g0",
            "initial state",
            "constant inertial thrust and specific impulse",
        ],
        "production_manifest": ours["manifest"],
        "not_qualified": [
            "ADCS and pointing error",
            "thrust calibration",
            "propulsion power/thermal",
            "independent perturbation/ephemeris truth",
            "navigation covariance",
        ],
    }
