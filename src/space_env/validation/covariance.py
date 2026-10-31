"""Measured small-error covariance checks against a separate RK45 ensemble RHS."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from space_env.core.constants import MU_EARTH
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.covariance import CovarianceArc, NavigationSettings


def covariance_prediction_report() -> dict:
    duration = 60.0
    initial = np.array([7e6, 0.0, 0.0, 0.0, 7400.0, 1400.0])
    covariance = np.diag([100.0, 200.0, 300.0, 0.01, 0.02, 0.03])
    epoch, context = (
        Epoch.utc("2026-07-18T12:00:00Z"),
        FrameContext(EOPSeries.packaged()),
    )
    result = propagate(
        ReferenceState(epoch, initial[:3], initial[3:]),
        duration,
        context,
        sample_sec=1.0,
        max_step_sec=5.0,
    )
    prediction = CovarianceArc(
        ReferenceEphemeris.from_result(result),
        NavigationSettings(initial_covariance_si=covariance.tolist()),
    )
    deviations = np.concatenate(
        (
            np.sqrt(6) * np.linalg.cholesky(covariance).T,
            -np.sqrt(6) * np.linalg.cholesky(covariance).T,
        )
    )
    points = initial + deviations

    def rhs(_clock, values):
        states = values.reshape(-1, 6)
        radii = np.linalg.norm(states[:, :3], axis=1)
        return np.concatenate(
            (states[:, 3:], -MU_EARTH * states[:, :3] / radii[:, None] ** 3), axis=1
        ).ravel()

    comparison = solve_ivp(
        rhs,
        (0.0, duration),
        points.ravel(),
        method="RK45",
        rtol=1e-12,
        atol=np.tile([1e-6] * 3 + [1e-9] * 3, 12),
        max_step=1.0,
    )
    if not comparison.success:
        raise RuntimeError(comparison.message)
    final = comparison.y[:, -1].reshape(12, 6)
    centered = final - final.mean(axis=0)
    empirical = centered.T @ centered / len(final)
    predicted = prediction.covariance(duration)
    position_relative = float(
        np.linalg.norm(predicted[:3, :3] - empirical[:3, :3])
        / np.linalg.norm(empirical[:3, :3])
    )
    velocity_relative = float(
        np.linalg.norm(predicted[3:, 3:] - empirical[3:, 3:])
        / np.linalg.norm(empirical[3:, 3:])
    )
    return {
        "schema": "space-env.covariance-comparison/1",
        "passed": position_relative < 1e-5 and velocity_relative < 1e-5,
        "duration_sec": duration,
        "position_covariance_relative_error": position_relative,
        "velocity_covariance_relative_error": velocity_relative,
        "relative_tolerance": 1e-5,
        "ensemble": "12 deterministic symmetric sigma points, population covariance; not random Monte Carlo",
        "comparison": "separate vectorized central-gravity RK45 RHS; no production acceleration/STM",
        "shared_inputs": ["mu", "initial state/covariance", "central-gravity model"],
        "nominal_manifest": result["manifest"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "not_qualified": [
            "independent perturbation covariance",
            "long-arc nonlinear uncertainty",
            "navigation measurement/filter update",
            "mass/pointing/atmosphere uncertainty",
            "calibrated process noise",
        ],
    }
