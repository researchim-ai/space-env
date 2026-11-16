"""Matched central-gravity trajectory exchange. No implicit unit/frame conversion."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import MU_EARTH
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest


class Trajectory(StrictModel):
    case_id: str = Field(min_length=1, max_length=128)
    case_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frame: Literal["GCRS"]
    epoch_utc: Literal["2026-07-18T12:00:00Z"]
    time_basis: Literal["elapsed_SI_seconds"]
    units: Literal["m,m/s"]
    times_sec: tuple[FiniteScalar, ...] = Field(min_length=2, max_length=100001)
    states_si: tuple[
        tuple[
            FiniteScalar,
            FiniteScalar,
            FiniteScalar,
            FiniteScalar,
            FiniteScalar,
            FiniteScalar,
        ],
        ...,
    ] = Field(min_length=2, max_length=100001)

    @model_validator(mode="after")
    def matching_lengths(self):
        if len(self.times_sec) != len(self.states_si):
            raise ValueError("trajectory lengths differ")
        return self


class ExternalTrajectories(StrictModel):
    schema_id: Literal["space-env.external-trajectories/1"] = Field(alias="schema")
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    engine: Literal["Basilisk", "BSK-RL", "Orekit", "Tudat"]
    engine_version: str = Field(min_length=1, max_length=128)
    adapter_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver_settings: dict = Field(min_length=1, max_length=32)
    trajectories: tuple[Trajectory, ...] = Field(min_length=1, max_length=32)


def kepler_state(a, eccentricity, inclination, seconds):
    """Independent elliptic Kepler equation oracle; constants are shared inputs."""
    mean = math.sqrt(MU_EARTH / a**3) * seconds
    eccentric = mean
    for _ in range(50):
        correction = (eccentric - eccentricity * math.sin(eccentric) - mean) / (
            1 - eccentricity * math.cos(eccentric)
        )
        eccentric -= correction
        if abs(correction) < 1e-14:
            break
    else:
        raise ValueError("Kepler oracle did not converge")
    c, s = math.cos(eccentric), math.sin(eccentric)
    b = math.sqrt(1 - eccentricity**2)
    rate = math.sqrt(MU_EARTH / a**3) / (1 - eccentricity * c)
    return [
        a * (c - eccentricity),
        a * b * s * math.cos(inclination),
        a * b * s * math.sin(inclination),
        -a * s * rate,
        a * b * c * rate * math.cos(inclination),
        a * b * c * rate * math.sin(inclination),
    ]


def trajectory_manifest(profile="full"):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown benchmark profile")
    horizon = 3600 if profile == "smoke" else 21600
    cases = []
    for name, a, e, inc in [
        ("circular-leo", 7e6, 0.0, 0.0),
        ("eccentric-leo", 8e6, 0.1, 0.7),
        ("eccentric-meo", 26560000.0, 0.6, 1.0),
    ]:
        cases.append(
            {
                "id": name,
                "semimajor_axis_m": a,
                "eccentricity": e,
                "inclination_rad": inc,
                "initial_state_si": kepler_state(a, e, inc, 0.0),
                "mu_m3_s2": MU_EARTH,
                "forces": ["point_mass_earth"],
                "frame": "GCRS",
                "epoch_utc": "2026-07-18T12:00:00Z",
                "time_basis": "elapsed_SI_seconds",
                "units": "m,m/s",
                "times_sec": [float(t) for t in range(0, horizon + 1, 300)],
                "position_tolerance_m": 0.01,
                "velocity_tolerance_m_s": 1e-5,
            }
        )
    return {
        "schema": "space-env.trajectory-manifest/1",
        "profile": profile,
        "cases": cases,
        "scope": "Central Newtonian gravity only, shared MU; no perturbed orbit, mission or real-flight superiority claim.",
    }


def compare_states(case, states):
    expected = np.asarray(
        [
            kepler_state(
                case["semimajor_axis_m"],
                case["eccentricity"],
                case["inclination_rad"],
                t,
            )
            for t in case["times_sec"]
        ]
    )
    measured = np.asarray(states, dtype=float)
    if measured.shape != expected.shape or not np.isfinite(measured).all():
        raise ValueError("invalid trajectory shape or nonfinite states")
    delta = measured - expected
    position = float(np.max(np.linalg.norm(delta[:, :3], axis=1)))
    velocity = float(np.max(np.linalg.norm(delta[:, 3:], axis=1)))
    return {
        "case": case["id"],
        "max_position_error_m": position,
        "max_velocity_error_m_s": velocity,
        "passed": position <= case["position_tolerance_m"]
        and velocity <= case["velocity_tolerance_m_s"],
    }


def native_trajectories(manifest):
    if manifest != trajectory_manifest(manifest.get("profile")):
        raise ValueError("trajectory manifest differs from frozen protocol")
    context = FrameContext(EOPSeries.packaged())
    rows, trajectories = [], []
    for case in manifest["cases"]:
        state = case["initial_state_si"]
        result = propagate(
            ReferenceState(
                Epoch.utc(case["epoch_utc"]), np.array(state[:3]), np.array(state[3:])
            ),
            case["times_sec"][-1],
            context,
            sample_sec=300.0,
        )
        if result["times_sec"] != case["times_sec"]:
            raise ValueError("native time grid mismatch")
        rows.append(compare_states(case, result["states_si"]))
        trajectories.append(
            {
                "case_id": case["id"],
                "case_sha256": digest(case),
                **{
                    k: case[k]
                    for k in ("frame", "epoch_utc", "time_basis", "units", "times_sec")
                },
                "states_si": result["states_si"],
                "native_manifest": result["manifest"],
            }
        )
    return {
        "rows": rows,
        "trajectories": trajectories,
        "passed": all(r["passed"] for r in rows),
    }


def compare_external(manifest, value):
    """Validate declared provenance and exact matching; not authenticate engine execution."""
    if manifest != trajectory_manifest(manifest.get("profile")):
        raise ValueError("trajectory manifest differs from frozen protocol")
    submission = ExternalTrajectories.model_validate(value)
    if submission.manifest_sha256 != digest(manifest):
        raise ValueError("external manifest identity mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    ids = [t.case_id for t in submission.trajectories]
    if len(set(ids)) != len(ids) or set(ids) != set(cases):
        raise ValueError("external case matrix incomplete or duplicated")
    rows = []
    for trajectory in submission.trajectories:
        case = cases[trajectory.case_id]
        if (
            trajectory.case_sha256 != digest(case)
            or list(trajectory.times_sec) != case["times_sec"]
        ):
            raise ValueError("external case identity or grid mismatch")
        initial = np.asarray(trajectory.states_si[0]) - case["initial_state_si"]
        if np.linalg.norm(initial[:3]) > 1e-6 or np.linalg.norm(initial[3:]) > 1e-9:
            raise ValueError("external initial state mismatch")
        rows.append(compare_states(case, trajectory.states_si))
    return {
        "engine": submission.engine,
        "engine_version": submission.engine_version,
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
        "submission_sha256": digest(value),
        "provenance": "adapter identity is submitted metadata, not independently attested execution",
        "scope": "each engine against the same analytic oracle; accuracy agreement is not overall superiority",
    }
