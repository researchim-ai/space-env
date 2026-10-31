"""Verify frozen external-formula arcs; keep shared-input limitations visible."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.simulation.snapshot import digest

# Predeclared engineering comparison budget, not a calibrated mission accuracy.
POSITION_TOLERANCE_M = 10.0
VELOCITY_TOLERANCE_M_S = 0.01


def verify_orbit_fixture(payload: dict) -> dict:
    required = {"schema", "provenance", "inputs", "expected", "sha256"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid external orbit fixture fields")
    body = {k: v for k, v in payload.items() if k != "sha256"}
    if payload["schema"] != "space-env.external-orbit-fixture/1" or payload[
        "sha256"
    ] != digest(body):
        raise ValueError("unsupported or corrupt orbit fixture")
    provenance = payload["provenance"]
    required_provenance = {
        "library",
        "version",
        "source_url",
        "source_sha256",
        "license",
        "generator_sha256",
        "independence",
        "shared_inputs",
        "integrator",
        "scope",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("fixture provenance is incomplete")
    if any(not provenance[k] for k in required_provenance):
        raise ValueError("empty fixture provenance")
    if provenance["independence"] != "external-force-formula/shared-inputs":
        raise ValueError(
            "this verifier only supports declared shared-input crosschecks"
        )
    data, expected = payload["inputs"], payload["expected"]
    if set(data) != {
        "epoch_utc",
        "frame",
        "state_si",
        "eop",
        "forces",
        "duration_sec",
        "sample_sec",
        "constants",
    }:
        raise ValueError("unknown fixture inputs")
    if data["frame"] != "GCRS" or set(expected) != {"times_sec", "states_si"}:
        raise ValueError("fixture requires GCRS position and velocity in SI")
    from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR

    forces = ForceModel(**data["forces"])
    constants = {
        "mu_m3_s2": MU_EARTH,
        "j2": J2_EARTH,
        "radius_m": R_EARTH_EQUATOR,
    }
    if forces.third_body:
        constants.update(mu_sun_m3_s2=1.32712440018e20, mu_moon_m3_s2=4.9048695e12)
        if (
            "ERFA Sun/Moon positions via shared _ephemerides provider"
            not in provenance["shared_inputs"]
        ):
            raise ValueError("combined fixture must declare shared Sun/Moon inputs")
    if data["constants"] != constants:
        raise ValueError("fixture physical constants differ")
    eop = data["eop"]
    context = FrameContext(
        EOPSeries(
            eop["rows"],
            source=eop["source"],
            allow_predictions=eop["allow_predictions"],
        )
    )
    initial = np.asarray(data["state_si"], dtype=float)
    if initial.shape != (6,) or not np.isfinite(initial).all():
        raise ValueError("invalid fixture initial state")
    if not forces.j2 or any((forces.drag, forces.srp)):
        raise ValueError(
            "fixture scope is central+J2 with optional Sun/Moon third body"
        )
    result = propagate(
        ReferenceState(Epoch.utc(data["epoch_utc"]), initial[:3], initial[3:]),
        data["duration_sec"],
        context,
        model=forces,
        sample_sec=data["sample_sec"],
    )
    actual, reference = (
        np.asarray(result["states_si"]),
        np.asarray(expected["states_si"], dtype=float),
    )
    times = np.asarray(expected["times_sec"], dtype=float)
    if (
        reference.shape != actual.shape
        or not np.isfinite(reference).all()
        or not np.array_equal(times, result["times_sec"])
    ):
        raise ValueError("fixture samples and coverage differ")
    if not np.allclose(reference[0], initial, atol=1e-12, rtol=0):
        raise ValueError("fixture initial conditions differ")
    difference = actual - reference
    position = float(np.max(np.linalg.norm(difference[:, :3], axis=1)))
    velocity = float(np.max(np.linalg.norm(difference[:, 3:], axis=1)))
    return {
        "schema": "space-env.external-orbit-comparison/1",
        "fixture_sha256": payload["sha256"],
        "passed": position <= POSITION_TOLERANCE_M
        and velocity <= VELOCITY_TOLERANCE_M_S,
        "duration_sec": data["duration_sec"],
        "position_error_m": position,
        "velocity_error_m_s": velocity,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "velocity_tolerance_m_s": VELOCITY_TOLERANCE_M_S,
        "provenance": provenance,
        "production_sources": result["manifest"]["model_source_sha256"],
        "not_qualified": [
            "independent frames/EOP/constants",
            "combined drag/SRP perturbations",
            "SRP/eclipse",
            "atmosphere/weather",
            "navigation covariance",
            "flight or mission-wide accuracy",
        ],
    }


def verify_packaged_orbit_fixtures() -> dict:
    root = Path(__file__).resolve().parents[1] / "reference/fixtures"
    paths = sorted(root.glob("boinor_j2_*.json"))
    if not paths:
        raise ValueError("no frozen orbit fixtures installed")
    rows = [verify_orbit_fixture(json.loads(p.read_text())) for p in paths]
    return {
        "schema": "space-env.orbit-fixture-suite/1",
        "passed": all(r["passed"] for r in rows),
        "cases": rows,
        "qualification_status": "needs_external_evidence",
    }
