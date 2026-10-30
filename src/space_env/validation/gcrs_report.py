"""Measured reference-profile checks, with scope/tolerance declared before running."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.frames import c2t_matrix


def reference_physics_report(*, long_arcs: bool = False, crosscheck: bool = False) -> dict:
    path = Path(__file__).resolve().parents[1] / "reference/fixtures/c2t06a.json"
    fixture = json.loads(path.read_text())
    matrix = c2t_matrix(*fixture["tt_jd"], *fixture["ut1_jd"], fixture["xp_rad"], fixture["yp_rad"])
    error = float(np.max(np.abs(matrix-fixture["expected_matrix"])))
    rows = [{"name": "upstream_c2t06a", "measured": error, "tolerance": fixture["absolute_tolerance"],
             "passed": error <= fixture["absolute_tolerance"], "source": fixture["source"],
             "scope": "frozen backend regression; not an independent force/ephemeris validation"}]
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    context = FrameContext(EOPSeries.packaged())
    radius = 7e6
    speed, motion = np.sqrt(MU_EARTH/radius), np.sqrt(MU_EARTH/radius**3)
    initial = ReferenceState(epoch, np.array([radius, 0, 0]), np.array([0, 0, speed]))
    for duration in ([600, 86400, 259200] if long_arcs else [600]):
        result = propagate(initial, duration, context, sample_sec=600)
        states, phase = np.array(result["states_si"]), np.array(result["times_sec"])*motion
        exact = np.stack([radius*np.cos(phase), phase*0, radius*np.sin(phase),
                          -speed*np.sin(phase), phase*0, speed*np.cos(phase)], axis=1)
        position = float(np.max(np.linalg.norm(states[:, :3]-exact[:, :3], axis=1)))
        velocity = float(np.max(np.linalg.norm(states[:, 3:]-exact[:, 3:], axis=1)))
        rows.extend([
            {"name": f"circular_{duration}s_position_m", "measured": position, "tolerance": 1.,
             "passed": position <= 1., "scope": "closed-form circular two-body orbit only"},
            {"name": f"circular_{duration}s_velocity_m_s", "measured": velocity, "tolerance": .001,
             "passed": velocity <= .001, "scope": "closed-form circular two-body orbit only"},
        ])
    report = {"schema": "space-env.reference-validation/1", "profile": "gcrs-reference/1",
            "passed": all(row["passed"] for row in rows), "long_arcs": long_arcs, "checks": rows,
            "manifest": context.manifest(),
            "unqualified": ["perturbed 24/72h against independent external force models",
                            "complete mission executor migration", "dX/dY correction", "ADCS/thermal/RF fidelity"]}

    if crosscheck:
        from space_env.validation.force_crosscheck import force_crosscheck

        report["force_crosscheck"] = force_crosscheck(long_arcs=long_arcs)
        report["passed"] = report["passed"] and report["force_crosscheck"]["passed"]
    return report
