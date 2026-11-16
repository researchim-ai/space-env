"""Optional real-engine execution. Skips only if Basilisk is not installed."""

import pytest

pytest.importorskip("Basilisk")

from examples.validation.basilisk_trajectory import run
from space_env.validation.trajectory_exchange import (
    compare_external,
    trajectory_manifest,
)


def test_basilisk_full_central_profile():
    manifest = trajectory_manifest("full")
    submission = run(manifest)
    report = compare_external(manifest, submission)
    assert report["engine"] == "Basilisk"
    assert len(report["rows"]) == 3
    assert report["passed"]
    assert all(r["max_position_error_m"] <= 0.01 for r in report["rows"])
    assert all(r["max_velocity_error_m_s"] <= 1e-5 for r in report["rows"])
