"""Independent Kepler truth and strict cross-engine input matching."""

import copy

import pytest

from space_env.simulation.snapshot import digest
from space_env.validation.trajectory_exchange import (
    compare_external,
    kepler_state,
    native_trajectories,
    trajectory_manifest,
)


def submission(manifest):
    return {
        "schema": "space-env.external-trajectories/1",
        "manifest_sha256": digest(manifest),
        "engine": "Basilisk",
        "engine_version": "synthetic-fixture-not-an-engine-run",
        "adapter_source_sha256": "0" * 64,
        "solver_settings": {"oracle": "test-only"},
        "trajectories": [
            {
                **{
                    k: c[k]
                    for k in ("frame", "epoch_utc", "time_basis", "units", "times_sec")
                },
                "case_id": c["id"],
                "case_sha256": digest(c),
                "states_si": [
                    kepler_state(
                        c["semimajor_axis_m"],
                        c["eccentricity"],
                        c["inclination_rad"],
                        t,
                    )
                    for t in c["times_sec"]
                ],
            }
            for c in manifest["cases"]
        ],
    }


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_native_against_independent_kepler(profile):
    result = native_trajectories(trajectory_manifest(profile))
    assert result["passed"]
    assert len(result["rows"]) == 3


def test_external_matching_and_accuracy_failures():
    manifest = trajectory_manifest("smoke")
    data = submission(manifest)
    assert compare_external(manifest, data)["passed"]
    data["trajectories"][0]["states_si"][-1][0] += 10.0
    result = compare_external(manifest, data)
    assert not result["passed"]
    assert len(result["rows"]) == 3
    assert "not independently attested" in result["provenance"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "manifest",
        "case",
        "grid",
        "initial",
        "frame",
        "units",
        "epoch",
        "nonfinite",
        "shape",
    ],
)
def test_reject_incomparable_or_partial_submissions(fault):
    manifest = trajectory_manifest("smoke")
    data = submission(manifest)
    first = data["trajectories"][0]
    if fault == "missing":
        data["trajectories"].pop()
    elif fault == "duplicate":
        data["trajectories"][-1] = copy.deepcopy(first)
    elif fault == "manifest":
        data["manifest_sha256"] = "1" * 64
    elif fault == "case":
        first["case_sha256"] = "1" * 64
    elif fault == "grid":
        first["times_sec"][1] += 1.0
    elif fault == "initial":
        first["states_si"][0][0] += 1.0
    elif fault == "frame":
        first["frame"] = "TEME"
    elif fault == "units":
        first["units"] = "km,km/s"
    elif fault == "epoch":
        first["epoch_utc"] = "2026-07-18T12:00:01Z"
    elif fault == "nonfinite":
        first["states_si"][1][0] = float("nan")
    else:
        first["states_si"].pop()
    with pytest.raises(ValueError):
        compare_external(manifest, data)


def test_cannot_relax_frozen_thresholds():
    manifest = trajectory_manifest("smoke")
    manifest["cases"][0]["position_tolerance_m"] = 1e10
    with pytest.raises(ValueError, match="frozen"):
        compare_external(manifest, submission(manifest))
    with pytest.raises(ValueError, match="frozen"):
        native_trajectories(manifest)
