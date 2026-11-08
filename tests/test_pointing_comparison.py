"""Reproduced planner ablation cannot claim success from a forged result file."""

import copy
import shutil

import pytest

from examples.network.pointing_comparison import audit, run
from space_env.llm.episode import seal
from space_env.simulation.snapshot import canonical_json


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    root = tmp_path_factory.mktemp("pointing-comparison") / "report"
    report = run(root, ["equatorial-relay-chain"])
    assert all(r["status"] == "completed" for r in report["rows"])
    return root, report


def test_native_matched_ablation_uses_capture_tracking_guard(comparison):
    root, _ = comparison
    proof = audit(root)
    assert proof["replayed"] == 3
    assert proof["paired"][0]["fulfilled"] == {
        "serial": 1,
        "exact_small": 1,
        "pointing_bounded": 1,
    }
    assert not proof["external_llm_measured"]


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "metric", "method", "bound", "diagnostic", "artifact"],
)
def test_resealed_comparison_does_not_accept_false_planner_claims(
    comparison, tmp_path, fault
):
    source, report = comparison
    root = tmp_path / "report"
    shutil.copytree(source, root)
    report = copy.deepcopy(report)
    if fault == "missing":
        report["rows"].pop()
    elif fault == "duplicate":
        report["rows"][-1] = copy.deepcopy(report["rows"][0])
    elif fault == "metric":
        report["rows"][0]["metrics"]["fulfilled"] = 3
    elif fault == "method":
        report["rows"][0]["method"] = "invented"
    elif fault == "bound":
        report["rows"][0]["optimal_in_forecast"] = True
    elif fault == "artifact":
        report["rows"][0]["artifact"] = "../elsewhere.json"
    else:
        report["rows"][0]["diagnostic"]["all_requests_catalogued"] = True
    (root / "report.json").write_text(
        canonical_json(seal({k: v for k, v in report.items() if k != "sha256"}))
    )
    with pytest.raises(ValueError):
        audit(root)
