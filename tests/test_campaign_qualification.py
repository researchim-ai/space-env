"""Declared campaign geometry/causality, native witnesses and isolated scale semantics."""

import copy

import pytest

from space_env.scaling.benchmark import (
    ScaleSpec,
    Workload,
    compare_semantics,
    run_scale,
)
from space_env.scaling.campaign import (
    CampaignSpec,
    campaign_evidence,
    campaign_workload,
)
from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession


@pytest.fixture(scope="module")
def workload():
    return campaign_workload(
        {"cameras": 1, "relays": 1, "waves": 2, "horizon_sec": 40.0, "repeats": 1}
    )


def test_campaign_has_independent_arcs_later_arrivals_and_real_pipeline(workload):
    d = workload["scenario"]["definition"]
    t = workload["task"]
    assert len({digest(o["initial"]) for o in d["orbits"].values()}) == 2
    assert [r["release_sec"] for r in d["lifecycle"]["requests"]] == [0.0, 12.0]
    assert not d["operations"]["products"]
    assert len(d["lifecycle"]["jobs"]) == 2
    assert {c["kind"] for a in t["actions"] for c in a["commands"]} == {
        "imaging",
        "transform",
        "relay",
        "downlink",
    }
    assert all(
        c["actor_id"].startswith("relay")
        for a in t["actions"]
        for c in a["commands"]
        if c["kind"] == "downlink"
    )
    assert d["operations"]["stations"][0]["terminals"] == 1
    assert (
        len(d["lifecycle"]["cloud_masks"])
        == len(d["lifecycle"]["areas_of_interest"])
        == 2
    )
    assert all(
        c["state"] == "clear" and c["source_status"] == "synthetic"
        for c in d["lifecycle"]["cloud_masks"]
    )


def test_generator_repeatability_and_no_exposed_future_requests(workload):
    second = campaign_workload(
        {"cameras": 1, "relays": 1, "waves": 2, "horizon_sec": 40.0, "repeats": 1}
    )
    assert second == workload
    session = TaskSession(workload["scenario"], workload["task"])
    view = session.observe()
    assert not any(e["id"].startswith("w01-") for e in view["entities"])
    assert all(
        not c.get("forecast")
        for a in workload["task"]["actions"]
        for c in a["commands"]
    )


@pytest.mark.parametrize(
    "value",
    [
        {"cameras": 13},
        {"cameras": 12, "relays": 5},
        {"waves": 16, "wave_gap_sec": 60.0},
        {"waves": 15, "wave_gap_sec": 4.0, "cameras": 12, "horizon_sec": 90.0},
        {"horizon_sec": 301.0},
        {"horizon_sec": True},
        {"seed": True},
        {"mode": "unqualified"},
    ],
)
def test_campaign_rejects_outside_envelope(value):
    with pytest.raises(ValueError):
        CampaignSpec.model_validate(value)


def test_negative_cases_declare_upper_bound_before_execution():
    for mode in ["ttl", "ground_outage"]:
        w = campaign_workload(
            {
                "cameras": 1,
                "relays": 1,
                "waves": 1,
                "horizon_sec": 20.0,
                "repeats": 1,
                "mode": mode,
            }
        )
        assert w["min_completion_fraction"] == w["max_completion_fraction"] == 0.0
        if mode == "ttl":
            assert all(
                c["relay"]["ttl_sec"] == 0.01
                for a in w["task"]["actions"]
                for c in a["commands"]
                if c["kind"] == "relay"
            )
        else:
            assert (
                w["scenario"]["definition"]["operations"]["outages"][0]["end_sec"]
                == 20.0
            )


def test_witness_cannot_confuse_ground_ack_with_fulfilled_request(workload):
    report = {
        "lifecycle": {
            "requests": {"r": {"release_sec": 12.0, "status": "expired"}},
            "jobs": {"j": {"status": "completed"}},
            "artifacts": {"raw": {"source": "acquired"}},
            "acknowledgements": [{"actor": "relay-00"}],
        },
        "dtn": {"transfers": {"t": {"status": "completed"}}},
        "spacecraft_bus": {"states": {"a": {}}},
        "radio_network": {"model": "sampled"},
    }
    witness = campaign_evidence(
        workload["scenario"], report, workload["campaign_gates"]
    )
    assert not witness["passed"]
    assert witness["jobs_completed"] == 1 and witness["custody_completed"] == 1
    assert not witness["checks"]["later_arrivals_completed"]


def test_comparison_groups_reject_different_native_inputs(workload):
    a = copy.deepcopy(workload)
    a.update(id="plain", comparison_group="restart")
    b = copy.deepcopy(a)
    b.update(id="resumed", resume_after_decisions=2)
    ScaleSpec.model_validate({"id": "paired", "workloads": [a, b]})
    b["scenario"]["seed"] += 1
    with pytest.raises(ValueError):
        ScaleSpec.model_validate({"id": "bad", "workloads": [a, b]})
    with pytest.raises(ValueError):
        ScaleSpec.model_validate({"id": "single", "workloads": [a]})


def test_failed_repeat_is_never_dropped_from_semantic_readiness(workload):
    a = copy.deepcopy(workload)
    a.update(id="plain", comparison_group="restart")
    b = copy.deepcopy(a)
    b.update(id="resumed", resume_after_decisions=2)
    spec = ScaleSpec.model_validate({"id": "paired", "workloads": [a, b]})
    rows = [
        {
            "id": "plain",
            "passed": True,
            "native_backend_sha256": "x",
            "trace_sha256": "y",
        },
        {"id": "resumed", "passed": False, "error": {"type": "TimeoutExpired"}},
    ]
    result = compare_semantics(spec, rows)["restart"]
    assert (
        not result["passed"]
        and result["expected_repeats"] == 2
        and result["successful_repeats"] == 1
    )


@pytest.mark.parametrize("workers", [0, True, 5])
def test_isolated_concurrency_is_bounded(workload, workers):
    with pytest.raises(ValueError):
        ScaleSpec.model_validate(
            {"id": "bounded", "workers": workers, "workloads": [workload]}
        )


def test_real_isolated_parallel_resume_matches_uninterrupted(tmp_path):
    w = fleet_workload(
        {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 6.0}
    )
    a = Workload.model_validate(
        {**w, "id": "straight", "comparison_group": "recovery"}
    ).model_dump(mode="json")
    b = copy.deepcopy(a)
    b.update(id="resumed", resume_after_decisions=1)
    result = run_scale(
        {"id": "isolated-native", "workers": 2, "workloads": [a, b]},
        tmp_path / "result",
    )
    assert result["ok"], result
    assert [r["id"] for r in result["rows"]] == ["straight", "resumed"]
    assert len({r["worker_environment"]["pid"] for r in result["rows"]}) == 2
    assert result["semantic_comparisons"]["recovery"]["passed"]
    assert result["rows"][1]["resume_sec"] > 0
    assert result["rows"][1]["prefix_checkpoint_bytes"] > 0
