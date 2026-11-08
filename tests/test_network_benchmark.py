"""Split/information guards, optimizer mathematics and native dispatch admission."""

import copy

import numpy as np
import pytest

from space_env.planning import network_benchmark as benchmark
from space_env.planning.joint_network import schedule_network
from space_env.runtime import MissionRuntime
from space_env.simulation.snapshot import digest
from space_env.validation.network_evidence import seal, source_manifest


def synthetic_view(case):
    """Synthetic optimizer unit input; not a measured mission forecast."""
    cfg = case.configuration.mission
    actors = [f"fleet-{i:02d}" for i in range(3)]
    demands = [
        {
            "id": r.id,
            "owners": actors,
            "stations": ["Ground"],
            "release_sec": r.release_sec,
            "deadline_sec": r.deadline_sec,
            "priority": r.priority,
            "raw_bits": r.raw_bits,
            "output_bits": r.output_bits,
            "slew_cells": 1,
            "capture_cells": 1,
            "cpu_cells": 1,
            "relay_cells": 1,
            "ground_cells": 1,
            "owner_wh": 1.0,
            "relay_wh": 1.0,
            "receiver_wh": 0.5,
            "ground_wh": 1.0,
        }
        for r in cfg.requests
    ]
    return seal(
        {
            "schema": "space-env.joint-network-view/1",
            "horizon_sec": 180.0,
            "decision_sec": 5.0,
            "actors": {a: {"memory_bits": 1e8, "usable_wh": 100.0} for a in actors},
            "stations": {"Ground": 1},
            "demands": demands,
            "edges": [
                {"source": a, "destination": "Ground", "cells": [True] * 36}
                for a in actors
            ],
            "captures": [
                {"request": r.id, "actor": a, "cells": [True] * 36}
                for r in cfg.requests
                for a in actors
            ],
            "geometry_sha256": "0" * 64,
        }
    )


def training_data():
    suite = benchmark.network_suite()
    spec = benchmark.NetworkSuite.model_validate(suite)
    cases = [c for c in spec.cases if c.split == "train"]
    views, rows = {}, []
    for c in cases:
        views[c.id] = synthetic_view(c)
        for strategy in benchmark.STRATEGIES:
            winner = "request-1" if strategy == "earliest_deadline" else "request-0"
            outcomes = {
                r.id: {
                    "status": "completed" if r.id == winner else "released",
                    "priority": r.priority,
                    "reason": None,
                }
                for r in c.configuration.mission.requests
            }
            row = seal(
                {
                    "schema": "space-env.network-scale-row/1",
                    "case_id": c.id + "--" + strategy,
                    "input_sha256": digest(
                        benchmark.workload(c, strategy, views[c.id])
                    ),
                    "source_sha256": digest(source_manifest()),
                    "status": "completed",
                    "total": 3,
                    "native_fulfilled": 1,
                    "weighted_fulfilled": outcomes[winner]["priority"],
                    "priority_total": 18,
                    "request_outcomes": outcomes,
                    "predicted_fulfilled": 1,
                    "search_nodes": 3,
                    "search_budget_exhausted": False,
                    "source_unchanged": None,
                    "planning_sec": 0.1,
                    "checkpoint_sec": 0.1,
                    "elapsed_sec": 1.0,
                    "step_sec": {"p50": 0.01, "p95": 0.02, "max": 0.03},
                    "peak_rss_bytes": 1000000,
                    "peak_rss_method": "linux-proc-VmHWM-after-exec",
                    "checkpoint_bytes": 1000,
                    "sdk_steps": 36,
                }
            )
            rows.append(
                {
                    "case": c.id,
                    "group": c.group,
                    "split": c.split,
                    "strategy": strategy,
                    "forecast_sha256": views[c.id]["sha256"],
                    "result": row,
                }
            )
    return suite, views, rows


def test_suite_is_frozen_and_groups_do_not_cross_splits():
    suite = benchmark.network_suite()
    assert suite == benchmark.network_suite()
    assert len(suite["cases"]) == 12
    assert len({c["geometry_sha256"] for c in suite["cases"]}) == 6
    groups = {}
    for c in suite["cases"]:
        groups.setdefault(c["geometry_sha256"], set()).add(c["split"])
    assert all(len(v) == 1 for v in groups.values())


@pytest.mark.parametrize(
    "fault",
    [
        "geometry",
        "group",
        "renamed_geometry",
        "condition",
        "case",
        "duplicate",
        "missing_split",
    ],
)
def test_suite_rejects_leakage_and_mislabelled_inputs(fault):
    s = benchmark.network_suite()
    if fault == "geometry":
        s["cases"][0]["geometry_sha256"] = "f" * 64
    elif fault == "group":
        s["cases"][-1]["group"] = s["cases"][0]["group"]
    elif fault == "renamed_geometry":
        s["cases"][-1]["configuration"] = copy.deepcopy(s["cases"][1]["configuration"])
        s["cases"][-1]["configuration"]["mission"]["name"] = "Renamed clone"
        s["cases"][-1]["configuration"]["mission"]["seed"] = 99
        s["cases"][-1]["geometry_sha256"] = benchmark.geometry_key(
            s["cases"][-1]["configuration"]
        )
    elif fault == "condition":
        s["cases"][0]["condition"] = "station_outage"
    elif fault == "case":
        s["cases"][-1]["id"] = s["cases"][0]["id"]
    elif fault == "duplicate":
        s["cases"][1] = {**copy.deepcopy(s["cases"][0]), "id": "duplicate"}
    else:
        s["cases"] = [c for c in s["cases"] if c["group"] != "equatorial"]
    with pytest.raises(ValueError):
        benchmark.NetworkSuite.model_validate(s)


def test_names_and_seeds_do_not_create_new_geometry():
    cfg = benchmark.network_suite()["cases"][0]["configuration"]
    key = benchmark.geometry_key(cfg)
    cfg["mission"]["name"] = "Another name"
    cfg["mission"]["seed"] = 800
    cfg["mission"]["stations"][0]["name"] = "Another-ID"
    assert benchmark.geometry_key(cfg) == key


def test_phase_permutation_angle_wrap_epoch_and_equatorial_aliases_are_same_group():
    cfg = benchmark.network_suite()["cases"][0]["configuration"]
    key = benchmark.geometry_key(cfg)
    cfg["mission"]["epoch_utc"] = "2026-07-18T12:00:00.000Z"
    cfg["mission"]["planes"][0]["raan_deg"] = 360.0
    cfg["mission"]["planes"][0]["phases_deg"].reverse()
    assert benchmark.geometry_key(cfg) == key
    cfg["mission"]["planes"][0]["raan_deg"] = 10.0
    cfg["mission"]["planes"][0]["phases_deg"] = [
        (x - 10.0) % 360.0 for x in cfg["mission"]["planes"][0]["phases_deg"]
    ]
    assert benchmark.geometry_key(cfg) == key
    cfg["mission"]["stations"][0]["lon_deg"] = 180.0
    wrapped = benchmark.geometry_key(cfg)
    cfg["mission"]["stations"][0]["lon_deg"] = -180.0
    assert benchmark.geometry_key(cfg) == wrapped


def test_relabelled_angle_alias_cannot_cross_split():
    suite = benchmark.network_suite()
    clone = copy.deepcopy(suite["cases"][1])
    clone.update(id="wrapped-test", group="new-name", split="test")
    clone["configuration"]["mission"]["planes"][0]["phases_deg"].reverse()
    clone["configuration"]["mission"]["planes"][0]["raan_deg"] = 360.0
    clone["geometry_sha256"] = benchmark.geometry_key(clone["configuration"])
    suite["cases"][-1] = clone
    with pytest.raises(ValueError):
        benchmark.NetworkSuite.model_validate(suite)


def test_optimizer_learns_from_train_only_and_never_executes_native_future(monkeypatch):
    suite, views, rows = training_data()

    def forbidden(*args, **kwargs):
        raise AssertionError("Native rollout in optimizer")

    monkeypatch.setattr(MissionRuntime, "submit", forbidden)
    monkeypatch.setattr(MissionRuntime, "advance", forbidden)
    model = benchmark.fit_selector(suite, views, rows, seed=4)
    assert model == benchmark.fit_selector(suite, views, rows, seed=4)
    assert all(
        benchmark.choose_strategy(model, view) == "earliest_deadline"
        for view in views.values()
    )
    assert set(model["training_cases"]) == set(views)
    assert np.isfinite(model["weights"]).all()


@pytest.mark.parametrize(
    "fault",
    ["test_view", "test_row", "missing_row", "failed_run", "information", "score"],
)
def test_training_rejects_test_data_incomplete_rows_and_forged_rewards(fault):
    suite, views, rows = training_data()
    if fault == "test_view":
        views["retrograde-98-nominal"] = copy.deepcopy(next(iter(views.values())))
    elif fault == "test_row":
        rows[0]["case"] = "retrograde-98-nominal"
    elif fault == "missing_row":
        rows.pop()
    elif fault == "failed_run":
        rows[0]["result"]["status"] = "error"
    elif fault == "information":
        rows[0]["forecast_sha256"] = "f" * 64
    else:
        rows[0]["result"]["weighted_fulfilled"] = 999
    with pytest.raises(ValueError):
        benchmark.fit_selector(suite, views, rows)


def test_selector_rejects_tampering_and_invalid_weight_shape():
    suite, views, rows = training_data()
    model = benchmark.fit_selector(suite, views, rows)
    v = next(iter(views.values()))
    corrupted = copy.deepcopy(model)
    corrupted["weights"] = [[1.0]]
    with pytest.raises(ValueError):
        benchmark.choose_strategy(corrupted, v)
    corrupted = seal({k: val for k, val in corrupted.items() if k != "sha256"})
    with pytest.raises(ValueError):
        benchmark.choose_strategy(corrupted, v)


def test_explicit_request_order_cannot_drop_or_duplicate_demands():
    from tests.test_joint_network import seal as reseal
    from tests.test_joint_network import view

    v = view()
    v["demands"][0].update(owners=["b"], deadline_sec=5.0)
    v["demands"].append({**v["demands"][0], "id": "second", "priority": 1})
    v["captures"] = [
        {"request": rid, "actor": "b", "cells": [True] * 8}
        for rid in ("survey", "second")
    ]
    v = reseal(v)
    assert schedule_network(v)["allocations"][0]["request"] == "survey"
    assert (
        schedule_network(v, request_order=["second", "survey"])["allocations"][0][
            "request"
        ]
        == "second"
    )
    for invalid in (["second"], ["survey", "survey"], ["unknown", "survey"], "survey"):
        with pytest.raises(ValueError):
            schedule_network(v, request_order=invalid)


def comparison_rows():
    suite = benchmark.network_suite()
    policies = [*benchmark.STRATEGIES, "learned-0"]
    rows = [
        {
            "case": c["id"],
            "group": c["group"],
            "split": c["split"],
            "policy": p,
            "strategy": p if p in benchmark.STRATEGIES else "priority",
            "forecast_sha256": "f" * 64,
            "result": {
                "status": "completed",
                "total": 3,
                "weighted_fulfilled": 9,
                "priority_total": 18,
            },
        }
        for c in suite["cases"]
        for p in policies
    ]
    return suite, policies, rows


def test_comparison_preserves_failures_as_unknown_bounds_and_no_superiority():
    suite, policies, rows = comparison_rows()
    target = next(
        r for r in rows if r["split"] == "test" and r["policy"] == "learned-0"
    )
    target["result"] = {"status": "timeout", "total": 3, "native_fulfilled": None}
    report = benchmark.summarize(suite, rows, policies)
    aggregate = next(
        a
        for a in report["aggregates"]
        if a["split"] == "test" and a["policy"] == "learned-0"
    )
    assert aggregate["rows"] == 4 and aggregate["completed_rows"] == 3
    assert aggregate["mean_value_lower_bound"] < aggregate["mean_value_upper_bound"]
    assert all(not x["superiority_established"] for x in report["paired"])
    assert not report["expert_teacher_qualified"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "split", "denominator", "information", "label"]
)
def test_comparison_rejects_incomplete_or_unequal_pairs(fault):
    suite, policies, rows = comparison_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "split":
        rows[0]["split"] = "test"
    elif fault == "denominator":
        rows[0]["result"]["total"] = 1
    elif fault == "information":
        rows[0]["forecast_sha256"] = "0" * 64
    else:
        rows[0]["strategy"] = "earliest_deadline"
    with pytest.raises(ValueError):
        benchmark.summarize(suite, rows, policies)
