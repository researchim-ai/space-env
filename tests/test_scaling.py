"""B09 replay equivalence, bounded IO, isolation and cache accuracy gates."""

from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc

import numpy as np
import pytest

from space_env.planning import PlanningRun
from space_env.planning.benchmark import run_benchmark, verify_evaluation
from space_env.reference.frames import c2t_matrix, clear_frame_cache, frame_cache_info
from space_env.scaling.spool import RowSpool, SpoolBudget
from space_env.simulation.snapshot import canonical_json, digest
from space_env.simulation.streaming import (
    array_digest,
    atomic_jsonl,
    file_digest,
    iter_jsonl,
)
from space_env.tasks import TaskSession, iter_dataset
from tests.test_planning import scenario as fixture_scenario
from tests.test_planning import task


def scenario():
    return fixture_scenario.__wrapped__()


@pytest.mark.parametrize(
    "rows", [[], [{}], [{"x": "Земля", "a": [1, 2.5]}, {"b": False}]]
)
def test_stream_hash(rows, tmp_path):
    path = tmp_path / "rows.jsonl"
    result = atomic_jsonl(path, iter(rows))
    assert list(iter_jsonl(path)) == rows
    assert result["rows_sha256"] == array_digest(iter(rows)) == digest(rows)
    assert (
        result["file_sha256"]
        == file_digest(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "budget", [{"max_rows": 1}, {"max_row_bytes": 4}, {"max_bytes": 8}]
)
def test_atomic_budget_preserves_destination(budget, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("old")
    with pytest.raises(ValueError):
        atomic_jsonl(path, [{"x": 1}, {"x": 2}], **budget)
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_late_generator_failure(tmp_path):
    def broken():
        yield {"ok": True}
        raise RuntimeError("late producer failure")

    path = tmp_path / "out"
    with pytest.raises(RuntimeError):
        atomic_jsonl(path, broken())
    assert not path.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "payload", [b'{"x":NaN}\n', b'{"x":Infinity}\n', b"\n", b"not json\n"]
)
def test_reader_rejects_invalid(payload, tmp_path):
    path = tmp_path / "in"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        list(iter_jsonl(path))


def test_reader_bounded(tmp_path):
    path = tmp_path / "in"
    path.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="budget"):
        list(iter_jsonl(path, max_row_bytes=10))


def test_stream_memory_does_not_materialize_corpus(tmp_path):
    tracemalloc.start()
    try:
        receipt = atomic_jsonl(
            tmp_path / "out", ({"i": i, "text": "x" * 4096} for i in range(4000))
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert receipt["bytes"] > 16_000_000
    assert peak < 2_000_000


def test_spool_transaction_and_iteration(tmp_path):
    spool = RowSpool(tmp_path / "spool")
    try:
        rows = spool.rows("train")
        spool.begin()
        rows.append({"i": 0})
        spool[("trajectory", "a")] = {"split": "train"}
        spool.finish()
        spool.begin()
        rows.append({"i": 1})
        spool[("trajectory", "b")] = {"split": "holdout"}
        spool.finish(rollback=True)
        assert len(rows) == 1 and list(rows) == [{"i": 0}]
        assert ("trajectory", "b") not in spool
        assert spool[("trajectory", "a")] == {"split": "train"}
        with pytest.raises(KeyError):
            spool["missing"]
        assert spool.connection.execute("PRAGMA cache_size").fetchone()[0] == -2048
    finally:
        spool.close()


@pytest.mark.parametrize("mode", ["row", "disk"])
def test_spool_budget(tmp_path, mode):
    spool = RowSpool(
        tmp_path / "spool",
        max_bytes=65536,
        max_row_bytes=100 if mode == "row" else 10000,
    )
    try:
        with pytest.raises(SpoolBudget):
            for i in range(1000):
                spool.rows("train").append({"i": i, "x": "a" * 5000})
    finally:
        spool.close()


@pytest.mark.parametrize("compact_runtime", [False, True])
def test_compact_planning_cold_replay_resume(compact_runtime):
    t = task()
    t["runtime_audit"] = "compact" if compact_runtime else "full"
    run = PlanningRun(scenario(), t, {"name": "route_aware_rolling"})
    run.run(decisions=2)
    full = run.checkpoint()
    compact = run.checkpoint(compact=True)
    assert compact["schema"] == "space-env.planning-run/2"
    assert compact["episode"]["runtime"]["schema"] == "space-env.runtime-checkpoint/2"
    assert len(canonical_json(compact)) < len(canonical_json(full))
    restored = PlanningRun.from_checkpoint(compact)
    assert restored.checkpoint() == compact
    assert list(iter_dataset(compact["episode"]))
    restored.run()
    run.run()
    assert run.trace == restored.trace
    assert (
        run.session._runtime.checkpoint()["backend_report"]
        == restored.session._runtime.checkpoint()["backend_report"]
    )


def test_compact_evidence_tamper_is_caught_after_rehash():
    run = PlanningRun(scenario(), task())
    run.step()
    record = run.checkpoint(compact=True)["episode"]
    record["transitions"][0]["transition_sha256"] = "0" * 64
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        TaskSession.from_checkpoint(record)


def test_fork_private_history_shared_public_output_owned():
    run = PlanningRun(scenario(), task())
    run.step()
    original = run.session._runtime
    before = original.checkpoint()
    clone = original.fork()
    assert clone._operations is not original._operations
    assert clone._operations[0] is original._operations[0]
    clone._operations.append({"fork-only": True})
    assert original.checkpoint() == before
    exposed = original.checkpoint()
    exposed["operations"][0].clear()
    assert original.checkpoint() == before
    assert clone._backend is not original._backend
    assert clone._commands is not original._commands


@pytest.mark.parametrize("offset", [0.0, 1e-9, 0.5, 100.0])
def test_frame_cache_exact_and_owned(offset):
    clear_frame_cache()
    args = (2451545.0, offset, 2451545.0, offset, 1e-6, -2e-6)
    cached = c2t_matrix(*args)
    direct = c2t_matrix(*args, use_cache=False)
    assert np.array_equal(cached, direct)
    cached[:] = 0
    assert np.array_equal(c2t_matrix(*args), direct)
    assert frame_cache_info()["hits"] == 1


def test_frame_cache_bounded():
    clear_frame_cache()
    for i in range(4100):
        c2t_matrix(2451545.0, i / 100000, 2451545.0, i / 100000, 0.0, 0.0)
    assert frame_cache_info()["currsize"] == 4096


def test_isolated_workers_semantic_parity(tmp_path):
    manifest = {
        "id": "parallel",
        "cases": [
            {
                "id": "fixed",
                "family": "synthetic",
                "scenario": scenario(),
                "task": task(),
                "episode_seeds": [0, 1],
            }
        ],
        "policies": [{"name": "priority"}],
    }
    serial = run_benchmark(manifest, artifacts=tmp_path / "serial")
    parallel = run_benchmark(
        {**manifest, "workers": 2}, artifacts=tmp_path / "parallel"
    )
    assert parallel["passed_execution"]
    assert len({r["worker_meta"]["pid"] for r in parallel["rows"]}) == 2
    for s, p in zip(serial["rows"], parallel["rows"], strict=True):
        assert s["episode_sha256"] == p["episode_sha256"]
        assert s["decision_trace_sha256"] == p["decision_trace_sha256"]
    assert verify_evaluation(parallel)["ok"]


def registry_spec():
    t = task()
    t["decision_sec"] = 2.0
    return {
        "id": "registry",
        "templates": [
            {
                "id": "mission",
                "family": "delivery",
                "kind": "ground",
                "source": "synthetic test fixture",
                "scenario": scenario(),
                "task": t,
                "split": "train",
            }
        ],
        "policies": [{"name": "route_aware_rolling"}],
        "quality": {"require_teacher_gates": False},
        "export_roles": ["@central"],
    }


def test_registry_idempotent_cross_split_and_duplicate_rejected(tmp_path):
    import sqlite3

    from space_env.factory import build_dataset
    from space_env.factory.registry import register_bundle

    spec = registry_spec()
    first, other = tmp_path / "first", tmp_path / "other"
    assert build_dataset(spec, first)["ok"]
    db = tmp_path / "registry.sqlite"
    assert register_bundle(first, db)["status"] == "registered"
    assert register_bundle(first, db)["status"] == "already_registered"
    # A new manifest is not a new trajectory; same-fold duplicates are rejected.
    changed = copy.deepcopy(spec)
    changed["id"] = "new-label"
    assert build_dataset(changed, other)["ok"]
    with pytest.raises(ValueError, match="trajectory already registered"):
        register_bundle(other, db)
    changed["templates"][0]["split"] = "holdout"
    third = tmp_path / "third"
    assert build_dataset(changed, third)["ok"]
    with pytest.raises(ValueError, match="crosses registry splits"):
        register_bundle(third, db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM bundles").fetchone()[0] == 1
    with pytest.raises(ValueError, match="outside"):
        register_bundle(first, first / "bad.sqlite")


def test_registry_foreign_db_not_overwritten(tmp_path):
    import sqlite3

    from space_env.factory import build_dataset
    from space_env.factory.registry import register_bundle

    bundle, registry = tmp_path / "bundle", tmp_path / "foreign"
    assert build_dataset(registry_spec(), bundle)["ok"]
    with sqlite3.connect(registry) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('preserved')")
    with pytest.raises(ValueError, match="registry"):
        register_bundle(bundle, registry)
    with sqlite3.connect(registry) as conn:
        assert conn.execute("SELECT * FROM unrelated").fetchall() == [("preserved",)]


def test_scale_manifest_contract():
    from space_env.scaling.benchmark import ScaleSpec

    w = {
        "id": "w",
        "family": "unit",
        "source": "synthetic fixture",
        "scenario": scenario(),
        "task": task(),
    }
    value = ScaleSpec.model_validate({"id": "m", "workloads": [w]})
    assert value.workloads[0].repeats == 1
    with pytest.raises(ValueError):
        ScaleSpec.model_validate({"id": "m", "workloads": [w, w]})
    with pytest.raises(ValueError):
        ScaleSpec.model_validate({"id": "m", "workloads": [{**w, "repeats": True}]})


def test_scale_isolated_receipt_and_failure_denominator(tmp_path):
    from space_env.scaling.benchmark import run_scale

    w = {
        "id": "w",
        "family": "unit",
        "source": "synthetic fixture",
        "scenario": scenario(),
        "task": {**task(), "runtime_audit": "compact"},
        "policy": {"name": "priority"},
    }
    bad = {**w, "id": "bad", "scenario": {"invalid": True}}
    result = run_scale({"id": "m", "workloads": [w, bad]}, tmp_path / "result")
    assert not result["ok"] and len(result["rows"]) == 2
    assert result["rows"][0]["passed"] and result["rows"][0]["peak_rss_bytes"] > 0
    assert result["rows"][1]["error"] is not None
    assert (tmp_path / "result/manifest.json").is_file()
    assert json.loads((tmp_path / "result/result.json").read_text()) == result


def test_parallel_python310_fails_with_explicit_contract(monkeypatch):
    from space_env.planning import benchmark

    monkeypatch.setattr(benchmark, "version_info", (3, 10))
    manifest = {
        "id": "compat",
        "cases": [
            {"id": "fixed", "family": "unit", "scenario": scenario(), "task": task()}
        ],
        "policies": [{"name": "priority"}],
    }
    assert benchmark.BenchmarkSpec.model_validate(manifest).workers == 1
    with pytest.raises(ValueError, match="Python 3.11"):
        benchmark.BenchmarkSpec.model_validate({**manifest, "workers": 2})
