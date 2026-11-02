"""Independent orbit generation, resource envelope, repeated scale evidence."""

import copy

import pytest

from space_env.factory.quality import quality_report
from space_env.planning import PlanningRun
from space_env.scaling.benchmark import ScaleSpec, Workload, run_scale, summarize
from space_env.scaling.workloads import FleetWorkloadSpec, fleet_workload
from space_env.simulation.snapshot import digest


@pytest.mark.parametrize(
    "bad",
    [
        {"actors": 65},
        {"actors": True},
        {"products": 1024},
        {"products": 0},
        {"actors": 1, "planes": 2},
        {"horizon_sec": 1.0},
        {"horizon_sec": 21000.0, "decision_sec": 0.1},
        {"decision_sec": 3600.0, "actors": 1, "planes": 1},
        {"repeats": True},
    ],
)
def test_fleet_envelope_rejects_unsupported(bad):
    with pytest.raises(ValueError):
        FleetWorkloadSpec.model_validate(bad)


def test_distinct_planes_and_actor_states_are_deterministic():
    cfg = {"actors": 4, "products": 12, "planes": 2, "repeats": 1}
    a, b = fleet_workload(cfg), fleet_workload(cfg)
    assert digest(a) == digest(b)
    trajectories = a["scenario"]["definition"]["ephemerides"]
    assert len({e["sha256"] for e in trajectories.values()}) == 4
    # Propagated orbit payload hashes are independently integrity-bound.
    for e in trajectories.values():
        assert e["sha256"] == digest({k: v for k, v in e.items() if k != "sha256"})
    assert a["task"]["max_actions"] == 13
    assert sum(r["action_slots"] for r in a["task"]["roles"].values()) == 4
    assert a["task"]["max_entities"] >= 2 * 12 + 4 + 2


@pytest.mark.parametrize("mode,fulfilled", [("nominal", 9), ("ground_outage", 0)])
def test_independent_fleet_native_replay_and_ledgers(mode, fulfilled):
    w = fleet_workload(
        {"actors": 3, "products": 9, "planes": 2, "repeats": 1, "mode": mode}
    )
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    metrics = run.run()
    assert metrics["fulfilled"] == fulfilled
    assert metrics["total"] == 9
    restored = PlanningRun.from_checkpoint(run.checkpoint(compact=True))
    assert restored.trace == run.trace
    assert quality_report(restored)["passed"]


def summary_spec():
    return ScaleSpec.model_validate(
        {
            "id": "summary",
            "workloads": [
                {
                    "id": "w",
                    "family": "fixture",
                    "source": "synthetic",
                    "scenario": {},
                    "task": {},
                    "repeats": 2,
                }
            ],
        }
    )


def row(**kwargs):
    return {
        "id": "w",
        "passed": True,
        "error": None,
        "native_backend_sha256": "a",
        "trace_sha256": "b",
        "execution_sec": 2.0,
        "peak_rss_bytes": 100,
        **kwargs,
    }


@pytest.mark.parametrize(
    "rows,passes",
    [
        ([row(), row()], True),
        ([row()], False),
        ([row(), row(passed=False, error={"message": "timeout"})], False),
        ([row(), row(trace_sha256="changed")], False),
        ([], False),
    ],
)
def test_repeat_summary_retains_failures_and_nondeterminism(rows, passes):
    result = summarize(summary_spec(), rows)["w"]
    assert result["passed"] == passes
    assert result["observed_repeats"] == len(rows)
    assert result["expected_repeats"] == 2


def test_controlled_worker_threads_and_repeat_receipt(tmp_path):
    workload = fleet_workload({"actors": 2, "products": 4, "planes": 2, "repeats": 2})
    workload["worker_threads"] = 1
    result = run_scale({"id": "isolated", "workloads": [workload]}, tmp_path / "run")
    assert result["ok"]
    rows = result["rows"]
    assert len({r["worker_environment"]["pid"] for r in rows}) == 2
    assert all(
        set(r["worker_environment"]["thread_env"].values()) == {"1"} for r in rows
    )
    assert result["summary"][workload["id"]]["repeat_semantics_identical"]
    workload.update(repeats=1, max_step_p95_sec=1e-12)
    failed = run_scale(
        {"id": "latency-gate", "workloads": [workload]}, tmp_path / "failed"
    )
    assert not failed["ok"]
    assert not failed["rows"][0]["gates"]["step_latency_budget"]
    assert failed["rows"][0]["gates"]["native_replay"]
    assert failed["summary"][workload["id"]]["failed_repeats"] == 1


@pytest.mark.parametrize(
    "field", ["worker_threads", "max_step_p95_sec", "max_execution_sec"]
)
def test_measurement_budget_invalid(field):
    with pytest.raises(ValueError):
        Workload.model_validate(
            {
                "id": "x",
                "family": "unit",
                "source": "synthetic",
                "scenario": {},
                "task": {},
                field: 0,
            }
        )


def test_causal_masks_do_not_compute_native_outcome_reports(monkeypatch):
    from space_env.planning.model import make_input
    from space_env.tasks.observation import admissible

    w = fleet_workload({"actors": 2, "products": 4, "planes": 2, "repeats": 1})
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    runtime = run.session._runtime
    report = runtime.report()
    assert runtime.horizon_sec == report["horizon_sec"]
    assert runtime.command_status() == report["commands"]
    expected = make_input(run.session)

    def forbidden():
        raise AssertionError("outcome report in causal mask/input path")

    monkeypatch.setattr(runtime, "report", forbidden)
    assert make_input(run.session) == expected
    projection = copy.deepcopy(run.session._projections["@central"])
    assert (
        projection.build(runtime, "@central", 0, run.session._actions)
        == run.session.observe()
    )
    assert admissible(runtime, run.session._task, run.session._actions[0], "@central")[
        0
    ]


def test_lightweight_command_reader_owns_output():
    w = fleet_workload({"actors": 2, "products": 4, "planes": 2, "repeats": 1})
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    run.step()
    runtime = run.session._runtime
    assert runtime.command_status() == runtime.report()["commands"]
    owned = runtime.command_status()
    assert owned
    owned[0].clear()
    assert runtime.command_status()[0]


def test_64_actor_policy_budget_covers_native_joint_capacity():
    w = fleet_workload({"actors": 64, "products": 64, "planes": 4, "repeats": 1})
    assert w["policy"]["max_candidates"] == 64
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    run.step()
    assert run.metrics()["fulfilled"] == 64
    assert len(run.trace[0]["decision"]["action_ids"]) == 64
