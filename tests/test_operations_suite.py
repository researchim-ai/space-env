"""Frozen inputs, actor boundary, and actual native operations evidence."""

import copy
import json

import pytest

from space_env.llm.episode import STEP, LLMEpisode
from space_env.llm.operations_suite import (
    POLICIES,
    TASKS,
    audit,
    compile_suite,
    run_suite,
    validate_suite,
)
from space_env.tasks.session import TaskSession


@pytest.fixture(scope="module")
def suite():
    return compile_suite()


def test_frozen_inputs(suite):
    assert compile_suite() == suite
    assert len(suite["cases"]) == 5
    assert {c["family"] for c in suite["cases"]} == set(TASKS)
    changed = copy.deepcopy(suite)
    changed["cases"][0]["steps"][0]["advance_sec"] += 0.1
    with pytest.raises(ValueError, match="Frozen"):
        validate_suite(changed)


def test_actor_only(suite):
    case = suite["cases"][0]
    episode = LLMEpisode(
        TaskSession(case["scenario"], case["task"]), task_text="Operate", group="test"
    )
    assert "@central" not in episode.observe()["roles"]
    result = episode.invoke(
        STEP,
        json.dumps(
            {
                "choices": {"operator": ["invented"]},
                "expected_revision": 0,
                "idempotency_key": "bad",
            }
        ),
        call_id="bad",
    )
    assert not result["ok"]
    assert result["observation"]["revision"] == 0


def test_native_matrix_and_replay(tmp_path):
    directory = tmp_path / "suite"
    report = run_suite(directory)
    assert len(report["rows"]) == 5 * len(POLICIES)
    assert not report["external_llm_measured"]
    for row in report["rows"]:
        if row["status"] == "completed":
            assert not row["export_qualification"]["training_exported"]
            assert not row["export_qualification"]["operation_expert_qualified"]
            assert row["export_qualification"]["split"] == "test"
    assert all(r["status"] == "completed" for r in report["rows"]), report["rows"]
    for row in report["rows"]:
        if row["policy"] == "wait":
            assert row["native"]["score"]["completion"] == 0
        elif row["family"] == "resource-conflict":
            assert row["native"]["score"]["completion"] == 3
            intervals = row["native"]["lifecycle"]["resource_lanes"]["intervals"]
            assert any(
                i["actors"]["a"]["radio_bits"] > 0
                and len(i["actors"]["a"]["managed"]) == 2
                for i in intervals
            )
        elif row["family"] == "comms-recovery":
            assert row["native"]["score"]["completion"] == 2
            assert any(
                t["status"] == "cancelled"
                for t in row["native"]["dtn"]["transfers"].values()
            )
        else:
            operations = row["native"]["lifecycle"]["unloading"]["operations"]
            op = next(iter(operations.values()))
            assert op["consumed_propellant_kg"] > 0
            assert op["status"] == (
                "completed" if row["case"] == "unload-nominal" else "failed"
            )
    assert audit(directory) == {"valid": True, "rows": 10, "replayed": 10}


def test_unload_goal_uses_native_residual():
    from space_env.llm.operations_suite import operational_goal

    evidence = {
        "time_sec": 3.0,
        "lifecycle": {
            "unloading": {
                "operations": {
                    "native-command": {
                        "status": "completed",
                        "remaining_momentum_nms": [0.02, 0, 0],
                        "target_momentum_nms": [0, 0, 0],
                        "tolerance_nms": 1e-5,
                    }
                }
            }
        },
    }
    assert not operational_goal("momentum-unloading", evidence)
    evidence["lifecycle"]["unloading"]["operations"]["native-command"][
        "remaining_momentum_nms"
    ] = [0, 0, 0]
    assert operational_goal("momentum-unloading", evidence)
    evidence["time_sec"] = 4.0
    assert not operational_goal("momentum-unloading", evidence)
    evidence["time_sec"] = 3.0
    evidence["lifecycle"]["unloading"]["operations"] = {}
    assert not operational_goal("momentum-unloading", evidence)
