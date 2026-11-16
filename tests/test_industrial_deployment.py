import runpy
from pathlib import Path

import pytest

from space_env.tasks import TaskSession

EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/modeling/deployment.py")
)


@pytest.mark.parametrize("actors", [True, 1, 9])
def test_phasing_rejects_invalid_fleet_sizes(actors):
    with pytest.raises(ValueError):
        EXAMPLE["deployment_recipe"](actors)


def test_native_phasing_reference_succeeds_wait_fails():
    result = EXAMPLE["run"](2)
    assert not result["rows"][0]["record"]["metrics"]["success"]
    report = result["rows"][1]["record"]["metrics"]
    assert report["success"] and report["goal"]["completed"] == 2
    assert all(9.98 < r["propellant_kg"] < 10.0 for r in report["goal"]["resources"])


def test_one_wrong_burn_cannot_satisfy_fleet_goal():
    case = EXAMPLE["deployment_recipe"](2)
    command = case["task"]["actions"][0]["commands"][0]
    command["burn"]["direction_gcrs"] = [-x for x in command["burn"]["direction_gcrs"]]
    session = TaskSession(case["scenario"], case["task"])
    for index, step in enumerate(case["reference_steps"]):
        session.step(
            {"@central": step["action_ids"]},
            expected_revision=session.revision,
            idempotency_key=str(index),
            advance_sec=step["advance_sec"],
        )
    status = session.goal_status()
    assert not status["success"] and status["completed"] == 1
