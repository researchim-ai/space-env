"""Configuration requirements are enforced per actor without new truth exposure."""

import copy
import json
from typing import get_args

import numpy as np
import pytest

from space_env.reference.thermal_network import HEAT_SOURCES
from space_env.tasks import TaskSession, TaskSpec
from space_env.tasks.modeling import Feature, inspect_task, model_registry
from space_env.tasks.modeling_suite import compile_suite


@pytest.fixture(scope="module")
def cases():
    return {c["id"]: c for c in compile_suite()["cases"]}


def case(cases, name="impulse-coupled"):
    return copy.deepcopy(cases[name])


@pytest.mark.parametrize(
    "feature,name",
    [
        ("dynamic_orbit", "orbit-nominal"),
        ("spacecraft_bus", "attitude-nominal"),
        ("rigid_attitude", "attitude-rigid"),
        ("body_thrust", "impulse-chemical"),
        ("rotating_body_thrust", "impulse-coupled"),
        ("momentum_unloading", "momentum-nominal"),
    ],
)
def test_enabled_requirements_execute_and_replay(cases, feature, name):
    c = case(cases, name)
    c["task"]["model_requirements"] = [{"feature": feature, "actors": ["a"]}]
    s = TaskSession(c["scenario"], c["task"])
    assert s.modeling_profile()["actors"]["a"][feature] == "enabled"
    for i, step in enumerate(c["reference_steps"]):
        outcome = s.step(
            {"a": step["action_ids"]},
            expected_revision=i,
            idempotency_key=str(i),
            advance_sec=step["advance_sec"],
        )
        if outcome["terminated"]:
            break
    assert s.goal_status()["success"]
    restored = TaskSession.from_checkpoint(s.checkpoint(compact=True))
    assert restored.modeling_profile() == s.modeling_profile()
    assert restored.goal_status() == s.goal_status()
    assert restored._task.model_requirements == s._task.model_requirements


@pytest.mark.parametrize(
    "feature",
    [
        "navigation_measurements",
        "state_estimation",
        "continuous_relative_safety",
        "docking_contact",
    ],
)
def test_unimplemented_requirements_reject_instead_of_ideal_fallback(cases, feature):
    c = case(cases)
    c["task"]["model_requirements"] = [{"feature": feature, "actors": ["a"]}]
    with pytest.raises(ValueError, match="not_implemented|not_configured"):
        TaskSession(c["scenario"], c["task"])


@pytest.mark.parametrize(
    "feature",
    [
        "rigid_attitude",
        "body_thrust",
        "rotating_body_thrust",
        "momentum_unloading",
        "thermal_network",
        "covariance_prediction",
    ],
)
def test_other_actor_hardware_cannot_satisfy_requirement(cases, feature):
    c = case(cases)
    c["task"]["model_requirements"] = [{"feature": feature, "actors": ["b"]}]
    with pytest.raises(ValueError, match="not_configured"):
        TaskSession(c["scenario"], c["task"])


def test_partial_navigation_and_thermal_are_actor_specific(cases):
    c = case(cases, "attitude-rigid")
    d = c["scenario"]["definition"]
    d["orbits"]["a"]["navigation"] = {
        "frame": "GCRS",
        "initial_covariance_si": np.eye(6).tolist(),
    }
    d["spacecraft_bus"]["a"]["thermal_network"] = {
        "model_id": "synthetic-registry-test",
        "source": "test assumptions",
        "parameter_status": "synthetic",
        "nodes": {
            "hull": {
                "initial_k": 285.0,
                "capacity_j_per_k": 100.0,
                "trip_k": 330.0,
                "resume_k": 310.0,
            }
        },
        "links": [],
        "routing": {k: {"hull": 1.0} for k in HEAT_SOURCES},
        "earth_environment": False,
    }
    c["task"]["model_requirements"] = [
        {"feature": f, "actors": ["a"]}
        for f in ["thermal_network", "covariance_prediction"]
    ]
    profile = inspect_task(c["scenario"], c["task"])["profile"]["actors"]
    for feature in ["thermal_network", "covariance_prediction"]:
        assert profile["a"][feature] == "enabled"
        assert profile["b"][feature] == "not_configured"
    s = TaskSession(c["scenario"], c["task"])
    result = s.step(
        {"a": ["wait"]},
        expected_revision=0,
        idempotency_key="thermal-nav",
        advance_sec=0.1,
    )
    assert result["elapsed_sec"] == 0.1
    assert (
        TaskSession.from_checkpoint(s.checkpoint(compact=True)).modeling_profile()
        == s.modeling_profile()
    )


@pytest.mark.parametrize(
    "requirements",
    [
        [{"feature": "unknown", "actors": ["a"]}],
        [{"feature": "body_thrust", "actors": []}],
        [{"feature": "body_thrust", "actors": ["a", "a"]}],
        [{"feature": "body_thrust", "actors": ["missing"]}],
        [
            {"feature": "body_thrust", "actors": ["a"]},
            {"feature": "body_thrust", "actors": ["a"]},
        ],
        [{"feature": "body_thrust", "actors": [" "]}],
    ],
)
def test_malformed_requirements_rejected_by_recipe(cases, requirements):
    c = case(cases)
    c["task"]["model_requirements"] = requirements
    with pytest.raises(ValueError):
        TaskSpec.model_validate(c["task"])


def test_empty_requirements_preserve_legacy_task_json(cases):
    task = TaskSpec.model_validate(cases["orbit-nominal"]["task"])
    assert "model_requirements" not in task.model_dump(mode="json", by_alias=True)


def test_profile_is_read_only_and_never_inserted_in_actor_observation(cases):
    c = case(cases)
    s = TaskSession(c["scenario"], c["task"])
    before = s.checkpoint(compact=True)
    view = s.observe("a")
    p = s.modeling_profile()
    p["actors"]["a"]["rigid_attitude"] = "invented"
    assert s.modeling_profile()["actors"]["a"]["rigid_attitude"] == "enabled"
    assert s.checkpoint(compact=True) == before and s.observe("a") == view
    assert "modeling_profile" not in view
    assert not s._runtime.command_status()


def test_registry_and_schema_agree_and_cannot_be_mutated():
    a = model_registry()
    assert set(a["features"]) == set(get_args(Feature))
    a["features"].clear()
    assert model_registry()["features"]


def test_cli_registry_and_preflight(cases, tmp_path, monkeypatch):
    from space_env.tasks.modeling import main

    c = case(cases)
    scenario = tmp_path / "scenario.json"
    task = tmp_path / "task.json"
    out = tmp_path / "out.json"
    scenario.write_text(json.dumps(c["scenario"]))
    task.write_text(json.dumps(c["task"]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "models",
            "--scenario",
            str(scenario),
            "--task",
            str(task),
            "--output",
            str(out),
        ],
    )
    main()
    r = json.loads(out.read_text())
    assert r["valid"] and r["time_sec"] == 0
    monkeypatch.setattr("sys.argv", ["models", "--output", str(out)])
    main()
    assert json.loads(out.read_text()) == model_registry()
    monkeypatch.setattr(
        "sys.argv",
        [
            "models",
            "--scenario",
            str(scenario),
            "--task",
            str(task),
            "--output",
            str(task),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert json.loads(task.read_text()) == c["task"]
