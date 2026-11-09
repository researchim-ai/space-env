"""Relative-state goals: native dynamics, frames, privacy and negative evidence."""

import copy
import math

import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.relative import (
    closest_approach,
    relative_state_rtn,
    rtn_basis,
    rtn_state_to_eci,
)
from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, verify_record
from space_env.tasks import MissionGoal, MissionTaskEnv, TaskSession
from space_env.tasks.goals import GOAL_FEATURES, goal_columns
from space_env.tasks.modeling_suite import compile_suite


@pytest.fixture(scope="module")
def cases():
    return {c["id"]: c for c in compile_suite()["cases"]}


def test_rotating_frame_velocity_is_not_rotated_delta_v():
    radius, offset = 7e6, 100.0
    omega = math.sqrt(MU_EARTH / radius**3)
    r, v = np.array([radius, 0.0, 0.0]), np.array([0.0, radius * omega, 0.0])
    state = relative_state_rtn(r + [offset, 0, 0], v + [0, omega * offset, 0], r, v)
    np.testing.assert_allclose(state, [offset, 0, 0, 0, 0, 0], atol=1e-12)
    assert omega * offset > 0.1


@pytest.mark.parametrize("acceleration", [None, [0.1, -0.2, 0.3]])
def test_frame_round_trip_and_orthonormality(acceleration):
    r, v = np.array([7e6, 20.0, 40.0]), np.array([3.0, 7500.0, 100.0])
    state = np.array([50.0, -20.0, 5.0, 0.1, -0.3, 0.7])
    inertial = rtn_state_to_eci(state, r, v, reference_acceleration=acceleration)
    actual = relative_state_rtn(
        *[inertial[:3], inertial[3:], r, v], reference_acceleration=acceleration
    )
    np.testing.assert_allclose(actual, state, atol=1e-9)
    basis = rtn_basis(r, v)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-15)
    assert np.linalg.det(basis) == pytest.approx(1.0)


def test_frame_rate_matches_differentiated_noncentral_trajectory():
    r, v, a = (
        np.array([7e6, 0.0, 0.0]),
        np.array([0.0, 7500.0, 0.0]),
        np.array([-8.0, 0.0, 0.2]),
    )
    rho, dv = np.array([50.0, -30.0, 20.0]), np.array([0.2, -0.1, 0.3])
    actual = relative_state_rtn(r + rho, v + dv, r, v, reference_acceleration=a)[3:]
    h = 0.001

    def projected(t):
        chief = r + v * t + a * t * t / 2
        return rtn_basis(chief, v + a * t).T @ (rho + dv * t)

    np.testing.assert_allclose(
        actual, (projected(h) - projected(-h)) / (2 * h), atol=1e-8
    )


@pytest.mark.parametrize(
    "r,v",
    [
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0], [0.0, 1.0, 0.0]),
    ],
)
def test_degenerate_rtn_frames_rejected(r, v):
    with pytest.raises(ValueError):
        rtn_basis(r, v)


def test_closest_approach_does_not_look_beyond_horizon():
    state = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    distance, epoch, closest = closest_approach(state, 0.0, 0.75, 1.0)
    assert epoch == 0.75 and distance == 0.25
    np.testing.assert_allclose(closest, [0.25, 0, 0, -1, 0, 0])
    assert closest_approach(state, 0.0, 0.0, 1.0)[1] == 0.0


@pytest.mark.parametrize(
    "h,dt",
    [
        (1.0, 0.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (2e6, 1.0),
    ],
)
def test_invalid_sampling_rejected(h, dt):
    with pytest.raises(ValueError):
        closest_approach(np.zeros(6), 0.001, h, dt)


@pytest.mark.parametrize(
    "mode,success",
    [
        ("nominal", True),
        ("waypoints", True),
        ("passive", True),
        ("wrong-direction", False),
        ("target-mismatch", False),
        ("keepout", False),
    ],
)
def test_relative_native_execution_and_cold_replay(cases, mode, success):
    c = cases["relative-" + mode]
    s = TaskSession(c["scenario"], c["task"])
    ep = LLMEpisode(
        s,
        task_text=c["task_text"],
        group=c["id"],
        split="train",
        policy_origin="scripted_privileged",
    )
    chooser = reference_policy(
        c["reference_steps"], {a.id: a.role for a in s._task.actions}
    )
    record = run_agent(ep, chooser)
    assert record["metrics"]["success"] is success
    assert record["metrics"]["sft_eligible"] is success
    assert verify_record(record)["verified"]
    restored = TaskSession.from_checkpoint(ep._session.checkpoint(compact=True))
    assert restored.goal_status() == ep._session.goal_status()
    if mode == "keepout":
        assert ep._session.goal_status()["failures"] == [
            "relative_keepout:relative-state"
        ]
    if mode == "waypoints":
        assert ep._session.goal_status()["achieved"] == [
            "relative-cutoff",
            "relative-state",
        ]


def test_passive_goal_can_be_solved_without_maneuver(cases):
    c = cases["relative-passive"]
    s = TaskSession(c["scenario"], c["task"])
    ep = LLMEpisode(s, task_text=c["task_text"], group=c["id"], split="test")
    record = run_agent(ep, public_policy("wait", 1.0))
    assert record["metrics"]["success"]
    assert not ep._session._runtime.command_status()


def test_role_cannot_derive_hidden_reference_state(cases):
    c = copy.deepcopy(cases["relative-nominal"])
    c["task"]["roles"] = {"a": {"actors": ["a"]}, "b": {"actors": ["b"]}}
    with pytest.raises(ValueError, match="reference visibility"):
        TaskSession(c["scenario"], c["task"])


@pytest.mark.parametrize(
    "change",
    [
        {"reference_actor": "a"},
        {"reference_actor": "missing"},
        {"frame": "RTN"},
        {"at_sec": 6.0},
        {"position_m": [0.0, 0.0, 0.0], "min_range_m": 2.0},
        {"position_m": [float("nan"), 0.0, 0.0]},
    ],
)
def test_invalid_relative_recipes_rejected(cases, change):
    c = copy.deepcopy(cases["relative-nominal"])
    c["task"]["goal"]["conditions"][0].update(change)
    with pytest.raises(ValueError):
        TaskSession(c["scenario"], c["task"])


def test_gym_relative_features_reference_and_keepout(cases):
    c = cases["relative-nominal"]
    env = MissionTaskEnv(c["scenario"], c["task"])
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    names = info["goal_feature_names"]
    assert len(names) == len(GOAL_FEATURES) + 3
    assert obs["goals"][0, names.index("kind_code")] == 6
    assert obs["goals"][0, names.index("range_m")] == pytest.approx(100.0)
    assert obs["goals"][0, names.index("min_range_m")] == 50.0
    ref = int(obs["goals"][0, names.index("reference_index")])
    view = env.session.observe()
    assert any(e["id"] == "b" and e["index"] == ref for e in view["entities"])
    assert obs["goal_feature_mask"][0, names.index("reference_index")] == 1
    old = MissionGoal.model_validate(cases["orbit-nominal"]["task"]["goal"])
    assert goal_columns(old) == GOAL_FEATURES
    env.close()


def test_relative_waypoint_clipping_idempotency_and_resume(cases):
    c = cases["relative-waypoints"]
    s = TaskSession(c["scenario"], c["task"])
    first = s.step(
        {"a": ["control"]},
        expected_revision=0,
        idempotency_key="first",
        advance_sec=5.0,
    )
    assert first["elapsed_sec"] == 2.0 and first["reward"] == 1.0
    assert first["goal"]["achieved"] == ["relative-cutoff"]
    restored = TaskSession.from_checkpoint(s.checkpoint(compact=True))
    assert (
        restored.step(
            {"a": ["control"]},
            expected_revision=0,
            idempotency_key="first",
            advance_sec=5.0,
        )
        == first
    )
    last = restored.step(
        {"a": ["wait"]}, expected_revision=1, idempotency_key="last", advance_sec=5.0
    )
    assert last["elapsed_sec"] == 3.0 and last["reward"] == 1.0 and last["terminated"]
    assert last["goal"]["success"]


@pytest.mark.parametrize(
    "mode,success", [("nominal", True), ("wrong-direction", False)]
)
def test_relative_goals_gate_native_sft_and_keep_failed_rl(
    cases, tmp_path, mode, success
):
    from space_env.llm.episode import export_training

    c = cases["relative-" + mode]
    s = TaskSession(c["scenario"], c["task"])
    ep = LLMEpisode(
        s,
        task_text=c["task_text"],
        group=c["id"],
        split="train",
        policy_origin="scripted_privileged",
    )
    record = run_agent(
        ep,
        reference_policy(c["reference_steps"], {a.id: a.role for a in s._task.actions}),
    )
    assert export_training(record, tmp_path / "rl.jsonl")["rows"] == 1
    if success:
        assert (
            export_training(record, tmp_path / "sft.jsonl", format="sft")["rows"] == 1
        )
    else:
        with pytest.raises(ValueError):
            export_training(record, tmp_path / "sft.jsonl", format="sft")
