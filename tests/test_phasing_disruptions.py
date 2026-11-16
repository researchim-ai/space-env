"""Causal dynamic closures, native continuity, recovery and retained partial goals."""

import json

import pytest

from space_env.industrial.agent import verify_episode
from space_env.industrial.dispatch import subtract_interval
from space_env.industrial.missions import make_episode, run_mission
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.industrial.session import restore
from space_env.simulation.snapshot import canonical_json


def config(mode="recoverable"):
    return {
        "actors": 2,
        "generation": {
            "split": "train",
            "seed": 0,
            "windows": True,
            "disruptions": mode,
        },
    }


def advance(session, action=None):
    return session.step(
        orbit_policy(session.observe()) if action is None else action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def test_half_open_subtraction():
    assert subtract_interval([[0.0, 5.0], [10.0, 20.0]], 5.0, 10.0) == [
        [0.0, 5.0],
        [10.0, 20.0],
    ]
    assert subtract_interval([[0.0, 20.0]], 5.0, 10.0) == [[0.0, 5.0], [10.0, 20.0]]
    assert subtract_interval([[5.0, 10.0]], 5.0, 10.0) == []


def test_paired_profiles_identical_until_release_and_goals_never_retargeted():
    recover = OrbitSession(config())
    critical = OrbitSession(config("critical"))
    original = recover.observe()["objective"]
    assert recover.observe() == critical.observe()
    while not recover.observe()["dispatch_updates"]:
        assert recover.observe() == critical.observe()
        action = orbit_policy(recover.observe())
        advance(recover, action)
        advance(critical, action)
        assert canonical_json(
            recover._native.checkpoint(compact=True)
        ) == canonical_json(critical._native.checkpoint(compact=True))
    assert (
        recover.observe()["dispatch_updates"] != critical.observe()["dispatch_updates"]
    )
    assert recover.observe()["objective"] == critical.observe()["objective"] == original
    assert critical.observe()["planning_status"]["no_future_dispatch_windows"]
    clone = restore(json.loads(json.dumps(recover.checkpoint())))
    assert clone.observe() == recover.observe()
    while not recover._done:
        advance(recover)
        advance(clone)
        advance(critical)
    assert clone.metrics() == recover.metrics()
    assert recover.metrics()["success"]
    assert (
        not critical.metrics()["success"]
        and critical.metrics()["goal"]["completed"] == 1
    )
    assert recover.observe()["objective"] == original


def test_future_event_presence_does_not_change_public_prefix():
    plain = {"actors": 2, "generation": {"split": "train", "seed": 0, "windows": True}}
    future = {
        **plain,
        "dispatch_events": [{"at_sec": 20.0, "kind": "capacity", "capacity": 0}],
    }
    left, right = OrbitSession(plain), OrbitSession(future)
    assert left.observe() == right.observe()
    while left.observe()["time_sec"] < 15:
        action = orbit_policy(left.observe())
        advance(left, action)
        advance(right, action)
        assert left.observe() == right.observe()


def test_closing_windows_does_not_cancel_a_running_burn_or_reset_physics():
    plain = {"actors": 2, "generation": {"split": "train", "seed": 0, "windows": True}}
    changed = {
        **plain,
        "dispatch_events": [
            {
                "at_sec": 10.0,
                "kind": "close_windows",
                "actor": "sat-00",
                "start_sec": 10.0,
                "end_sec": 40.0,
            }
        ],
    }
    left, right = OrbitSession(plain), OrbitSession(changed)
    while not left._done:
        advance(left)
        advance(right)
        assert canonical_json(left._native.checkpoint(compact=True)) == canonical_json(
            right._native.checkpoint(compact=True)
        )
    assert left.metrics()["success"] and right.metrics()["success"]
    assert right.metrics()["released_dispatch_updates"] == 1


def test_rejected_action_still_releases_recovery_and_idempotency_does_not_repeat_it():
    value = {
        "actors": 2,
        "generation": {"split": "train", "seed": 0, "windows": True},
        "dispatch_events": [
            {"at_sec": 0.0, "kind": "capacity", "capacity": 0},
            {"at_sec": 5.0, "kind": "capacity", "capacity": 1},
        ],
    }
    session = OrbitSession(value)
    observation = session.observe()
    assert (
        observation["dispatch_capacity"] == 0
        and len(observation["dispatch_updates"]) == 1
    )
    name = next(r["id"] for r in observation["catalogue"] if r["id"] != "wait")
    result = session.step(
        {"action_ids": [name]}, expected_revision=0, idempotency_key="reject"
    )
    assert result["feedback"]["code"] == "dispatch_capacity"
    assert (
        result["observation"]["dispatch_capacity"] == 1
        and result["observation"]["time_sec"] == 5
    )
    assert len(result["observation"]["dispatch_updates"]) == 2
    assert (
        session.step(
            {"action_ids": [name]}, expected_revision=0, idempotency_key="reject"
        )
        == result
    )
    assert session.metrics()["released_dispatch_updates"] == 2
    assert not session.metrics()["launched_actors"]


@pytest.mark.parametrize(
    "event",
    [
        {"at_sec": 2.0, "kind": "capacity", "capacity": 0},
        {
            "at_sec": 10.0,
            "kind": "close_windows",
            "actor": "sat-00",
            "start_sec": 5.0,
            "end_sec": 20.0,
        },
        {
            "at_sec": 10.0,
            "kind": "close_windows",
            "actor": "unknown",
            "start_sec": 10.0,
            "end_sec": 20.0,
        },
        {"at_sec": 40.0, "kind": "capacity", "capacity": 1},
        {
            "at_sec": 10.0,
            "kind": "close_windows",
            "actor": "sat-00",
            "start_sec": 10.0,
            "end_sec": 50.0,
        },
    ],
)
def test_invalid_dispatch_timeline_rejected(event):
    with pytest.raises(ValueError):
        OrbitSession(
            {
                "generation": {"split": "train", "seed": 0, "windows": True},
                "dispatch_events": [event],
            }
        )


def test_conflicting_capacity_changes_and_nonwindowed_updates_rejected():
    with pytest.raises(ValueError, match="simultaneous"):
        OrbitSession(
            {
                "generation": {"split": "train", "seed": 0, "windows": True},
                "dispatch_events": [
                    {"at_sec": 5.0, "kind": "capacity", "capacity": 0},
                    {"at_sec": 5.0, "kind": "capacity", "capacity": 1},
                ],
            }
        )
    with pytest.raises(ValueError, match="published"):
        OrbitSession(
            {"dispatch_events": [{"at_sec": 5.0, "kind": "capacity", "capacity": 0}]}
        )
    with pytest.raises(ValueError, match="windowed"):
        OrbitSession(
            {"generation": {"split": "train", "seed": 0, "disruptions": "recoverable"}}
        )


def test_dynamic_episode_has_no_future_schedule_in_prompt_and_retains_provenance():
    episode = make_episode("orbit_and_deployment/dynamic_recoverable_test", max_calls=1)
    prompt = json.loads(episode.record()["messages"][1]["content"])
    assert not prompt["observation"]["dispatch_updates"]
    assert "private_timing" not in canonical_json(
        prompt
    ) and "dispatch_events" not in canonical_json(prompt)
    assert episode.record()["split"] == "test"
    with pytest.raises(ValueError, match="provenance"):
        make_episode("orbit_and_deployment/dynamic_critical_test", split="train")
    record = run_mission("orbit_and_deployment/dynamic_critical_train", policy="public")
    assert verify_episode(record)["valid"] and not record["metrics"]["success"]
