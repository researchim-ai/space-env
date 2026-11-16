"""Window timing, shared dispatch admission, native fuel and causal replay."""

import json

import pytest

from space_env.industrial.agent import verify_episode
from space_env.industrial.missions import make_episode, run_mission
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.industrial.session import restore
from space_env.tasks.phasing_generation import generated_phasing_recipe


def config(split="train", seed=0, actors=2):
    return {
        "actors": actors,
        "generation": {"split": split, "seed": seed, "windows": True},
    }


def test_delayed_schedule_is_constructive_with_one_dispatch_per_epoch():
    recipe = generated_phasing_recipe(4, config(actors=4)["generation"])
    launches = [
        (index * 5, step["action_ids"])
        for index, step in enumerate(recipe["reference_steps"])
        if step["action_ids"] != ["wait"]
    ]
    assert len(launches) == 4 and all(t > 0 and len(ids) == 1 for t, ids in launches)
    assert len({t for t, _ in launches}) == 4
    assert recipe == generated_phasing_recipe(4, config(actors=4)["generation"])
    for t, ids in launches:
        assert any(
            left <= t < right for left, right in recipe["launch_windows"][ids[0]]
        )
    assert all(len(windows) == 3 for windows in recipe["launch_windows"].values())


@pytest.mark.parametrize("actors", [1, 5, 8, True])
def test_windowed_scope_rejects_unsupported_fleets(actors):
    with pytest.raises(ValueError):
        OrbitSession(config(actors=actors))


def test_shared_dispatch_rejection_coasts_without_burning_and_is_idempotent():
    session = OrbitSession(config())
    before = session.metrics()["goal"]["resources"]
    by_actor = {}
    for row in session.observe()["catalogue"]:
        if row["id"] != "wait":
            by_actor.setdefault(row["commands"][0]["actor_id"], row["id"])
    assert len(by_actor) == 2
    actions = list(by_actor.values())
    value = {"action_ids": actions}
    result = session.step(value, expected_revision=0, idempotency_key="overbook")
    assert result["feedback"] == {"accepted": False, "code": "dispatch_capacity"}
    assert result["elapsed_sec"] == 5 and result["reward"] <= -0.1
    assert not session.metrics()["launched_actors"]
    assert [r["propellant_kg"] for r in session.metrics()["goal"]["resources"]] == [
        r["propellant_kg"] for r in before
    ]
    assert (
        session.metrics()["dispatch_rejections"] == 1
        and session.metrics()["native_rejections"] == 0
    )
    assert (
        session.step(value, expected_revision=0, idempotency_key="overbook") == result
    )
    assert session.revision == 1


def test_closed_half_open_window_admission_preserves_native_clock():
    session = OrbitSession(config())
    name = next(r["id"] for r in session.observe()["catalogue"] if r["id"] != "wait")
    windows = session.observe()["launch_windows"][name]
    closed = next(
        float(t)
        for t in (5, 10, 15, 20, 25)
        if not any(left <= t < right for left, right in windows)
    )
    while session.observe()["time_sec"] < closed:
        session.step(
            {},
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    row = next(r for r in session.observe()["catalogue"] if r["id"] == name)
    assert not row["enabled"] and row["window_reason"] == "launch_window_closed"
    result = session.step(
        {"action_ids": [name]},
        expected_revision=session.revision,
        idempotency_key="closed",
    )
    assert (
        result["feedback"]["code"] == "launch_window_closed"
        and result["elapsed_sec"] == 5
    )
    assert session.metrics()["time_sec"] == closed + 5
    restored = restore(json.loads(json.dumps(session.checkpoint())))
    assert (
        restored.observe() == session.observe()
        and restored.metrics() == session.metrics()
    )


def test_public_policy_matches_private_schedule_from_public_state_only():
    recipe = generated_phasing_recipe(2, config()["generation"])
    session = OrbitSession(config())
    for step in recipe["reference_steps"]:
        action = orbit_policy(session.observe())
        expected = [] if step["action_ids"] == ["wait"] else step["action_ids"]
        assert set(action["action_ids"]) == set(expected)
        session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    assert (
        session.metrics()["success"] and session.metrics()["dispatch_rejections"] == 0
    )
    assert len(session.metrics()["launched_actors"]) == 2


def test_launching_immediately_cannot_satisfy_delayed_targets():
    session = OrbitSession(config())
    while not session._done:
        observation = session.observe()
        options = [
            r["id"]
            for r in observation["catalogue"]
            if r["id"] != "wait" and r["enabled"]
        ]
        session.step(
            {"action_ids": options[:1]},
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    assert not session.metrics()["success"]


def test_already_launched_actor_cannot_spend_a_second_burn():
    session = OrbitSession(config())
    launched = None
    while launched is None:
        action = orbit_policy(session.observe())
        session.step(
            action,
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
        if action["action_ids"]:
            launched = action["action_ids"][0]
    result = session.step(
        {"action_ids": [launched]},
        expected_revision=session.revision,
        idempotency_key="again",
    )
    assert result["feedback"]["code"] == "already_launched"
    assert session.metrics()["dispatch_rejections"] == 1


def test_windowed_episode_cold_replay_and_provenance():
    record = run_mission("orbit_and_deployment/windowed_test", policy="public")
    assert record["split"] == "test" and verify_episode(record)["success"]
    prompt = json.loads(record["messages"][1]["content"])
    assert "private_timing" not in json.dumps(
        prompt
    ) and "reference_steps" not in json.dumps(prompt)
    with pytest.raises(ValueError, match="provenance"):
        make_episode("orbit_and_deployment/windowed_test", split="train")
