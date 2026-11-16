import json

import pytest

from space_env.industrial import NativeProximitySession, OperationsSession
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.gym import IndustrialEnv
from space_env.industrial.native_proximity import native_proximity_recipe
from space_env.industrial.operations import operations_policy


def test_permanent_fault_policy_stops_reset_loop_and_keeps_working():
    session = OperationsSession(
        {
            "events": [
                {
                    "at_sec": 4.0,
                    "event": {
                        "type": "spacecraft_fault",
                        "payload": {"actor": "camera-00", "subsystem": "radio"},
                    },
                }
            ],
            "permanent_faults": [["camera-00", "radio"]],
        }
    )
    while not session._done:
        obs = session.observe()
        assert all(v["revision"] == obs["revision"] for v in obs["roles"].values())
        session.step(
            operations_policy(obs),
            expected_revision=session.revision,
            idempotency_key=str(session.revision),
        )
    assert session.metrics()["fulfilled_requests"] > 0
    assert session._reset_history["camera-00"]["radio"]["attempts"] == 1
    assert not session._reset_history["camera-00"]["radio"]["accepted"]
    assert NativeProximitySession.kind != OperationsSession.kind


def test_reset_attempt_budget_does_not_advance_native_time_on_invalid_call():
    session = OperationsSession(
        {
            "events": [
                {
                    "at_sec": 0.0,
                    "event": {
                        "type": "spacecraft_fault",
                        "payload": {"actor": "camera-00", "subsystem": "radio"},
                    },
                }
            ],
            "permanent_faults": [["camera-00", "radio"]],
        }
    )
    session.step(
        {"kind": "reset", "actor": "camera-00", "subsystem": "radio"},
        expected_revision=0,
        idempotency_key="first",
    )
    before = session.checkpoint()
    with pytest.raises(ValueError, match="budget"):
        session.step(
            {"kind": "reset", "actor": "camera-00", "subsystem": "radio"},
            expected_revision=1,
            idempotency_key="second",
        )
    assert session.checkpoint() == before


def test_new_native_profile_tools_and_training_preserve_public_boundary(tmp_path):
    episode = IndustrialEpisode(
        NativeProximitySession(native_proximity_recipe()),
        policy_origin="scripted_public",
    )
    for i in range(3):
        episode.call(
            str(i),
            json.dumps(
                {
                    "expected_revision": episode.observe()["revision"],
                    "idempotency_key": str(i),
                    "action": {"kind": "inspect"},
                }
            ),
        )
    record = json.loads(json.dumps(episode.record()))
    assert record["metrics"]["success"] and verify_episode(record)["valid"]
    path = tmp_path / "native-sft.jsonl"
    export_training(record, path, format="sft")
    text = path.read_text()
    assert "initial_estimate_si" not in text and "prefix_checkpoint" not in text
    assert "relative_truth_rtn_si" not in text


def test_native_gym_seed_changes_sensor_stream():
    cfg = native_proximity_recipe()
    env = IndustrialEnv("native_proximity", cfg)
    a, info = env.reset(seed=3)
    b, _ = env.reset(seed=4)
    assert a != b and info["seed_changes_scenario"]
    assert env.observation_space.contains(a)
