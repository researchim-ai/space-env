"""Continuous campaigns through the same public tools used by LLM/Gym agents."""

import copy
import json

import pytest

from examples.modeling.campaign import recipe
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.gym import IndustrialEnv
from space_env.industrial.session import restore


@pytest.fixture(scope="module")
def configuration():
    case, campaign = recipe()
    return {"scenario": case["scenario"], "task": case["task"], "campaign": campaign}


def act(session, choices, dt):
    return session.step(
        {"choices": {"a": choices}, "advance_sec": dt},
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def test_continuity_checkpoint_and_privacy(configuration):
    session = AgentCampaignSession(configuration)
    act(session, ["burn"], 0.5)
    restored = restore(session.checkpoint())
    assert act(session, ["wait"], 0.5) == act(restored, ["wait"], 0.5)
    act(session, ["deliver"], 2.0)
    before = session._native._session._runtime._backend.report()
    assert session.metrics()["completed_phases"] == ["deliver"]
    view = session.observe()
    assert "campaign" not in view and "goal" not in view["native"]
    assert "completed_phases" not in json.dumps(view)
    act(session, ["wait"], 1.0)
    after = session._native._session._runtime._backend.report()
    assert before["products"] == after["products"]
    assert (
        after["energy_ledger"]["a"]["consumed_wh"]
        >= before["energy_ledger"]["a"]["consumed_wh"]
    )
    assert session.metrics()["success"]
    assert restore(session.checkpoint()).checkpoint() == session.checkpoint()


def test_invalid_actions_atomic_and_idempotent(configuration):
    session = AgentCampaignSession(configuration)
    cp = session.checkpoint()
    for dt in [float("nan"), float("inf"), 0.0, 3601.0]:
        with pytest.raises(ValueError):
            act(session, ["wait"], dt)
        assert session.checkpoint() == cp
    response = act(session, ["burn"], 1.0)
    assert (
        session.step(
            {"choices": {"a": ["burn"]}, "advance_sec": 1.0},
            expected_revision=0,
            idempotency_key="0",
        )
        == response
    )
    with pytest.raises(ValueError, match="conflict"):
        session.step(
            {"choices": {"a": ["wait"]}, "advance_sec": 1.0},
            expected_revision=0,
            idempotency_key="0",
        )


def test_episode_replay_and_training(configuration, tmp_path):
    episode = IndustrialEpisode(
        AgentCampaignSession(configuration),
        policy_origin="scripted_privileged",
        max_calls=8,
        task_text="Deliver the product, then retain it through the next phase.",
    )
    for index, (choices, dt) in enumerate(
        [(["burn"], 1.0), (["deliver"], 2.0), (["wait"], 1.0)]
    ):
        episode.call(
            str(index),
            json.dumps(
                {
                    "expected_revision": index,
                    "idempotency_key": str(index),
                    "action": {"choices": {"a": choices}, "advance_sec": dt},
                }
            ),
        )
    record = episode.record()
    assert verify_episode(record)["success"]
    export_training(record, tmp_path / "sft.jsonl", format="sft")
    export_training(record, tmp_path / "rl.jsonl", format="rl")
    row = json.loads((tmp_path / "sft.jsonl").read_text())
    assert "scenario" not in row and "final" not in row
    assert row["policy_origin"] == "scripted_privileged"


def test_deadline_failure_and_gym_budget(configuration):
    session = AgentCampaignSession(configuration)
    result = act(session, ["wait"], 5.0)
    assert result["elapsed_sec"] == 4.0
    assert session.metrics()["failed"] and not session.metrics()["success"]
    env = IndustrialEnv("native_campaign", configuration, max_calls=1)
    observation, _ = env.reset(seed=2)
    assert env.observation_space.contains(observation)
    observation, reward, terminal, truncated, info = env.step("invalid JSON")
    assert truncated and not terminal and reward == -0.1
    assert env.observation_space.contains(observation)
    assert not info["trainer_only"]["success"]


def test_no_environment_injections(configuration):
    config = copy.deepcopy(configuration)
    config["task"]["actions"].append(
        {
            "id": "unsafe",
            "role": "a",
            "kind": "inject",
            "event": {"type": "spacecraft_recover", "payload": {"actor": "a"}},
        }
    )
    with pytest.raises(ValueError):
        AgentCampaignSession(config)
