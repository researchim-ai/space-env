import copy
import json

import pytest

from space_env.industrial import ProximitySession
from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.gym import IndustrialEnv
from space_env.simulation.snapshot import digest


def call(episode, index, action):
    return episode.call(
        str(index),
        json.dumps(
            {
                "action": action,
                "expected_revision": episode.observe()["revision"],
                "idempotency_key": str(index),
            }
        ),
    )


def successful(split="train"):
    episode = IndustrialEpisode(
        ProximitySession(), policy_origin="scripted_public", split=split
    )
    for i in range(3):
        call(episode, i, {"kind": "inspect"})
    return episode


def test_successful_tools_replay_and_export_without_truth(tmp_path):
    record = json.loads(json.dumps(successful().record()))
    assert verify_episode(record)["valid"]
    for format in ("sft", "rl"):
        path = tmp_path / (format + ".jsonl")
        export_training(record, path, format=format)
        row = json.loads(path.read_text())
        assert row["success"] and row["return"] == 3
        assert "initial_truth_si" not in path.read_text()
        assert "initial" not in row and "final" not in row and "sources" not in row


def test_invalid_attempts_spend_budget_and_survive_replay():
    episode = IndustrialEpisode(
        ProximitySession(), max_calls=2, policy_origin="scripted_public"
    )
    assert episode.call("a", '{"expected_revision":0,"expected_revision":0}')["error"]
    assert episode.call("b", "{}")["error"]
    assert episode.done and episode.observe()["revision"] == 0
    record = episode.record()
    assert record["call_budget_exhausted"] and verify_episode(record)["valid"]


def test_model_cannot_multiply_reward_with_idempotent_retry():
    episode = IndustrialEpisode(ProximitySession(), policy_origin="scripted_public")
    payload = json.dumps(
        {
            "action": {"kind": "inspect"},
            "expected_revision": 0,
            "idempotency_key": "same",
        }
    )
    assert episode.call("a", payload)["reward"] == 1
    assert episode.call("b", payload)["error"] == "invalid_decision"
    assert episode.observe()["revision"] == 1


def test_agent_instructions_distinguish_retreat_goal_from_aborted_inspection():
    episode = IndustrialEpisode(ProximitySession({"mode": "retreat"}))
    instruction = episode.record()["messages"][0]["content"]
    assert "does not complete an inspection" in instruction
    assert "retreat task, a safe retreat is the objective" in instruction


def test_resealed_forged_metric_and_message_rejected():
    for field in ("metrics", "messages"):
        record = copy.deepcopy(successful().record())
        if field == "metrics":
            record[field]["success"] = False
        else:
            record[field][0]["content"] = "invented transcript"
        record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
        with pytest.raises(ValueError, match="replay"):
            verify_episode(record)


@pytest.mark.parametrize("split", ["test", "development"])
def test_evaluation_split_cannot_be_exported_for_training(tmp_path, split):
    with pytest.raises(ValueError, match="train split"):
        export_training(successful(split).record(), tmp_path / "bad.jsonl")


def test_failed_sft_rejected_but_rl_retains_failure(tmp_path):
    episode = IndustrialEpisode(
        ProximitySession({"horizon_steps": 1}), policy_origin="scripted_public"
    )
    call(episode, 0, {"kind": "coast"})
    record = episode.record()
    with pytest.raises(ValueError, match="SFT"):
        export_training(record, tmp_path / "sft.jsonl", format="sft")
    assert export_training(record, tmp_path / "rl.jsonl")["rows"] == 1


def test_gym_same_contract_and_invalid_call_truncation():
    env = IndustrialEnv("proximity", {}, max_calls=2)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step("{}")
    assert reward == -0.1 and not terminated and truncated
    assert "trainer_only" in info
    with pytest.raises(ValueError, match="finished"):
        env.step("{}")


@pytest.mark.parametrize("kind", ["operations", "proximity", "service"])
def test_tool_json_schema_references_resolve(kind):
    jsonschema = pytest.importorskip("jsonschema")
    from space_env.industrial import OperationsSession, ServiceSession
    from space_env.industrial.study import suite

    configuration = next(
        c["configuration"] for c in suite()["cases"] if c["kind"] == "service"
    )

    session = {
        "operations": lambda: OperationsSession(),
        "proximity": ProximitySession,
        "service": lambda: ServiceSession(configuration),
    }[kind]()
    episode = IndustrialEpisode(session)
    schema = episode.tool_schema()["parameters"]
    action = {"kind": "coast"} if kind == "proximity" else {}
    jsonschema.validate(
        {"expected_revision": 0, "idempotency_key": "a", "action": action}, schema
    )
