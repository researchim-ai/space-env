"""Physical regime separation, causal choice and native execution checks."""

import json

import pytest

from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.missions import make_episode
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.industrial.phasing_corpus import corpus_manifest, run_case
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks.phasing_generation import RANGES, generated_phasing_recipe


@pytest.mark.parametrize("split", ["train", "development", "test"])
def test_recipe_deterministic_and_two_choices_per_actor(split):
    config = {"split": split, "seed": 13}
    first = generated_phasing_recipe(2, config)
    assert first == generated_phasing_recipe(2, config)
    assert first["generation"]["horizon_sec"] in RANGES[split]["horizons"]
    assert len(first["task"]["actions"]) == 4
    assert len(first["reference_steps"][0]["action_ids"]) == 2
    lower, upper = RANGES[split]["thrust_centinewtons"]
    for p in first["generation"]["parameters"]:
        assert lower / 100 <= p["thrust_n"] <= upper / 100
    assert all(
        c["at_sec"] == first["generation"]["horizon_sec"]
        for c in first["task"]["goal"]["conditions"]
    )


def test_disjoint_regimes_manifest_is_owned_and_no_silent_filtering():
    manifest = corpus_manifest(samples_per_split=1)
    assert len(manifest["cases"]) == 3
    assert len({r["generation"]["physics_sha256"] for r in manifest["cases"]}) == 3
    assert manifest["sha256"] == digest(
        {k: v for k, v in manifest.items() if k != "sha256"}
    )
    horizons = [set(RANGES[s]["horizons"]) for s in RANGES]
    assert all(
        not a.intersection(b) for i, a in enumerate(horizons) for b in horizons[i + 1 :]
    )
    manifest["regimes"]["train"]["horizons"] = ()
    assert RANGES["train"]["horizons"]
    for invalid in (0, 33, True):
        with pytest.raises(ValueError):
            corpus_manifest(samples_per_split=invalid)


@pytest.mark.parametrize(
    "generation",
    [
        {"split": "unknown", "seed": 0},
        {"split": "test", "seed": True},
        {"split": "train", "seed": -1},
    ],
)
def test_strict_distribution_config(generation):
    with pytest.raises(ValueError):
        OrbitSession({"generation": generation})


def test_public_solver_selects_reference_without_private_schedule():
    recipe = generated_phasing_recipe(2, {"split": "test", "seed": 2})
    session = OrbitSession({"generation": {"split": "test", "seed": 2}})
    action = orbit_policy(session.observe())
    assert set(action["action_ids"]) == set(recipe["reference_steps"][0]["action_ids"])
    episode = make_episode("orbit_and_deployment/generated_test", max_calls=1)
    prompt = json.loads(episode.record()["messages"][1]["content"])
    assert "generation" not in prompt and "reference_steps" not in canonical_json(
        prompt
    )
    assert "seed" not in prompt["observation"] and "split" not in prompt["observation"]


def test_correct_policy_executes_and_train_export_contains_no_private_recipe(tmp_path):
    case = {"configuration": {"actors": 2, "generation": {"split": "train", "seed": 0}}}
    record = run_case(case, "public")
    assert verify_episode(record)["success"]
    assert record["metrics"]["native_rejections"] == 0
    output = tmp_path / "sft.jsonl"
    export_training(record, output, format="sft")
    row = json.loads(output.read_text())
    assert "generation" not in canonical_json(row["messages"])
    assert "initial" not in row and row["success"]


def test_wrong_directions_are_not_success_even_with_valid_actions():
    session = OrbitSession({"generation": {"split": "development", "seed": 0}})
    good = set(orbit_policy(session.observe())["action_ids"])
    wrong = [
        r["id"]
        for r in session.observe()["catalogue"]
        if r["id"] != "wait" and r["id"] not in good
    ]
    session.step({"action_ids": wrong}, expected_revision=0, idempotency_key="wrong")
    index = 1
    while not session._done:
        session.step({}, expected_revision=index, idempotency_key=str(index))
        index += 1
    assert (
        not session.metrics()["success"] and session.metrics()["native_rejections"] == 0
    )


def test_generation_provenance_cannot_be_relabelled_even_with_valid_tools():
    session = OrbitSession({"generation": {"split": "test", "seed": 0}})
    with pytest.raises(ValueError, match="provenance"):
        IndustrialEpisode(session, split="train")
    episode = make_episode("orbit_and_deployment/generated_test", max_calls=1)
    assert episode.record()["split"] == "test"
    with pytest.raises(ValueError, match="provenance"):
        make_episode("orbit_and_deployment/generated_test", split="train")


def test_generated_gym_inherits_distribution_split():
    from space_env.industrial.missions import make_env

    env = make_env("orbit_and_deployment/generated_development", max_calls=1)
    env.reset()
    assert env._episode.record()["split"] == "development"
    env.close()


def test_fixed_gym_accepts_explicit_null_generation():
    from space_env.industrial.gym import IndustrialEnv

    env = IndustrialEnv("native_orbits", {"actors": 2, "generation": None}, max_calls=1)
    env.reset()
    assert env._episode.record()["split"] == "train"
    env.close()
