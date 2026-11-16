"""Family dispatch, causal prompts, replay and trainer/agent boundaries."""

import json

import pytest

from space_env.industrial.agent import (
    IndustrialEpisode,
    export_training,
    verify_episode,
)
from space_env.industrial.missions import (
    baseline_action,
    make_env,
    make_episode,
    make_session,
    mission_families,
    mission_recipe,
    run_mission,
)
from space_env.industrial.proximity import ProximitySession
from space_env.simulation.snapshot import canonical_json, digest


def test_discovery_is_lazy_detached_and_distinguishes_roadmap(monkeypatch):
    import space_env.industrial.native_service as service

    monkeypatch.setattr(
        service,
        "native_service_recipe",
        lambda *a: pytest.fail("constructed recipe during discovery"),
    )
    catalog = mission_families()
    assert len(catalog["families"]) == 5 and len(catalog["missions"]) == 51
    ids = [row["id"] for row in catalog["missions"]]
    assert len(set(ids)) == len(ids)
    assert not any(row["id"] in ids for row in catalog["roadmap"])
    assert all(not value for value in catalog["qualification"].values())
    assert "configuration" not in canonical_json(catalog)
    catalog["families"][0]["profiles"].clear()
    assert mission_families()["families"][0]["profiles"]


@pytest.mark.parametrize(
    "name",
    [
        "unknown",
        "../nominal",
        "orbit_and_deployment/nominal",
        "earth_observation",
        "communications/arbitrary",
    ],
)
def test_unknown_or_unavailable_missions_cannot_silently_fall_back(name):
    with pytest.raises(ValueError, match="unknown or unavailable"):
        make_session(name)


def test_family_brief_does_not_disclose_event_variant():
    for family, profiles in [
        ("earth_observation", ["nominal", "new_demand", "station_outage"]),
        ("fault_management", ["resettable", "permanent", "external_repair"]),
    ]:
        recipes = [mission_recipe(family + "/" + p) for p in profiles]
        assert len({r["task_text"] for r in recipes}) == 1
        assert len({r["sha256"] for r in recipes}) == len(recipes)
        for recipe in recipes:
            assert recipe["sha256"] == digest(
                {k: v for k, v in recipe.items() if k != "sha256"}
            )
    early = make_episode("earth_observation/new_demand", max_calls=1)
    prompt = json.loads(early.record()["messages"][1]["content"])
    assert "task_text" in prompt
    assert "deferred_request_at_sec" not in canonical_json(prompt)
    assert "configuration" not in prompt and "sources" not in prompt


@pytest.mark.parametrize("value", ["", " ", 1, "x" * 8193])
def test_task_brief_is_bounded_public_text(value):
    with pytest.raises(ValueError, match="task_text"):
        IndustrialEpisode(ProximitySession(), task_text=value)


def test_custom_brief_survives_replay_and_verified_sft_export(tmp_path):
    episode = IndustrialEpisode(
        ProximitySession(),
        task_text="Inspect the declared waypoint.",
        policy_origin="scripted_public",
    )
    for index in range(3):
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": index,
                    "idempotency_key": str(index),
                    "action": {"kind": "inspect"},
                }
            ),
        )
    record = json.loads(json.dumps(episode.record()))
    assert verify_episode(record)["success"]
    path = tmp_path / "sft.jsonl"
    export_training(record, path, format="sft")
    row = json.loads(path.read_text())
    assert json.loads(row["messages"][1]["content"])["task_text"] == record["task_text"]
    assert "initial" not in row and "configuration" not in row
    record["messages"][1]["content"] = "{}"
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay"):
        verify_episode(record)


@pytest.mark.parametrize(
    "mission",
    [
        "earth_observation/new_demand",
        "communications/nominal",
        "proximity/latency",
        "fault_management/permanent",
    ],
)
def test_common_gym_goal_available_and_call_budget_enforced(mission):
    env = make_env(mission, max_calls=1)
    observation, _ = env.reset(seed=9)
    assert env.observation_space.contains(observation)
    value = json.loads(observation)
    assert value["task_text"] == mission_recipe(mission)["task_text"]
    kind = next(r["kind"] for r in mission_families()["missions"] if r["id"] == mission)
    result = env.step(
        canonical_json(
            {
                "expected_revision": value["revision"],
                "idempotency_key": "step",
                "action": baseline_action(kind, value),
            }
        )
    )
    assert result[2] or result[3]
    assert json.loads(result[0])["task_text"] == value["task_text"]
    assert "trainer_only" not in json.loads(result[0])
    assert verify_episode(env._episode.record())["valid"]
    env.close()


def test_full_native_relative_mission_uses_same_tool_and_replays():
    record = run_mission("proximity/latency", policy="public")
    assert record["metrics"]["success"] and record["metrics"]["native_nonlinear_orbits"]
    assert record["task_text"]
    assert verify_episode(record)["valid"]


def test_recipe_mutation_does_not_change_future_missions():
    first = mission_recipe("fault_management/permanent")
    first["configuration"]["events"][0]["event"]["payload"]["actor"] = "invented"
    assert (
        mission_recipe("fault_management/permanent")["configuration"]["events"][0][
            "event"
        ]["payload"]["actor"]
        == "camera-00"
    )
