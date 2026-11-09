"""Sensor-aware task integration: estimates, privacy, native delivery and replay."""

import copy
import json

import numpy as np
import pytest

from space_env.llm.episode import LLMEpisode, export_training, verify_record
from space_env.llm.ollama import wire_context
from space_env.navigation.projection import NAVIGATION_FEATURES, project_navigation
from space_env.navigation.task_suite import compile_suite, execute
from space_env.simulation.snapshot import digest
from space_env.tasks import MissionTaskEnv, TaskSession, TaskSpec
from space_env.tasks.observation import FEATURES


@pytest.fixture(scope="module")
def cases():
    return {c["id"]: c for c in compile_suite()}


def make(cases, name="nominal"):
    c = copy.deepcopy(cases[name])
    return c, TaskSession(c["scenario"], c["task"])


def test_role_sees_only_own_estimated_orbit_and_declared_actions(cases):
    _c, s = make(cases)
    v = s.observe("a")
    assert set(v["navigation"]) == {"a"}
    assert v["state_source"] == "native-sensor-estimate/1"
    actor = next(e for e in v["entities"] if e["kind"] == "actor")
    truth = s._runtime._backend.mission.ephemerides["a"].state(0.0)
    estimate = v["navigation"]["a"]["estimate_si"]
    assert actor["id"] == "a"
    assert actor["values"]["r_x_m"] == estimate[0] and estimate[0] != truth[0]
    assert not {"range_m", "q_w", "mass_kg"} & actor["values"].keys()
    assert all(e["kind"] in {"actor", "product"} for e in v["entities"])
    assert all(r["kind"] == "owns" for r in v["relations"])
    assert v["action_mask"] == [1, 1]
    assert all(r["reason"] == "declarative" for r in v["catalogue"])
    assert v["binding"] == digest({k: x for k, x in v.items() if k != "binding"})
    assert "true_bias" not in json.dumps(v)
    assert s.modeling_profile()["actors"]["a"]["state_estimation"] == "enabled"


def test_masks_do_not_reveal_true_contact(cases):
    _, yes = make(cases)
    _, no = make(cases, "no-contact")
    assert yes.observe("a")["action_mask"] == no.observe("a")["action_mask"] == [1, 1]
    assert yes.observe("a")["catalogue"] == no.observe("a")["catalogue"]


def test_allowlist_drops_new_geometry_fields_instead_of_forwarding(cases):
    _c, s = make(cases)
    raw = s.observe("a")
    raw["entities"][0]["values"]["SECRET_GEOMETRY"] = 12345.0
    raw["entities"].append(
        {
            "id": "hidden-link",
            "key": "link:hidden-link",
            "kind": "link",
            "index": 100,
            "values": {"range_m": 987654.0},
        }
    )
    out = project_navigation(raw, s._task, "a", raw["navigation"])
    assert "SECRET_GEOMETRY" not in json.dumps(out) and "hidden-link" not in json.dumps(
        out
    )
    assert [e["index"] for e in out["entities"]] == list(range(len(out["entities"])))


@pytest.mark.parametrize(
    "name,success", [("nominal", True), ("delayed-drop", True), ("no-contact", False)]
)
def test_native_delivery_sensor_replay_and_training_gate(
    cases, tmp_path, name, success
):
    record = execute(cases[name], "reference", split="train")
    assert record["metrics"]["success"] is success
    assert record["metrics"]["sft_eligible"] is success
    assert verify_record(record)["verified"]
    assert export_training(record, tmp_path / "rl.jsonl")["rows"] == 1
    if success:
        assert (
            export_training(record, tmp_path / "sft.jsonl", format="sft")["rows"] == 1
        )
        text = (tmp_path / "sft.jsonl").read_text()
        assert "true_bias" not in text and "initial_estimate_si" not in text
        assert "native-sensor-estimate/1" in text
    else:
        with pytest.raises(ValueError):
            export_training(record, tmp_path / "sft.jsonl", format="sft")


def test_idempotency_restore_does_not_duplicate_sensor_samples(cases):
    _c, s = make(cases, "delayed-drop")
    row = s.step(
        {"a": ["wait"]}, expected_revision=0, idempotency_key="first", advance_sec=2.0
    )
    before = s._navigation.measurement_history("a")
    assert (
        s.step(
            {"a": ["wait"]},
            expected_revision=0,
            idempotency_key="first",
            advance_sec=2.0,
        )
        == row
    )
    assert s._navigation.measurement_history("a") == before
    restored = TaskSession.from_checkpoint(s.checkpoint(compact=True))
    assert restored.observe() == s.observe()
    assert restored._navigation.measurement_history("a") == before
    assert (
        s.observe("a")["navigation"]["a"]["received_samples"][-1]["received_sec"] <= 2.0
    )


def test_filter_failure_is_atomic_with_native_executor(cases, monkeypatch):
    _c, s = make(cases)
    before = s.checkpoint(compact=True)

    def fail(runtime, filters):
        filters["a"]["history"].append({"corruption": True})
        raise ValueError("injected filter failure")

    monkeypatch.setattr(s._navigation, "_consume", fail)
    with pytest.raises(ValueError, match="injected filter failure"):
        s.step(
            {"a": ["wait"]}, expected_revision=0, idempotency_key="bad", advance_sec=1.0
        )
    assert s.checkpoint(compact=True) == before and s.revision == 0


def test_gym_full_covariance_and_missing_packet_masks(cases):
    c = cases["delayed-drop"]
    env = MissionTaskEnv(c["scenario"], c["task"])
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["navigation_feature_names"] == list(NAVIGATION_FEATURES)
    indices = np.flatnonzero(obs["navigation_mask"])
    assert len(indices) == 2
    i = indices[0]
    names = info["navigation_feature_names"]
    assert obs["navigation_feature_mask"][i, names.index("p_0_0")] == 1
    assert obs["navigation_feature_mask"][i, names.index("last_z_0")] == 0
    assert obs["navigation"][i, names.index("sample_count")] == 0
    actor = next(
        e
        for e in env.session.observe()["entities"]
        if e["kind"] == "actor" and e["id"] == "a"
    )
    assert (
        obs["entities"][actor["index"], FEATURES.index("r_x_m")]
        == c["task"]["navigation_sensors"]["a"]["initial_estimate_si"][0]
    )
    env.close()


def test_pettingzoo_has_own_navigation_only(cases):
    from space_env.tasks.parallel import MissionParallelEnv

    c = cases["nominal"]
    env = MissionParallelEnv(c["scenario"], c["task"])
    obs, _ = env.reset()
    for role, value in obs.items():
        assert env.observation_space(role).contains(value)
        assert sum(value["navigation_mask"]) == 1
        assert set(env.session.observe(role)["navigation"]) == {role}
    obs, _, _, _, _ = env.step({"a": 0, "b": 0})
    assert all(env.observation_space(a).contains(v) for a, v in obs.items())
    env.close()


def test_llm_brief_keeps_covariance_without_secret_sensor_config(cases):
    c, s = make(cases)
    ep = LLMEpisode(s, task_text=c["task_text"], group="sensor", split="test")
    messages, _, _ = wire_context(
        ep.messages, ep.tool_schemas(), pairs=2, profile="brief_v1"
    )
    text = json.dumps(messages)
    assert "covariance_si" in text and "true_bias" not in text
    assert "initial_covariance_si" not in text and "error_si" not in text


@pytest.mark.parametrize(
    "mode",
    [
        "missing_actor",
        "maneuver",
        "injection",
        "goal_errors",
        "pointing",
        "decision_budget",
    ],
)
def test_unsupported_sensor_combinations_rejected(cases, mode):
    c = copy.deepcopy(cases["nominal"])
    t = c["task"]
    if mode == "missing_actor":
        del t["navigation_sensors"]["b"]
    if mode == "maneuver":
        t["actions"][0]["commands"][0] = {
            "command_id": "burn",
            "actor_id": "a",
            "kind": "finite_burn",
            "duration_sec": 1.0,
            "burn": {
                "frame": "GCRS",
                "direction_gcrs": [0.0, 0.0, 1.0],
                "thrust_n": 0.5,
            },
        }
    if mode == "injection":
        t["actions"] = [
            {
                "id": "event",
                "role": "a",
                "kind": "inject",
                "event": {
                    "type": "transmitter_rate",
                    "payload": {"actor": "a", "rate_bps": 1.0},
                },
            }
        ]
    if mode == "goal_errors":
        t["goal"]["conditions"] = [
            {
                "id": "physical",
                "kind": "orbit_state",
                "actor": "a",
                "at_sec": 3.0,
                "position_m": [-7e6, 0.0, 0.0],
                "velocity_m_s": [0.0, 0.0, -7500.0],
                "position_tolerance_m": 1.0,
                "velocity_tolerance_m_s": 0.1,
            }
        ]
    if mode == "pointing":
        t["pointing_masks"] = True
    if mode == "decision_budget":
        t["max_decisions"] = 513
    with pytest.raises(ValueError):
        TaskSpec.model_validate(t)


@pytest.mark.parametrize("mode", ["unknown", "other_actor", "duplicate", "blank"])
def test_product_goal_rejects_invalid_ownership_or_ids(cases, mode):
    c = copy.deepcopy(cases["nominal"])
    condition = c["task"]["goal"]["conditions"][0]
    if mode == "unknown":
        condition["product_ids"] = ["absent"]
    if mode == "other_actor":
        condition["actor"] = "b"
    if mode == "duplicate":
        condition["product_ids"] = ["ready-data", "ready-data"]
    if mode == "blank":
        condition["product_ids"] = [" "]
    with pytest.raises(ValueError):
        TaskSession(c["scenario"], c["task"])


def test_legacy_task_does_not_gain_sensor_fields(cases):
    c = copy.deepcopy(cases["nominal"])
    c["task"].pop("navigation_sensors")
    c["task"].pop("model_requirements")
    s = TaskSession(c["scenario"], c["task"])
    assert "navigation_sensors" not in s._task.model_dump(mode="json", by_alias=True)
    assert "navigation" not in s.observe("a")
    env = MissionTaskEnv(c["scenario"], c["task"])
    obs, _ = env.reset()
    assert "navigation" not in obs
    env.close()


def test_delayed_initial_view_is_independent_of_hidden_orbit_coordinates(cases):
    c = copy.deepcopy(cases["delayed-drop"])
    first = TaskSession(c["scenario"], c["task"]).observe("a")
    c["scenario"]["definition"]["orbits"]["a"]["initial"]["r_m"][0] -= 1000.0
    other = TaskSession(c["scenario"], c["task"]).observe("a")
    assert first == other
    assert first["navigation"]["a"]["received_samples"] == []


def test_late_product_is_not_successful_delivery(cases):
    c = copy.deepcopy(cases["nominal"])
    c["scenario"]["definition"]["operations"]["products"][0]["deadline_sec"] = 0.1
    record = execute(c, "reference", split="train")
    assert not record["metrics"]["success"] and not record["metrics"]["sft_eligible"]
    assert not record["metrics"]["goal"]["success"]
