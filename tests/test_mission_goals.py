"""Physical mission objectives share native execution, replay and training gates."""

import copy
import json

import numpy as np
import pytest

from space_env.llm.benchmark import public_policy, reference_policy, run_agent
from space_env.llm.episode import (
    FINISH,
    STEP,
    LLMEpisode,
    export_training,
    seal,
    verify_record,
)
from space_env.llm.ollama import wire_context
from space_env.simulation.snapshot import canonical_json
from space_env.tasks import (
    MissionGoal,
    MissionTaskEnv,
    TaskSession,
    TaskSpec,
    VerifiedEpisode,
)
from space_env.tasks.goals import GOAL_FEATURES
from space_env.tasks.modeling_suite import compile_suite


@pytest.fixture(scope="module")
def cases():
    return {c["id"]: c for c in compile_suite()["cases"]}


def session(cases, name, mutate=None):
    case = copy.deepcopy(cases[name])
    if mutate:
        mutate(case)
    return TaskSession(case["scenario"], case["task"])


def run(cases, name, *, method="reference", mutate=None):
    c = cases[name]
    s = session(cases, name, mutate)
    ep = LLMEpisode(
        s,
        task_text=c["task_text"],
        group=name,
        split="train",
        max_calls=128,
        policy_origin="scripted_privileged"
        if method == "reference"
        else "scripted_public",
    )
    policy = (
        reference_policy(c["reference_steps"], {a.id: a.role for a in s._task.actions})
        if method == "reference"
        else public_policy("wait", s._task.decision_sec)
    )
    record = run_agent(ep, policy)
    assert verify_record(record)["verified"]
    return ep, record


@pytest.mark.parametrize(
    "name",
    [
        "attitude-nominal",
        "attitude-rigid",
        "momentum-nominal",
        "impulse-chemical",
        "impulse-electric",
        "impulse-coupled",
        "orbit-nominal",
        "orbit-waypoints",
        "delivery",
    ],
)
def test_native_physical_goals_complete_and_reward(cases, name):
    ep, r = run(cases, name)
    assert r["metrics"]["success"] and r["metrics"]["sft_eligible"]
    assert r["metrics"]["failed_calls"] == 0
    goal = r["metrics"]["goal"]
    assert goal["success"] and goal["completed"] == goal["total"]
    assert r["metrics"]["return"] == goal["total"]
    restored = TaskSession.from_checkpoint(ep._session.checkpoint(compact=True))
    assert restored.goal_status() == ep._session.goal_status()


@pytest.mark.parametrize(
    "name",
    [
        "attitude-wrong-direction",
        "attitude-rigid-loose",
        "momentum-limited",
        "orbit-wrong-direction",
    ],
)
def test_physical_negative_cases_never_become_positive_sft(cases, name, tmp_path):
    ep, r = run(cases, name)
    assert not r["metrics"]["success"] and not r["metrics"]["sft_eligible"]
    assert r["metrics"]["goal"]["terminal"]
    with pytest.raises(ValueError, match="SFT"):
        export_training(r, tmp_path / "sft.jsonl", format="sft")
    with pytest.raises(ValueError, match="SFT"):
        list(
            VerifiedEpisode(ep._session.checkpoint(compact=True)).iter_rows(
                format="sft"
            )
        )
    assert export_training(r, tmp_path / "rl.jsonl")["rows"] == 1


@pytest.mark.parametrize(
    "name",
    [
        "attitude-nominal",
        "attitude-rigid",
        "momentum-nominal",
        "impulse-chemical",
        "impulse-electric",
        "impulse-coupled",
        "orbit-nominal",
        "orbit-waypoints",
        "delivery",
    ],
)
def test_wait_is_a_measured_failure(cases, name):
    _, r = run(cases, name, method="wait")
    assert not r["metrics"]["success"]
    assert r["metrics"]["failed_calls"] == 0


def test_unloading_has_actual_non_delivery_sft_and_finish(cases, tmp_path):
    c = cases["momentum-nominal"]
    s = session(cases, "momentum-nominal")
    ep = LLMEpisode(s, task_text=c["task_text"], group="unload", split="train")
    assert not ep.invoke(FINISH, "{}", call_id="premature")["ok"]
    # The premature failed claim remains RL data, not an error-free SFT sample.
    args = canonical_json(
        {
            "choices": {"a": ["control"]},
            "expected_revision": 0,
            "idempotency_key": "unload",
            "advance_sec": 3.0,
        }
    )
    response = ep.invoke(STEP, args, call_id="physical")
    assert response["terminated"] and response["reward"] == 1.0
    r = ep.record()
    assert r["metrics"]["success"] and not r["metrics"]["sft_eligible"]
    assert r["metrics"]["fulfilled"] == 0 and r["metrics"]["total"] == 2
    good, record = run(cases, "momentum-nominal")
    assert export_training(record, tmp_path / "unload.jsonl", format="sft")["rows"] == 1
    rows = list(
        VerifiedEpisode(good._session.checkpoint(compact=True)).iter_rows(format="sft")
    )
    assert rows
    assert all(x["reward"] >= 0 for x in rows)
    finished = LLMEpisode(good._session, task_text="Verify unloading.", group="prefix")
    response = finished.invoke(FINISH, "{}", call_id="verify")
    assert response["ok"] and response["elapsed_sec"] == 0 and response["reward"] == 0
    assert verify_record(finished.record())["verified"]


def test_resource_failure_is_terminal_and_never_positive(cases):
    def tighter(c):
        c["task"]["goal"]["limits"]["a"]["max_energy_j"] = 0.0

    _, r = run(cases, "momentum-nominal", mutate=tighter)
    assert not r["metrics"]["success"]
    assert r["metrics"]["return"] < 0
    assert r["metrics"]["goal"]["failures"] == ["energy_budget:a"]


def test_fuel_reserve_is_native_not_declared_burn_intent(cases):
    def tighter(c):
        p = c["scenario"]["definition"]["orbits"]["a"]["propulsion"]
        c["task"]["goal"]["limits"]["a"]["min_propellant_kg"] = p["propellant_kg"]

    _, r = run(cases, "orbit-nominal", mutate=tighter)
    assert not r["metrics"]["success"]
    assert "fuel_reserve:a" in r["metrics"]["goal"]["failures"]


def test_waypoint_latches_survive_prefix_and_do_not_double_reward(cases):
    s = session(cases, "orbit-waypoints")
    first = s.step(
        {"a": ["control"]},
        expected_revision=0,
        idempotency_key="first",
        advance_sec=5.0,
    )
    assert first["elapsed_sec"] == 2.0  # Cannot jump over an intermediate epoch.
    assert first["goal"]["achieved"] == ["burn-cutoff"] and first["reward"] == 1.0
    checkpoint = s.checkpoint(compact=True)
    restored = TaskSession.from_checkpoint(checkpoint)
    assert restored.goal_status() == s.goal_status()
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
    assert last["elapsed_sec"] == 3.0 and last["terminated"]
    assert last["reward"] == 1.0 and last["discount"] == 0.0
    assert (
        last["goal"]["conditions"][0]["met"]
        and not last["goal"]["conditions"][0]["current_met"]
    )


def test_actor_projection_and_numeric_masks_hide_other_actors(cases):
    s = session(cases, "momentum-nominal")
    assert len(s.observe("a")["goal"]["conditions"]) == 1
    assert s.observe("b")["goal"]["conditions"] == []
    assert "success" not in s.observe("b")["goal"]
    ep = LLMEpisode(s, task_text="Unload.", group="g")
    messages, _, _ = wire_context(
        ep.messages, ep.tool_schemas(), 2, "json_fixed_v1", 1.0
    )
    roles = json.loads(messages[1]["content"])["roles"]
    assert roles["b"]["goal"]["conditions"] == []
    assert roles["a"]["goal"]["conditions"][0]["kind"] == "momentum"
    from space_env.tasks.parallel import MissionParallelEnv

    env = MissionParallelEnv(
        cases["momentum-nominal"]["scenario"], cases["momentum-nominal"]["task"]
    )
    obs, _ = env.reset(seed=0)
    assert obs["a"]["goal_mask"].sum() == 1 and obs["b"]["goal_mask"].sum() == 0
    assert np.count_nonzero(obs["b"]["goals"]) == 0
    assert obs["b"]["goal_resource_mask"].sum() == 0
    assert np.count_nonzero(obs["b"]["goal_resources"]) == 0
    assert np.count_nonzero(obs["b"]["goal_feature_mask"]) == 0
    assert env.observation_space("a").contains(obs["a"])


def test_gym_uses_same_goal_reward_and_terminal(cases):
    c = cases["momentum-nominal"]
    env = MissionTaskEnv(c["scenario"], c["task"], variable_duration=True)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["goal_feature_names"] == list(GOAL_FEATURES)
    action = {"choices": np.array([1, 0]), "duration": 0}
    obs, reward, done, truncated, info = env.step(action)
    total = reward
    while not done:
        obs, reward, done, truncated, info = env.step(
            {"choices": np.array([0, 0]), "duration": 0}
        )
        total += reward
    assert done and not truncated and info["goal"]["success"] and total == 1
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize(
    "bad", ["unknown-kind", "nan", "quaternion", "actor", "epoch", "callback"]
)
def test_invalid_or_unsupported_goal_rejected_before_steps(cases, bad):
    c = copy.deepcopy(cases["attitude-nominal"])
    goal = c["task"]["goal"]
    condition = goal["conditions"][0]
    if bad == "unknown-kind":
        condition["kind"] = "python"
    elif bad == "nan":
        condition["tolerance_deg"] = float("nan")
    elif bad == "quaternion":
        condition["quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
    elif bad == "actor":
        condition["actor"] = "uncontrolled"
    elif bad == "epoch":
        goal["deadline_sec"] = 99999.0
    else:
        goal["callback"] = "os.system"
    with pytest.raises(ValueError):
        TaskSession(c["scenario"], c["task"])


def test_resealed_goal_success_tamper_fails_cold_replay(cases):
    _, record = run(cases, "orbit-wrong-direction")
    record = copy.deepcopy(record)
    record["metrics"]["success"] = True
    record["metrics"]["goal"]["success"] = True
    record = seal({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_record(record)


def test_absent_goal_keeps_legacy_serialization(cases):
    task = copy.deepcopy(cases["delivery"]["task"])
    task.pop("goal")
    spec = TaskSpec.model_validate(task)
    assert "goal" not in spec.model_dump(mode="json", by_alias=True)
    assert isinstance(
        MissionGoal.model_validate(cases["delivery"]["task"]["goal"]), MissionGoal
    )


def test_queries_never_roll_out_future_or_mutate_native_world(cases, monkeypatch):
    from space_env.runtime import MissionRuntime

    s = session(cases, "orbit-nominal")
    before = s.checkpoint(compact=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Future execution during goal query")

    for method in ("advance", "submit", "forecast", "forecast_pointing"):
        monkeypatch.setattr(MissionRuntime, method, forbidden)
    assert not s.goal_status()["success"]
    assert s.goal_status("b")["conditions"] == []
    assert s.checkpoint(compact=True) == before


def test_extra_running_burn_cannot_hide_behind_completed_impulse(cases):
    from space_env.runtime import AccessPolicy, MissionRuntime
    from space_env.tasks.goals import evaluate_goal

    c = cases["impulse-chemical"]
    runtime = MissionRuntime(c["scenario"], access=AccessPolicy(truth=True))
    command = copy.deepcopy(c["task"]["actions"][0]["commands"][0])
    runtime.submit([command], expected_revision=0, idempotency_key="burn-1")
    runtime.advance(5.0, expected_revision=runtime.revision, idempotency_key="finish-1")
    goal = MissionGoal.model_validate(c["task"]["goal"])
    assert evaluate_goal(goal, runtime)["success"]
    command["command_id"] = "extra-burn"
    runtime.submit(
        [command], expected_revision=runtime.revision, idempotency_key="burn-2"
    )
    runtime.advance(
        1.0, expected_revision=runtime.revision, idempotency_key="partial-2"
    )
    status = evaluate_goal(goal, runtime)
    assert not status["success"] and not status["conditions"][0]["met"]
    assert np.linalg.norm(status["conditions"][0]["errors"]["impulse_ns"]) > 0.1


def test_fault_and_recovery_do_not_erase_safety_violation(cases):
    c = copy.deepcopy(cases["momentum-nominal"])
    c["task"]["roles"]["a"]["action_slots"] = 2
    c["task"]["max_joint_actions"] = 3
    for name, kind in (
        ("fault", "spacecraft_fault"),
        ("repair", "spacecraft_fault_cleared"),
    ):
        c["task"]["actions"].append(
            {
                "id": name,
                "role": "a",
                "kind": "inject",
                "event": {"type": kind, "payload": {"actor": "a", "subsystem": "cpu"}},
            }
        )
    s = TaskSession(c["scenario"], c["task"])
    r = s.step(
        {"@central": ["fault", "repair"]},
        expected_revision=0,
        idempotency_key="incident",
    )
    assert not s._runtime._backend.report()["spacecraft_bus"]["states"]["a"]["faults"]
    assert r["terminated"] and not r["goal"]["success"]
    assert r["goal"]["failures"] == ["health:a"]
    assert (
        TaskSession.from_checkpoint(s.checkpoint(compact=True)).goal_status()
        == s.goal_status()
    )


def test_orbit_release_cannot_exceed_its_window(cases):
    c = copy.deepcopy(cases["orbit-waypoints"])
    c["task"]["goal"]["conditions"][0]["not_before_sec"] = 3.0
    with pytest.raises(ValueError, match="release misses"):
        TaskSession(c["scenario"], c["task"])


def test_initial_achievement_is_preserved_without_free_reward(cases):
    def initial(c):
        condition = copy.deepcopy(c["task"]["goal"]["conditions"][0])
        condition.update(
            id="initial",
            completion="achieve",
            require_command=False,
            quaternion_wxyz=c["scenario"]["definition"]["spacecraft_bus"]["a"][
                "attitude"
            ]["initial_quaternion_wxyz"],
        )
        c["task"]["goal"]["conditions"].insert(0, condition)

    _, record = run(cases, "attitude-nominal", mutate=initial)
    assert record["metrics"]["success"]
    first = record["metrics"]["goal"]["conditions"][0]
    assert first["met"] and not first["current_met"]
    assert record["metrics"]["return"] == 1.0


def test_new_physical_goal_corpus_keeps_negative_rl_and_gates_sft(cases, tmp_path):
    from space_env.llm.corpus import build_corpus, verify_corpus

    inputs = []
    for name in ("momentum-nominal", "momentum-limited"):
        _, record = run(cases, name)
        path = tmp_path / (name + ".json")
        path.write_text(canonical_json(record))
        inputs.append(path)
    corpus = tmp_path / "corpus"
    build_corpus(inputs, corpus)
    summary = verify_corpus(corpus)["summary"]
    assert summary["rows"] == {"rl": 2, "sft": 1, "wire-sft": 0}
    assert summary["successful_train_episodes"] == 1


def test_fractional_waypoints_preserve_physical_clock_and_restore(cases):
    from space_env.tasks.modeling_suite import _orbit_target

    c = copy.deepcopy(cases["orbit-waypoints"])
    c["task"]["actions"][0]["commands"][0]["duration_sec"] = 0.2
    orbit = c["scenario"]["definition"]["orbits"]["a"]
    velocity = orbit["initial"]["v_m_s"]
    direction = (np.array(velocity) / np.linalg.norm(velocity)).tolist()
    for condition, at in zip(c["task"]["goal"]["conditions"], (0.3, 0.6)):
        target = _orbit_target(orbit, direction, firing=0.2, end=at)
        condition.update(at_sec=at, position_m=target[:3], velocity_m_s=target[3:])
    c["task"]["goal"]["deadline_sec"] = 0.6
    s = TaskSession(c["scenario"], c["task"])
    while not s._done:
        response = s.step(
            {"a": ["control"] if s.revision == 0 else ["wait"]},
            expected_revision=s.revision,
            idempotency_key=str(s.revision),
            advance_sec=0.1,
        )
        assert response["elapsed_sec"] > 1e-9
    assert s.goal_status()["success"] and s.goal_status()["time_sec"] == pytest.approx(
        0.6
    )
    assert (
        TaskSession.from_checkpoint(s.checkpoint(compact=True)).goal_status()
        == s.goal_status()
    )


def test_numeric_goal_context_has_type_tolerances_and_budget_masks(cases):
    from space_env.tasks.goals import GOAL_KINDS, GOAL_RESOURCE_FEATURES

    c = cases["momentum-nominal"]
    env = MissionTaskEnv(c["scenario"], c["task"])
    obs, _ = env.reset(seed=0)
    kinds = GOAL_FEATURES.index("kind_code")
    tol = GOAL_FEATURES.index("tolerance_nms")
    unavailable = GOAL_FEATURES.index("tolerance_ns")
    assert obs["goals"][0, kinds] == GOAL_KINDS["momentum"]
    assert obs["goals"][0, tol] == 1e-5 and obs["goal_feature_mask"][0, tol] == 1
    assert obs["goal_feature_mask"][0, unavailable] == 0
    energy = GOAL_RESOURCE_FEATURES.index("energy_budget_j")
    margin = GOAL_RESOURCE_FEATURES.index("energy_margin_j")
    fuel = GOAL_RESOURCE_FEATURES.index("fuel_reserve_kg")
    assert obs["goal_resources"][0, energy] == obs["goal_resources"][0, margin] == 200.0
    assert obs["goal_resource_feature_mask"][0, fuel] == 0
    assert env.observation_space.contains(obs)


def test_actor_without_own_goal_still_observes_only_its_declared_budget(cases):
    from space_env.tasks.goals import GOAL_RESOURCE_FEATURES
    from space_env.tasks.parallel import MissionParallelEnv

    c = copy.deepcopy(cases["momentum-nominal"])
    c["task"]["goal"]["limits"]["b"] = {"max_energy_j": 500.0}
    env = MissionParallelEnv(c["scenario"], c["task"])
    obs, _ = env.reset(seed=0)
    row = np.flatnonzero(obs["b"]["goal_resource_mask"])
    assert len(row) == 1 and obs["b"]["goal_mask"].sum() == 0
    assert (
        obs["b"]["goal_resources"][
            row[0], GOAL_RESOURCE_FEATURES.index("energy_budget_j")
        ]
        == 500.0
    )
    view = env.session.observe("b")["goal"]
    assert [r["actor"] for r in view["resources"]] == ["b"]
    assert env.observation_space("b").contains(obs["b"])
