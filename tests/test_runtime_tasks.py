"""One executor, causal local masks, physical-time transitions and verified data."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    reference_scenario,
)
from space_env.simulation.snapshot import digest
from space_env.tasks import (
    MissionTaskEnv,
    TaskSession,
    TaskTools,
    export_dataset,
    objective_registry,
)
from tests.test_custody_network import spec as custody_fixture


@pytest.fixture(scope="module")
def scenario():
    return reference_scenario(
        custody_fixture.__wrapped__(), id="task-test", lineage="task-family"
    )


def task():
    def submit(id, role, kind, **kwargs):
        return {
            "id": id,
            "role": role,
            "kind": "submit",
            "commands": [
                {
                    "command_id": id,
                    "actor_id": role,
                    "kind": kind,
                    "duration_sec": 4.0,
                    **kwargs,
                }
            ],
        }

    return {
        "id": "delivery",
        "roles": {
            "a": {"actors": ["a"], "action_slots": 2},
            "b": {"actors": ["b"]},
            "operator": {"stations": ["Ground"]},
        },
        "decision_sec": 0.5,
        "max_decision_sec": 5.0,
        "max_decisions": 100,
        "actions": [
            submit(
                "relay",
                "a",
                "relay",
                destination_id="b",
                product_id="a",
                relay={"ttl_sec": 20.0},
            ),
            submit(
                "ground-a", "a", "downlink", destination_id="Ground", product_id="a"
            ),
            submit(
                "ground-b", "b", "downlink", destination_id="Ground", product_id="a"
            ),
            {"id": "cancel-a", "role": "a", "kind": "cancel_actor", "actor": "a"},
            {
                "id": "recover-ground",
                "role": "operator",
                "kind": "inject",
                "event": {
                    "type": "station_recovered",
                    "payload": {"station": "Ground"},
                },
            },
        ],
    }


def step(session, ids=(), dt=None, role="@central", key=None):
    return session.step(
        {role: list(ids)},
        expected_revision=session.revision,
        idempotency_key=key or f"d/{session.revision}",
        advance_sec=dt,
    )


def test_runtime_fork_preserves_aliases_and_independent_state(scenario):
    runtime = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    clone = runtime.fork()
    clone.advance(1.0, expected_revision=0, idempotency_key="a")
    assert runtime.time_sec == 0 and clone.time_sec == 1.0
    assert clone._backend.mission is not runtime._backend.mission
    assert clone._backend.mission.dtn is not runtime._backend.mission.dtn
    with pytest.raises(RuntimeContractError, match="truth"):
        MissionRuntime(scenario).fork()


def test_native_local_inventory_and_station_telemetry_no_future_events(scenario):
    s = copy.deepcopy(scenario)
    s["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 5.0, "end_sec": 10.0}
    ]
    m = MissionRuntime(s)
    assert m.inventory("a")["products"][0]["remaining_bits"] == 2e6
    assert not m.station_observation("Ground")["outage"]
    assert "outages" not in m.station_observation("Ground")
    with pytest.raises(RuntimeContractError):
        m.inventory("missing")


def test_task_wait_and_physical_time_discount(scenario):
    s = TaskSession(scenario, task())
    r = step(s, dt=0.25)
    assert r["elapsed_sec"] == 0.25 and r["discount"] == pytest.approx(0.99**0.25)
    assert r["reward"] == 0 and s.observe()["time_sec"] == 0.25
    assert r["feedback"]["accepted"]


def test_mask_checks_local_ownership_without_advancing_or_querying_future(scenario):
    s = TaskSession(scenario, task())
    before = s.checkpoint()
    view = s.observe("a")
    assert view["action_mask"] == [1, 1, 1, 0, 0, 0]
    assert s.checkpoint() == before
    assert all(r["kind"] != "actor" or r["id"] == "a" for r in view["entities"])
    assert not any(r["key"] == "product:b" for r in view["entities"])


def test_hidden_future_and_seed_changes_do_not_change_current_observation_or_binding(
    scenario,
):
    first = copy.deepcopy(scenario)
    second = copy.deepcopy(scenario)
    first["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 20.0, "end_sec": 25.0}
    ]
    second["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 21.0, "end_sec": 26.0}
    ]
    second["seed"] = 999
    second["definition"]["dtn_network"]["seed"] = 1234
    a, b = TaskSession(first, task()), TaskSession(second, task())
    assert a.observe() == b.observe()
    assert a.observe("a") == b.observe("a")


def test_remote_inventory_does_not_shift_local_slots(scenario):
    a = TaskSession(scenario, task())
    data = copy.deepcopy(scenario)
    data["definition"]["products"][1]["size_bits"] = 3e6
    data["definition"]["products"][1]["id"] = "hidden-remote-product"
    b = TaskSession(data, task())
    assert a.observe("a") == b.observe("a")
    assert a.observe() != b.observe()


def test_masked_attempt_costs_time_but_cannot_command_other_role(scenario):
    s = TaskSession(scenario, task())
    r = step(s, ["ground-b"], role="a")
    assert not r["feedback"]["accepted"] and r["feedback"]["code"] == "role_scope"
    assert r["elapsed_sec"] == 0.5 and r["reward"] < 0
    assert not s._runtime.report()["commands"]


def test_custody_is_not_ground_reward_then_ack_gives_reward_once(scenario):
    s = TaskSession(scenario, task())
    r = step(s, ["relay"])
    assert r["reward"] == 0
    for _ in range(5):
        r = step(s)
    assert r["reward"] == 0 and s._runtime.inventory("b")["products"][0]["id"] == "a"
    assert s.observe("b")["action_mask"][3] == 1
    r = step(s, ["ground-b"])
    total = r["reward"]
    for _ in range(6):
        total += step(s)["reward"]
    assert total == 2.0  # Reference default primary unit is on-time completed Mbits.
    assert step(s)["reward"] == 0


def test_stale_revision_and_idempotency_survive_later_steps(scenario):
    s = TaskSession(scenario, task())
    r = step(s, key="first")
    step(s)
    assert s.step({"@central": []}, expected_revision=0, idempotency_key="first") == r
    before = s.checkpoint()
    with pytest.raises(RuntimeContractError, match="revision"):
        s.step({"@central": []}, expected_revision=0, idempotency_key="stale")
    with pytest.raises(RuntimeContractError, match="reused"):
        s.step({"@central": ["relay"]}, expected_revision=0, idempotency_key="first")
    assert s.checkpoint() == before


def test_unknown_action_and_schema_errors_have_no_effect(scenario):
    s = TaskSession(scenario, task())
    before = s.checkpoint()
    with pytest.raises(ValueError):
        step(s, ["secret-does-not-exist"])
    with pytest.raises(ValueError):
        step(s, dt=10.0)
    assert s.checkpoint() == before


def test_duplicate_choice_rejected_with_one_time_charge(scenario):
    s = TaskSession(scenario, task())
    r = step(s, ["relay", "relay"])
    assert r["feedback"]["code"] == "duplicate_choice" and r["elapsed_sec"] == 0.5
    assert not s._runtime.report()["commands"]


def test_cancel_and_resubmit_one_atomic_epoch(scenario):
    s = TaskSession(scenario, task())
    step(s, ["relay"])
    r = step(s, ["cancel-a", "ground-a"])
    assert r["feedback"]["accepted"]
    assert s._runtime.report()["commands"][0]["status"] == "cancelled"
    assert s._runtime.report()["commands"][1]["command"]["kind"] == "downlink"


def test_joint_cancellation_rolled_back_when_another_choice_is_rejected(scenario):
    s = TaskSession(scenario, task())
    step(s, ["relay"])
    r = step(s, ["cancel-a", "ground-b"])
    assert not r["feedback"]["accepted"]
    assert s._runtime.report()["commands"][0]["status"] == "running"


def test_station_operator_uses_same_recovery_dispatcher(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 0.0, "end_sec": 5.0}
    ]
    s = TaskSession(data, task())
    v = s.observe("operator")
    assert any(
        r["kind"] == "station" and r["values"]["outage"] == 1 for r in v["entities"]
    )
    r = step(s, ["recover-ground"], role="operator")
    assert (
        r["feedback"]["accepted"]
        and not s._runtime.station_observation("Ground")["outage"]
    )


def test_role_authority_cannot_be_embedded_in_template(scenario):
    data = task()
    data["actions"][0]["commands"][0]["actor_id"] = "b"
    with pytest.raises(ValueError, match="another role"):
        TaskSession(scenario, data)
    data = task()
    data["actions"][-1]["event"]["type"] = "station_outage_added"
    with pytest.raises(ValueError, match="unsupported"):
        TaskSession(scenario, data)


@pytest.mark.parametrize(
    "field,value", [("max_entities", 1), ("max_actions", 1), ("max_relations", 1)]
)
def test_no_silent_capacity_clipping(scenario, field, value):
    data = task()
    data[field] = value
    with pytest.raises(ValueError, match="capacity"):
        TaskSession(scenario, data)


def test_entity_capacity_failure_rolls_back_entire_decision(scenario):
    probe = TaskSession(scenario, task())
    data = task()
    data["max_entities"] = len(probe.observe()["entities"])
    s = TaskSession(scenario, data)
    before = s.checkpoint()
    with pytest.raises(ValueError, match="capacity"):
        step(s, ["relay"])
    assert s.checkpoint() == before


def test_runtime_physics_failure_does_not_become_fake_transition(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["dtn_network"]["max_events"] = 1
    s = TaskSession(data, task())
    before = s.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        step(s, ["relay"])
    assert s.checkpoint() == before


def test_horizon_termination_and_decision_budget_truncation_differ(scenario):
    data = task()
    data["max_decisions"] = 1
    s = TaskSession(scenario, data)
    r = step(s)
    assert r["truncated"] and not r["terminated"] and r["discount"] > 0
    with pytest.raises(RuntimeContractError, match="reset"):
        step(s)
    data = task()
    data["max_decision_sec"] = 30.0
    s = TaskSession(scenario, data)
    r = step(s, dt=30.0)
    assert r["terminated"] and not r["truncated"] and r["discount"] == 0


def test_reset_seed_preserve_and_derive_network_are_explicit(scenario):
    data = task()
    data["seed_policy"] = "derive-network"
    a, b = TaskSession(scenario, data, seed=123), TaskSession(scenario, data, seed=123)
    assert a._runtime.checkpoint() == b._runtime.checkpoint()
    c = TaskSession(scenario, data, seed=124)
    assert (
        a._runtime._spec.definition["dtn_network"]["seed"]
        != c._runtime._spec.definition["dtn_network"]["seed"]
    )
    assert "seed" not in json.dumps(a.observe())
    assert (
        TaskSession(scenario, task(), seed=1)._runtime.checkpoint()
        == TaskSession(scenario, task(), seed=2)._runtime.checkpoint()
    )


def test_tool_bound_role_filters_observation_and_central_privilege(scenario):
    s = TaskSession(scenario, task())
    tools = TaskTools(s, "a")
    result = tools.step(["ground-b"], expected_revision=0, idempotency_key="attempt")
    assert not result["feedback"]["accepted"] and result["observation"]["role"] == "a"
    assert "role_observations" not in result
    with pytest.raises(PermissionError):
        tools.joint_step(
            {"b": ["ground-b"]}, expected_revision=1, idempotency_key="foreign"
        )
    assert TaskTools(s).joint_schema()["name"] == "space_env_task_joint_step"


def test_task_replay_and_verified_exports_do_not_include_private_scenario_in_messages(
    scenario, tmp_path
):
    s = TaskSession(scenario, task())
    step(s, ["relay"])
    step(s)
    path = tmp_path / "episode.json"
    s.save(path)
    restored = TaskSession.load(path)
    assert restored.checkpoint() == s.checkpoint()
    for mode in ["rl", "sft"]:
        output = tmp_path / (mode + ".jsonl")
        result = export_dataset(s.checkpoint(), output, format=mode)
        rows = [json.loads(r) for r in output.read_text().splitlines()]
        assert result["rows"] == 2 and all(r["replay_verified"] for r in rows)
        assert all("scenario" not in r and "runtime" not in r for r in rows)
        if mode == "sft":
            text = json.dumps([r["messages"] for r in rows])
            assert "episode_seed" not in text and "loss_probability" not in text
    bad = s.checkpoint()
    bad["transitions"][0]["transition"]["reward"] = 999.0
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay mismatch"):
        TaskSession.from_checkpoint(bad)


def test_sft_rejections_are_not_teacher_success(scenario, tmp_path):
    s = TaskSession(scenario, task())
    step(s, ["ground-b"])
    assert (
        export_dataset(s.checkpoint(), tmp_path / "sft.jsonl", format="sft")["rows"]
        == 0
    )
    assert export_dataset(s.checkpoint(), tmp_path / "rl.jsonl")["rows"] == 1


def test_lineage_split_shared_across_seed_variants(scenario, tmp_path):
    data = task()
    data["seed_policy"] = "derive-network"
    splits = []
    for seed in [1, 2, 3]:
        s = TaskSession(scenario, data, seed=seed)
        step(s)
        splits.append(
            export_dataset(s.checkpoint(), tmp_path / f"{seed}.jsonl")["split"]
        )
    assert len(set(splits)) == 1


def test_registry_copy_cannot_mutate_reward(scenario):
    registry = objective_registry()
    registry["delivery"]["completion"] = 999
    s = TaskSession(scenario, task())
    assert objective_registry()["delivery"]["completion"] == 1
    data = task()
    data["objective"] = "efficient"
    s = TaskSession(scenario, data)
    assert step(s)["reward"] < 0


def test_gym_facade_checker_and_unknown_feature_masks(scenario):
    data = task()
    data["max_decisions"] = 4
    env = MissionTaskEnv(scenario, data)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert np.all(obs["entities"][obs["feature_mask"] == 0] == 0)
    assert not obs["entity_mask"].all()


def test_gym_and_structured_tools_execute_identical_runtime(scenario):
    env = MissionTaskEnv(scenario, task())
    env.reset(seed=0)
    direct = TaskSession(scenario, task(), seed=0)
    tools = TaskTools(direct)
    actions = np.zeros(task().get("max_joint_actions", 8), dtype=np.int64)
    actions[0] = 1
    result = env.step(actions)
    other = tools.step(["relay"], expected_revision=0, idempotency_key="human")
    assert (
        result[1] == other["reward"]
        and env.session._runtime.checkpoint() == direct._runtime.checkpoint()
    )


def test_pettingzoo_parallel_api_and_live_agents(scenario):
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from space_env.tasks.parallel import MissionParallelEnv

    data = task()
    data["max_decisions"] = 4
    env = MissionParallelEnv(scenario, data)
    parallel_api_test(env, num_cycles=8)
    obs, infos = env.reset(seed=7)
    assert set(obs) == {"a", "b", "operator"}
    for a in env.agents:
        assert env.observation_space(a).contains(obs[a])
    _, _, _, _, infos = env.step(
        {"a": np.array([3, 0]), "b": 0, "operator": 0}
    )  # a tries b's action.
    assert infos["a"]["feedback"]["code"] == "role_scope"
    assert not env.session._runtime.report()["commands"]


def test_parallel_and_centralized_commands_have_identical_runtime(scenario):
    pytest.importorskip("pettingzoo")
    from space_env.tasks.parallel import MissionParallelEnv

    env = MissionParallelEnv(scenario, task())
    env.reset(seed=0)
    session = TaskSession(scenario, task(), seed=0)
    env.step({"a": np.array([1, 0]), "b": 0, "operator": 0})
    step(session, ["relay"])
    assert env.session._runtime.checkpoint() == session._runtime.checkpoint()
    for _ in range(5):
        env.step({a: np.array([0, 0]) if a == "a" else 0 for a in env.agents})
        step(session)
    assert env.session._runtime.checkpoint() == session._runtime.checkpoint()
    env.step({"a": np.array([0, 0]), "b": 3, "operator": 0})
    step(session, ["ground-b"])
    assert env.session._runtime.checkpoint() == session._runtime.checkpoint()


def test_only_local_feedback_enters_local_rl_export(scenario, tmp_path):
    s = TaskSession(scenario, task())
    s.step(
        {"b": ["relay"], "a": ["wait"]}, expected_revision=0, idempotency_key="foreign"
    )
    out = tmp_path / "a.jsonl"
    export_dataset(s.checkpoint(), out, role="a")
    row = json.loads(out.read_text())
    assert row["feedback"] == {"accepted": False, "code": "joint_rejected"}
    assert set(row["action"]["choices"]) == {"a"}


def test_local_export_projects_central_joint_actions(scenario, tmp_path):
    s = TaskSession(scenario, task())
    step(s, ["relay"])
    out = tmp_path / "local.jsonl"
    export_dataset(s.checkpoint(), out, role="a")
    assert json.loads(out.read_text())["action"]["choices"] == {"a": ["relay"]}
    export_dataset(s.checkpoint(), out, role="b")
    assert json.loads(out.read_text())["action"]["choices"] == {"b": ["wait"]}


def test_reward_waits_for_ground_ack_in_managed_thermal_mission(scenario):
    from tests.test_custody_network import coupled_scenario

    data = coupled_scenario(custody_fixture.__wrapped__(), thermal=True)
    t = task()
    t["actions"] = [
        {
            "id": "image",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "i",
                    "actor_id": "a",
                    "kind": "imaging",
                    "duration_sec": 2.0,
                    "request_id": "request",
                }
            ],
        },
        {
            "id": "cpu",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "c",
                    "actor_id": "a",
                    "kind": "transform",
                    "duration_sec": 2.0,
                    "job_id": "compress",
                }
            ],
        },
        {
            "id": "relay",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "r",
                    "actor_id": "a",
                    "kind": "relay",
                    "duration_sec": 2.0,
                    "product_id": "final",
                    "destination_id": "b",
                    "relay": {"ttl_sec": 10.0},
                }
            ],
        },
        {
            "id": "ground-b",
            "role": "b",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "g",
                    "actor_id": "b",
                    "kind": "downlink",
                    "duration_sec": 2.0,
                    "product_id": "final",
                    "destination_id": "Ground",
                }
            ],
        },
    ]
    s = TaskSession(data, t)
    assert step(s, ["image"], dt=2.0)["reward"] == 0
    assert step(s, ["cpu"], dt=2.0)["reward"] == 0
    assert step(s, ["relay"], dt=2.0)["reward"] == 0
    assert step(s, ["ground-b"], dt=1.0)["reward"] == 1.0
    assert step(s, dt=1.0)["reward"] == 0
    assert s._runtime._backend.lifecycle.transport is s._runtime._backend.mission.dtn
    assert s._runtime._backend.bus is s._runtime._backend.mission.bus


def test_efficient_reward_matches_actual_native_energy_not_soc_loss(scenario):
    t = task()
    t["objective"] = "efficient"
    s = TaskSession(scenario, t)
    r = step(s, ["relay"], dt=0.5)
    # Source TX 10 W, receiver listen 2 W, both for 0.5s.
    assert r["reward_components"]["energy_j"] == pytest.approx(6.0)
    assert r["reward"] == pytest.approx(-0.001 * 6.0 - 0.001 * 0.5)


def test_source_task_and_registry_objects_cannot_mutate_session(scenario):
    t = task()
    s = TaskSession(scenario, t)
    before = s.observe()
    t["actions"][0]["commands"][0]["destination_id"] = "a"
    info = s.describe()
    info["roles"]["a"]["actors"].append("b")
    assert s.observe() == before


def test_control_checkpoint_edit_cannot_be_relabelled_as_verified_data(
    scenario, tmp_path
):
    s = TaskSession(scenario, task())
    step(s)
    data = s.checkpoint()
    data["runtime"]["report"]["time_sec"] = 999
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    with pytest.raises(ValueError, match="runtime replay mismatch"):
        export_dataset(data, tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()


def test_queued_control_rejection_does_not_leak_hidden_validation_detail(scenario):
    data = task()
    data["actions"][0]["commands"][0]["destination_id"] = "missing-node"
    s = TaskSession(scenario, data)
    r = step(s, ["relay"])
    assert r["feedback"]["code"] == "runtime_rejected"
    assert "missing-node" not in json.dumps(r["feedback"])
    assert s._runtime.time_sec == 0.5


def test_variable_duration_gym_masks_and_pettingzoo_use_one_physical_clock(scenario):
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from space_env.tasks.parallel import MissionParallelEnv

    t = task()
    t["durations_sec"] = [0.25, 0.5, 1.0]
    t["max_decisions"] = 4
    gym = MissionTaskEnv(scenario, t, variable_duration=True)
    gym.reset(seed=2)
    sampled = gym.action_space.sample(mask=gym.action_masks())
    assert gym.action_space.contains(sampled)
    choices = np.zeros(t.get("max_joint_actions", 8), dtype=np.int64)
    assert gym.step({"choices": choices, "duration": 0})[4]["elapsed_sec"] == 0.25
    parallel = MissionParallelEnv(scenario, t, variable_duration=True)
    parallel_api_test(parallel, num_cycles=8)
    parallel.reset(seed=2)
    actions = {
        a: {"choices": np.array([0, 0]) if a == "a" else 0, "duration": 2}
        for a in parallel.agents
    }
    actions["a"]["duration"] = 0
    _, _, _, _, infos = parallel.step(actions)
    assert all(v["elapsed_sec"] == 0.25 for v in infos.values())


def test_future_reference_parameters_hidden_until_target_visible(scenario):
    from tests.test_custody_network import coupled_scenario

    data = coupled_scenario(custody_fixture.__wrapped__())
    data["definition"]["lifecycle"]["requests"][0]["release_sec"] = 5.0
    t = task()
    t["actions"] = [
        {
            "id": "slot",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "i",
                    "actor_id": "a",
                    "kind": "imaging",
                    "request_id": "request",
                    "duration_sec": 2.0,
                }
            ],
        }
    ]
    s = TaskSession(data, t)
    assert "commands" not in s.observe()["catalogue"][1]
    step(s, dt=5.0)
    assert s.observe()["catalogue"][1]["commands"][0]["request_id"] == "request"
    actor = next(r for r in s.observe("a")["entities"] if r["kind"] == "actor")
    assert all(k in actor["values"] for k in ["r_x_m", "v_z_m_s", "q_w", "mass_kg"])


def test_cli_run_resume_replay_export_parity(scenario, tmp_path):
    from types import SimpleNamespace

    from space_env.tasks.cli import run_command

    base = TaskSession(scenario, task())
    step(base, ["relay"], key="start")
    for name, value in [
        ("scenario", scenario),
        ("task", task()),
        (
            "plan",
            {
                "decisions": [
                    {
                        "choices": {"@central": ["relay"]},
                        "expected_revision": 0,
                        "idempotency_key": "start",
                        "advance_sec": None,
                    }
                ]
            },
        ),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    out = tmp_path / "record.json"
    assert run_command(
        SimpleNamespace(
            command="task-run",
            scenario=tmp_path / "scenario.json",
            task=tmp_path / "task.json",
            plan=tmp_path / "plan.json",
            output=out,
        )
    )["ok"]
    assert TaskSession.load(out).checkpoint() == base.checkpoint()
    assert run_command(SimpleNamespace(command="task-replay", record=out))["ok"]
    assert (
        run_command(
            SimpleNamespace(
                command="task-export",
                record=out,
                output=tmp_path / "rl.jsonl",
                format="rl",
                role="@central",
            )
        )["rows"]
        == 1
    )


def test_runtime_fork_controlled_impulse_does_not_change_parent_orbit():
    from space_env.runtime import dynamic_reference_scenario
    from tests.test_reference_dynamic import dynamic_spec

    scenario = dynamic_reference_scenario(
        dynamic_spec.__wrapped__(), id="fork-dynamic", lineage="fork"
    )
    original = MissionRuntime(scenario, access=AccessPolicy(truth=True))
    before = original.checkpoint()
    branch = original.fork()
    branch.submit(
        [
            {
                "command_id": "kick",
                "actor_id": "a",
                "kind": "impulse",
                "duration_sec": 0.0,
                "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 1.0, 0.0]},
            }
        ],
        expected_revision=0,
        idempotency_key="kick",
    )
    branch.advance(1.0, expected_revision=1, idempotency_key="time")
    assert original.checkpoint() == before
    assert (
        branch.observe("a")["actors"]["a"]["orbit"]["state_si"]
        != original.observe("a")["actors"]["a"]["orbit"]["state_si"]
    )


def test_task_template_cannot_probe_privileged_forecast_binding(scenario):
    data = task()
    data["actions"][0]["commands"][0]["forecast"] = {
        "scenario_sha256": "0" * 64,
        "trajectory_sha256": "0" * 64,
        "revision": 0,
        "geometry_revision": 0,
        "actor_id": "a",
    }
    with pytest.raises(ValueError, match="privileged forecast"):
        TaskSession(scenario, data)
