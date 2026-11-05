"""Live/offline equality, actor authority, bounded incident ancestry and Gym."""

import copy
import json

import numpy as np
import pytest

from examples.planning.incident_mission import configuration
from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.actor_projection import project_actor_observation
from space_env.planning.chain import propose_chain
from space_env.planning.incident import recover_incident
from space_env.planning.incident_journal import (
    append_incident,
    create_journal,
    verify_journal,
)
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import digest


@pytest.fixture(scope="module")
def origin():
    value = configuration()
    return propose_chain(
        MissionRuntime(value["scenario"], access=AccessPolicy(forecast=True)),
        value["scenario"],
        value["task"],
        goals=value["goals"],
    )


def choose(controller, plan=None, wait=False):
    owners = {a["id"]: a["role"] for a in controller._plan["task"]["actions"]}
    ids = (
        ["wait"]
        if wait
        else (plan or controller._plan)["steps"][controller.cursor]["action_ids"]
    )
    return {
        r: [a for a in ids if owners.get(a) == r] or ["wait"] for r in controller.roles
    }


def step(controller, choices=None, key=None):
    obs = controller.observe()
    return controller.step(
        choices or choose(controller),
        expected_bindings={r: o["binding"] for r, o in obs.items()},
        idempotency_key=key or f"policy/{controller.cursor}",
    )


def test_projection_is_owned_and_rejects_wrong_role_or_binding(origin):
    c = ActorCoordinator(origin)
    role = c.roles[0]
    raw = c._session.observe(role)
    v = project_actor_observation(raw, c._spec, role)
    v["catalogue"].clear()
    assert c.observe()[role]["catalogue"]
    assert raw["binding"] != c.observe()[role]["binding"]
    raw["time_sec"] += 1.0
    with pytest.raises(ValueError, match="binding"):
        project_actor_observation(raw, c._spec, role)


@pytest.mark.parametrize(
    "kind", ["foreign", "operator", "missing", "duplicate", "unknown", "stale"]
)
def test_actor_invalid_inputs_do_not_advance(origin, kind):
    c = ActorCoordinator(origin)
    before = c.cursor
    choices = choose(c, wait=True)
    obs = c.observe()
    bindings = {r: o["binding"] for r, o in obs.items()}
    if kind == "foreign":
        choices[c.roles[0]] = [c.catalogues[c.roles[1]][1]]
    elif kind == "operator":
        choices["@incident"] = ["wait"]
        bindings["@incident"] = "0" * 64
    elif kind == "missing":
        choices.pop(c.roles[0])
    elif kind == "duplicate":
        choices[c.roles[0]] = ["wait", "wait"]
    elif kind == "unknown":
        choices[c.roles[0]] = ["foreign-command"]
    else:
        bindings[c.roles[0]] = "0" * 64
    with pytest.raises(ValueError):
        c.step(choices, expected_bindings=bindings, idempotency_key="invalid")
    assert c.cursor == before


def test_actor_idempotency_and_catalogue_ownership(origin):
    c = ActorCoordinator(origin)
    choices = choose(c)
    bindings = {r: v["binding"] for r, v in c.observe().items()}
    one = c.step(choices, expected_bindings=bindings, idempotency_key="once")
    cursor = c.cursor
    one["observations"].clear()
    two = c.step(choices, expected_bindings=bindings, idempotency_key="once")
    assert two["observations"] and c.cursor == cursor
    with pytest.raises(ValueError, match="conflict"):
        c.step(choose(c, wait=True), expected_bindings=bindings, idempotency_key="once")
    catalogues = c.catalogues
    catalogues.clear()
    assert c.catalogues


@pytest.fixture(scope="module")
def incident(origin):
    x = PlanExecution(origin["plan"])
    x.run(decisions=50)
    return recover_incident(
        origin,
        x.checkpoint(),
        {
            "disruptions": [
                {"station": "g02", "end_sec": 120.0},
                {"station": "g06", "end_sec": 120.0},
            ]
        },
    )


def test_live_projection_exactly_matches_offline_and_restore(incident):
    from space_env.planning.incident_data import role_rows
    from space_env.tasks import VerifiedEpisode

    c = ActorCoordinator(incident)
    before = c.observe()
    response = step(c)
    sealed = c.as_execution()
    verified = VerifiedEpisode(sealed["execution"]["episode"])
    for role in c.roles:
        row = next(role_rows(verified, incident, role))
        assert (
            row["observation"] == before[role] == response["before_observations"][role]
        )
        assert row["next_observation"] == response["observations"][role]
    restored = ActorCoordinator.from_checkpoint(c.checkpoint())
    assert restored.observe() == c.observe()
    tampered = c.checkpoint()
    tampered["trace"][0]["response_sha256"] = "0" * 64
    tampered["sha256"] = digest({k: v for k, v in tampered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay"):
        ActorCoordinator.from_checkpoint(tampered)


def test_joint_gym_spaces_seed_and_policy_step(origin):
    from space_env.planning.actor_gym import ActorMissionEnv

    env = ActorMissionEnv(origin)
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert info["physical_episode_seed"] == origin["plan"]["episode_seed"]
    numeric = {
        r: env.action_ids[r].index(choose(env.coordinator)[r][0]) for r in env.roles
    }
    obs, reward, terminated, truncated, info = env.step(numeric)
    assert env.observation_space.contains(obs) and np.isfinite(reward)
    assert not terminated and not truncated and info["reward_scope"] == "joint_team"
    with pytest.raises(ValueError):
        env.step({r: -1 for r in env.roles})


@pytest.fixture(scope="module")
def journal(origin):
    j = create_journal(origin)
    x = PlanExecution(origin["plan"])
    x.run(decisions=50)
    j = append_incident(
        j,
        x.checkpoint(),
        {
            "disruptions": [
                {"station": "g02", "end_sec": 120.0},
                {"station": "g06", "end_sec": 120.0},
            ]
        },
    )
    # Actor chooses to wait during an already planned waiting cell, then returns
    # a fixed executed-prefix artifact. This is the same request protocol.
    c = ActorCoordinator(j)
    while c.cursor < 62:
        step(c, choose(c, wait=True) if c.cursor == 61 else None)
    record = c.as_execution()["execution"]
    return append_incident(
        j,
        record,
        {
            "disruptions": [
                {"station": "g02", "end_sec": 140.0},
                {"station": "g06", "end_sec": 140.0},
            ]
        },
    )


def test_flat_journal_native_replay_and_monotonic_prefix(journal):
    assert verify_journal(journal, native=True)["epochs"] == 2
    assert journal["epochs"][1]["parent_sha256"] == journal["epochs"][0]["sha256"]
    assert (
        journal["epochs"][1]["result"]["committed_prefix_steps"]
        > journal["epochs"][0]["result"]["committed_prefix_steps"]
    )
    assert all(
        "source_record" not in n["result"] and "source_proposal" not in n["result"]
        for n in journal["epochs"]
    )
    c = ActorCoordinator(journal)
    while not c.done:
        step(c)
    assert (
        c.as_execution()["execution"]["episode"]["runtime"]["backend_report"][
            "lifecycle"
        ]["metrics"]["fulfilled_requests"]
        == 2
    )
    assert len(list(c.verified_rows(sft=True))) > 0


@pytest.mark.parametrize("kind", ["parent", "cursor", "edit", "report"])
def test_resealed_journal_corruption_denied(journal, kind):
    j = copy.deepcopy(journal)
    n = j["epochs"][1]
    if kind == "parent":
        n["parent_sha256"] = "0" * 64
    elif kind == "cursor":
        n["cursor"] = 0
    elif kind == "edit":
        n["actor_edits"][0]["action_ids"] = [
            next(
                a["id"]
                for a in j["epochs"][0]["result"]["plan"]["task"]["actions"]
                if a["kind"] == "inject"
            )
        ]
    else:
        n["source_episode_sha256"] = "0" * 64
    n["sha256"] = digest({k: v for k, v in n.items() if k != "sha256"})
    j["sha256"] = digest({k: v for k, v in j.items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_journal(j, native=kind == "report")


def test_actor_failed_goal_does_not_export_sft(origin):
    c = ActorCoordinator(origin)
    step(c, choose(c, wait=True))
    with pytest.raises(ValueError, match="SFT"):
        list(c.verified_rows(sft=True))


def test_actor_data_native_and_inventory(origin, tmp_path):
    from space_env.planning.actor_data import publish_actor_data, verify_actor_data

    c = ActorCoordinator(origin)
    step(c, choose(c, wait=True))
    root = tmp_path / "data"
    manifest = publish_actor_data(c.checkpoint(), root)
    assert not manifest["goal"]["passed"] and all(
        x["format"] == "rl" for x in manifest["training_files"].values()
    )
    assert verify_actor_data(root, native=True)["native_rows_recomputed"]
    (root / "extra").write_text("x")
    with pytest.raises(ValueError, match="inventory"):
        verify_actor_data(root)


@pytest.mark.parametrize("kind", ["duplicate_trace", "boolean_cursor"])
def test_checkpoint_noncanonical_trace_is_refused(origin, kind):
    c = ActorCoordinator(origin)
    step(c, choose(c, wait=True))
    record = c.checkpoint()
    if kind == "duplicate_trace":
        record["trace"].append(copy.deepcopy(record["trace"][0]))
    else:
        record["cursor"] = True
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        ActorCoordinator.from_checkpoint(record)


def test_actor_file_cli_roundtrip_no_overwrite(origin, tmp_path, capsys):
    from space_env.cli import main
    from space_env.simulation.snapshot import canonical_json

    reference = tmp_path / "reference.json"
    reference.write_text(canonical_json(origin))
    checkpoint = tmp_path / "controller.json"
    assert main(["actor-controller-start", str(reference), str(checkpoint)]) is None
    capsys.readouterr()
    assert main(["actor-controller-verify", str(checkpoint)]) is None
    assert json.loads(capsys.readouterr().out)["native_trace_replayed"]
    with pytest.raises(SystemExit) as error:
        main(["actor-controller-start", str(reference), str(checkpoint)])
    assert error.value.code == 2


def test_gym_multislot_mask_and_duplicate_padding(origin):
    from space_env.planning.actor_gym import ActorMissionEnv

    task = copy.deepcopy(origin["source_task"])
    for role in task["roles"].values():
        role["action_slots"] = 2
    task["max_joint_actions"] = 4
    proposal = propose_chain(
        MissionRuntime(origin["plan"]["scenario"], access=AccessPolicy(forecast=True)),
        origin["plan"]["scenario"],
        task,
    )
    env = ActorMissionEnv(proposal)
    obs, _ = env.reset(seed=4)
    for role in env.roles:
        assert len(obs[role]["action_mask"]) == len(env.action_ids[role])
    obs, _, _, _, _ = env.step({r: np.array([0, 0]) for r in env.roles})
    assert env.observation_space.contains(obs)


def test_pettingzoo_parallel_interface(origin):
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from space_env.planning.actor_parallel import ActorParallelEnv

    env = ActorParallelEnv(origin)
    parallel_api_test(env, num_cycles=3)
    assert "@incident" not in env.possible_agents
