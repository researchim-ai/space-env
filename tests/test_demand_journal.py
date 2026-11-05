"""Incoming demand is native state, cumulative service and explicit policy epochs."""

import copy
import json

import pytest

from examples.planning.demand_mission import configuration, policy_step, run
from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.demand_contract import (
    DemandGoals,
    demand_evidence,
    effective_scenario,
)
from space_env.planning.demand_journal import verify_demand_journal
from space_env.planning.mission_plan import validate_plan
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSpec


def read(path):
    return json.loads(path.read_text())


def reseal(value):
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    return value


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("demand") / "mission"
    result = run(configuration("two_updates"), root)
    return root, result, read(root / "journal.json")


def test_native_two_updates_and_cumulative_service(completed):
    root, result, journal = completed
    assert result["epochs"] == 2
    assert result["fulfilled"] == result["total"] == 4
    assert result["mission_goal_passed"]
    assert len(set(result["catalogue_versions"])) == 3
    assert result["actor_rows"]["rl"] == result["actor_rows"]["sft"] > 0
    assert read(root / "journal-audit.json")["native_epochs_recomputed"]
    assert (
        len(
            journal["origin"]["plan"]["scenario"]["definition"]["lifecycle"]["requests"]
        )
        == 2
    )
    assert verify_demand_journal(journal)["total_added_requests"] == 2


def test_catalogue_templates_and_history_immutable(completed):
    _, _, journal = completed
    previous = journal["origin"]["plan"]
    for node in journal["epochs"]:
        current = node["result"]["plan"]
        old = {a["id"]: a for a in previous["task"]["actions"]}
        new = {a["id"]: a for a in current["task"]["actions"]}
        assert all(new[k] == v for k, v in old.items())
        assert previous["scenario"] == current["scenario"]
        previous = current


def test_rebind_retires_old_owner_and_rejects_stale_bindings(completed):
    root, _, _ = completed
    old = ActorCoordinator.from_checkpoint(read(root / "before-0.json"))
    bindings = {r: o["binding"] for r, o in old.observe().items()}
    new = old.rebind(read(root / "journal-0.json"))
    with pytest.raises(ValueError, match="superseded"):
        old.observe()
    cursor = new.cursor
    with pytest.raises(ValueError, match="binding"):
        new.step(
            {r: ["wait"] for r in new.roles},
            expected_bindings=bindings,
            idempotency_key="stale",
        )
    assert new.cursor == cursor
    for obs in new.observe().values():
        assert obs["catalogue_version"] == new.catalogue_version
        assert obs["schema"] == "space-env.versioned-actor-observation/1"
        assert not any(
            a["kind"] == "inject" for a in obs["catalogue"] if a["id"] != "wait"
        )


def test_bad_rebind_does_not_retire_controller(completed):
    root, _, journal = completed
    old = ActorCoordinator.from_checkpoint(read(root / "before-0.json"))
    with pytest.raises(ValueError, match="source"):
        old.rebind(journal)
    assert old.observe()


@pytest.mark.parametrize("field", ["intent", "source_task", "event", "version"])
def test_resealed_journal_tampering_refused(completed, field):
    _, _, original = completed
    journal = copy.deepcopy(original)
    node = journal["epochs"][0]
    core = node["result"]
    if field == "intent":
        core["intent_sha256"] = "0" * 64
    elif field == "source_task":
        core["source_task"]["id"] += "-tampered"
    elif field == "event":
        node["batch"]["additions"][0]["request"]["deadline_sec"] += 1
    else:
        node["catalogue_version"] = "0" * 64
    reseal(core)
    reseal(node)
    reseal(journal)
    with pytest.raises(ValueError):
        verify_demand_journal(journal)


def test_manifest_cannot_drop_incoming_request(completed):
    _, _, journal = completed
    plan = copy.deepcopy(journal["epochs"][-1]["result"]["plan"])
    plan["goals"]["demand"]["events"].pop()
    reseal(plan)
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("permission", ["demand_actors", "demand_stations"])
def test_explicit_demand_authority_required(completed, permission):
    _, _, journal = completed
    task = copy.deepcopy(journal["epochs"][0]["result"]["plan"]["task"])
    task["roles"]["@demand"][permission] = []
    with pytest.raises(ValueError):
        TaskSpec.model_validate(task)


def test_legacy_empty_fields_keep_original_serialization():
    task = configuration()["workload"]["task"]
    normalized = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    assert all("request_ref" not in a for a in normalized["actions"])
    assert all(
        "demand_actors" not in r and "demand_stations" not in r
        for r in normalized["roles"].values()
    )


@pytest.mark.parametrize("fault", ["duplicate", "backdated", "job"])
def test_invalid_intent_not_mutated(fault):
    value = configuration()
    base = value["workload"]["scenario"]
    before = copy.deepcopy(base)
    payload = value["updates"][0]["batch"]["additions"][0]
    event = {"at_sec": 102.0, "payload": payload}
    if fault == "duplicate":
        payload["request"]["id"] = base["definition"]["lifecycle"]["requests"][0]["id"]
    elif fault == "backdated":
        payload["request"]["release_sec"] = 100.0
    else:
        payload["jobs"][0]["id"] = base["definition"]["lifecycle"]["jobs"][0]["id"]
    with pytest.raises(ValueError):
        effective_scenario(base, [event])
    assert base == before


def test_missing_native_event_cannot_pass(completed):
    root, _, journal = completed
    report = copy.deepcopy(
        read(root / "execution.json")["execution"]["episode"]["runtime"][
            "backend_report"
        ]
    )
    report["lifecycle"]["events"] = []
    plan = journal["epochs"][-1]["result"]["plan"]
    _, evidence = demand_evidence(plan["scenario"], report, plan["goals"]["demand"])
    assert not evidence["passed"]
    assert not evidence["checks"]["committed_event_trace"]


def test_tight_deadline_retained_and_sft_suppressed(tmp_path):
    root = tmp_path / "negative"
    result = run(configuration("tight_deadline"), root)
    assert result["total"] == 3
    assert result["fulfilled"] == 2
    assert not result["mission_goal_passed"]
    assert result["actor_rows"]["rl"] > 0
    assert result["actor_rows"]["sft"] == 0


def test_checkpoint_continuation_exact(completed):
    root, _, _ = completed
    actor = ActorCoordinator.from_checkpoint(read(root / "actor-prefix.json"))
    while not actor.done:
        policy_step(actor)
    assert actor.checkpoint() == read(root / "controller.json")


def test_goal_floor_cannot_exceed_declared_denominator():
    value = configuration()
    event = {"at_sec": 102.0, "payload": value["updates"][0]["batch"]["additions"][0]}
    with pytest.raises(ValueError):
        DemandGoals(events=[event], min_fulfilled=2)
