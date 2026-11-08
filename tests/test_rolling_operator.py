"""Causal policy persistence, historical binding, and operator/agent isolation."""

import copy
import json
from pathlib import Path

import pytest

from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.rolling_planner import rebuild, stage
from space_env.simulation.snapshot import digest
from space_env.workspace import MissionWorkspace


def choose(op, token):
    return op.suggest(token)


def step(op, token, ids, key):
    return op.step(
        token,
        {
            "action_ids": ids,
            "expected_revision": op.session.revision,
            "control_epoch": op.control["epoch"],
            "idempotency_key": key,
        },
    )


def test_suggestion_does_not_change_physics_and_repeats_are_identical():
    m = MissionWorkspace.relay()
    op, token = m.operator, m.connection["tokens"]["operator"]
    before = op.session.checkpoint(compact=True)
    first = choose(op, token)
    assert first == choose(op, token)
    assert op.session.checkpoint(compact=True) == before
    assert op.planner_state["policy"]["revision"] == 0
    assert first["information"].startswith("current public")


def test_actual_rejected_choice_not_unexecuted_proposal_gets_cooldown():
    m = MissionWorkspace.relay()
    op, token = m.operator, m.connection["tokens"]["operator"]
    first = choose(op, token)
    assert "compute" not in first["action_ids"]
    result = step(op, token, ["compute"], "manual-reject")
    assert not result["feedback"]["accepted"]
    assert op.planner_state["executed"]
    assert op.planner_state["policy"]["last_decision"]["action_ids"] == ["compute"]
    choose(op, token)
    cooldown = op.planner_state["policy"]["cooldown"]
    assert "compute" in cooldown
    assert not set(first["action_ids"]) & set(cooldown)
    assert op.planner_state["policy"] == rebuild(op.session, proposed=True).checkpoint()


def test_skipped_suggestions_and_manual_steps_preserve_consecutive_policy_history():
    m = MissionWorkspace.relay()
    op, token = m.operator, m.connection["tokens"]["operator"]
    choose(op, token)
    for i in range(3):
        step(op, token, ["wait"], f"manual/{i}")
    assert op.planner_state["policy"]["revision"] == 2
    choose(op, token)
    assert op.planner_state["policy"]["revision"] == 3
    assert op.planner_state["policy"] == rebuild(op.session, proposed=True).checkpoint()


def test_first_activation_after_manual_prefix_bootstraps_from_actual_history():
    m = MissionWorkspace.relay()
    op, token = m.operator, m.connection["tokens"]["operator"]
    step(op, token, ["compute"], "rejected")
    step(op, token, ["wait"], "coast")
    assert op.planner_state is None
    choose(op, token)
    assert op.planner_state["policy"]["revision"] == 2
    assert op.planner_state["policy"] == rebuild(op.session, proposed=True).checkpoint()


@pytest.mark.parametrize("proposed", [False, True])
def test_durable_restart_keeps_same_next_decision(tmp_path, proposed):
    path = tmp_path / "rolling.sqlite3"
    m = MissionWorkspace.relay(store=OperatorStore(path))
    m.run(decisions=2)
    token = m.connection["tokens"]["operator"]
    if proposed:
        choose(m.operator, token)
    state = copy.deepcopy(m.operator.planner_state)
    expected = stage(m.operator)[1]
    reopened = OperatorStore(path).get(m.connection["session_id"])
    assert reopened.planner_state == state
    assert choose(reopened, token) == expected


def test_failed_storage_commit_rolls_back_only_planner_cache(tmp_path, monkeypatch):
    path = tmp_path / "rolling.sqlite3"
    m = MissionWorkspace.relay(store=OperatorStore(path))
    op, token = m.operator, m.connection["tokens"]["operator"]
    before = op.session.checkpoint(compact=True)

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(op.repository, "save", fail)
    with pytest.raises(OperatorError) as e:
        choose(op, token)
    assert e.value.status == 503
    assert op.planner_state is None
    assert op.session.checkpoint(compact=True) == before


def test_archive_roundtrip_and_legacy_archive_without_planner():
    m = MissionWorkspace.relay()
    m.run(decisions=2)
    archive = m.operator.archive(m.connection["tokens"]["admin"])
    reopened = OperatorStore().restore({"archive": archive})
    assert reopened["state"]["revision"] == 2
    legacy = copy.deepcopy(archive)
    legacy.pop("planner_state")
    legacy["sha256"] = digest({k: v for k, v in legacy.items() if k != "sha256"})
    store = OperatorStore()
    restored = store.restore({"archive": legacy})
    op = store.get(restored["session_id"])
    assert op.planner_state is None
    assert choose(op, restored["tokens"]["operator"])["expected_revision"] == 2


def test_resealed_policy_tamper_rejected_by_native_history_recomputation():
    m = MissionWorkspace.relay()
    m.run(decisions=2)
    archive = m.operator.archive(m.connection["tokens"]["admin"])
    cp = archive["planner_state"]["policy"]
    cp["cooldown"]["fake-action"] = 99
    cp["sha256"] = digest({k: v for k, v in cp.items() if k != "sha256"})
    archive["sha256"] = digest({k: v for k, v in archive.items() if k != "sha256"})
    with pytest.raises(ValueError, match="native history"):
        OperatorStore().restore({"archive": archive})


@pytest.mark.parametrize("profile", ["workspace", "relay"])
def test_agent_viewer_and_actor_cannot_query_operator_planner(profile):
    m = MissionWorkspace() if profile == "workspace" else MissionWorkspace.relay()
    roles = [
        "agent",
        "viewer",
        "admin",
        next(k for k in m.connection["tokens"] if k.startswith("role:")),
    ]
    before = m.operator.session.checkpoint(compact=True)
    for role in roles:
        with pytest.raises(OperatorError) as e:
            choose(m.operator, m.connection["tokens"][role])
        assert e.value.status == 403
    assert m.operator.planner_state is None
    assert m.operator.session.checkpoint(compact=True) == before


def test_native_clock_idempotency_does_not_advance_planner_twice():
    m = MissionWorkspace.relay()
    token = m.connection["tokens"]["operator"]
    decision = choose(m.operator, token)
    request = {
        "action_ids": decision["action_ids"],
        "expected_revision": 0,
        "control_epoch": 0,
        "idempotency_key": "same",
    }
    result = m.operator.step(token, request)
    state = copy.deepcopy(m.operator.planner_state)
    assert m.operator.step(token, request) == result
    assert m.operator.planner_state == state and m.operator.session.revision == 1


def test_observed_failed_isl_switches_to_alternative_without_future_outage_oracle():
    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/space_env/configs/operator_relay.json"
        ).read_text()
    )
    payload = {k: raw[k] for k in ("scenario", "task")}
    for link in payload["scenario"]["definition"]["operations"]["dtn_network"]["links"]:
        if (link["source"], link["destination"]) == ("a", "b"):
            link["outages"] = [[0.0, 20.0]]
    store = OperatorStore()
    created = store.create(payload)
    op = store.get(created["session_id"])
    token = created["tokens"]["operator"]
    trace = []
    while not op.session._done:
        d = choose(op, token)
        trace.append(d["action_ids"])
        step(op, token, d["action_ids"], str(op.session.revision))
    assert ["hop-second-b"] in trace and ["hop-second-c"] in trace
    assert trace.index(["hop-second-c"]) > trace.index(["hop-second-b"])
    report = op.session._runtime._backend.report()["lifecycle"]
    assert len(report["requests"]) == 2
    # A route proposal is not an ACK: the late second delivery remains a miss.
    assert sum(r["status"] == "completed" for r in report["requests"].values()) == 1
