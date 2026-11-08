"""A new plan preserves native prefix and leaves its source episode untouched."""

import json

import pytest

from space_env.api import operator
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_post
from space_env.api.workspace_replan import replan_workspace
from space_env.simulation.snapshot import digest
from space_env.workspace import MissionWorkspace, default_configuration


def mission(store=None, horizon=120.0):
    c = default_configuration()
    c["horizon_sec"] = horizon
    for r in c["requests"]:
        r["deadline_sec"] = horizon
    return MissionWorkspace(c, store=store)


def wait(m, seconds=5.0):
    state = m.state()
    return m.operator.step(
        m.connection["tokens"]["operator"],
        {
            "action_ids": ["wait"],
            "expected_revision": state["revision"],
            "advance_sec": seconds,
            "control_epoch": state["control"]["epoch"],
            "idempotency_key": f"manual/{state['revision']}",
        },
    )


def payload(m, key="replan"):
    state = m.state()
    return {
        "reason": "Пересчитать после ожидания",
        "expected_revision": state["revision"],
        "control_epoch": state["control"]["epoch"],
        "idempotency_key": key,
    }


def test_wait_deviation_replans_and_delivers_without_mutating_source(tmp_path):
    m = mission()
    wait(m)
    source = m.operator.session.checkpoint(compact=True)
    assert not m.state()["workspace"]["plan_aligned"]
    n = m.replan("Пересчёт после ручного ожидания", idempotency_key="fork")
    assert n.connection["session_id"] != m.connection["session_id"]
    assert n.state()["time_sec"] == 5.0 and n.state()["revision"] == 1
    assert n.state()["workspace"]["plan_aligned"]
    assert (
        n.state()["workspace"]["replan"]["source_checkpoint_sha256"] == source["sha256"]
    )
    assert (
        n.operator.session._runtime._backend.report()
        == m.operator.session._runtime._backend.report()
    )
    assert n.run()["workspace"]["fulfilled"] == 2
    assert m.operator.session.checkpoint(compact=True) == source
    n.export(tmp_path / "continuation.zip")
    assert n.operator.verify(n.connection["tokens"]["admin"])


def test_idempotent_retry_and_conflicting_key():
    m = mission()
    wait(m)
    p = payload(m)
    args = m.store, m.connection["session_id"], m.connection["tokens"]["operator"]
    first = replan_workspace(*args, p)
    again = replan_workspace(*args, p)
    assert again["session_id"] == first["session_id"]
    assert again["tokens"] == first["tokens"]
    assert len(m.store.sessions) == 2
    with pytest.raises(OperatorError, match="Ключ"):
        replan_workspace(*args, {**p, "reason": "Другая причина"})


@pytest.mark.parametrize("role", ["viewer", "agent", "admin", "role:a-00"])
def test_unauthorized_channels_cannot_replan(role):
    m = mission()
    wait(m)
    with pytest.raises(OperatorError) as e:
        replan_workspace(
            m.store,
            m.connection["session_id"],
            m.connection["tokens"][role],
            payload(m),
        )
    assert e.value.status == 403
    assert len(m.store.sessions) == 1


def test_stale_revision_and_control_epoch():
    m = mission()
    wait(m)
    for change in ({"expected_revision": 0}, {"control_epoch": 3}):
        with pytest.raises(OperatorError) as e:
            replan_workspace(
                m.store,
                m.connection["session_id"],
                m.connection["tokens"]["operator"],
                {**payload(m), **change},
            )
        assert e.value.code == "stale_control"


def test_repeated_replanning_after_wait_preserves_lineage():
    m = mission()
    wait(m)
    n = m.replan("Первый пересчёт", idempotency_key="first")
    wait(n)
    before = n.operator.session.checkpoint(compact=True)
    third = n.replan("Второй пересчёт", idempotency_key="second")
    w = third.state()["workspace"]
    assert w["replan"]["source_session_id"] == n.connection["session_id"]
    assert w["replan"]["prefix_time_sec"] == 10.0
    assert w["plan_aligned"]
    assert n.operator.session.checkpoint(compact=True) == before


def test_planning_failure_never_replaces_source():
    m = mission()
    wait(m)
    before = m.operator.session.checkpoint(compact=True)
    with pytest.raises(ValueError, match="Лимит"):
        m.replan(
            "Малый бюджет", idempotency_key="small", planner={"max_search_nodes": 1}
        )
    assert m.operator.session.checkpoint(compact=True) == before
    assert len(m.store.sessions) == 1


def test_initial_boundary_and_blank_reason_are_rejected():
    m = mission()
    assert not m.state()["workspace"]["replan_available"]
    with pytest.raises(OperatorError, match="начатая"):
        m.replan("До начала", idempotency_key="initial")
    wait(m)
    with pytest.raises(ValueError, match="причину"):
        m.replan("  ", idempotency_key="blank")


def test_started_chain_and_off_grid_boundaries_fail_without_changes():
    m = mission()
    m.step()  # paid slews, unfinished chain
    before = m.operator.session.checkpoint(compact=True)
    with pytest.raises(OperatorError, match="цепочки"):
        m.replan("Нельзя прерывать", idempotency_key="active")
    assert m.operator.session.checkpoint(compact=True) == before
    other = mission()
    wait(other, 2.0)
    assert not other.state()["workspace"]["replan_available"]
    with pytest.raises(OperatorError, match="сетки"):
        other.replan("Вне сетки", idempotency_key="off-grid")


def test_completed_chain_is_retained_and_not_executed_again():
    c = default_configuration()
    c["horizon_sec"] = 240.0
    c["requests"][0]["deadline_sec"] = 120.0
    c["requests"][1].update(release_sec=150.0, deadline_sec=240.0)
    m = MissionWorkspace(c)
    assert m.run(decisions=22)["workspace"]["fulfilled"] == 1
    before = digest(m.operator.session._runtime._backend.report())
    n = m.replan("Оставить доставленную заявку", idempotency_key="completed")
    w = n.state()["workspace"]
    assert w["replan"]["completed"] == ["survey-west"]
    assert w["fulfilled"] == 1 and w["plan_aligned"]
    assert digest(n.operator.session._runtime._backend.report()) == before
    future = n.operator.session._scenario["metadata"]["workspace"]["schedule"]["steps"][
        22:
    ]
    assert not any("survey-west" in a for s in future for a in s["action_ids"])


def test_durable_branch_restore_and_retry_does_not_create_new_branch(tmp_path):
    path = tmp_path / "operator.sqlite3"
    m = mission(OperatorStore(path))
    wait(m)
    p = payload(m)
    created = replan_workspace(
        m.store, m.connection["session_id"], m.connection["tokens"]["operator"], p
    )
    reopened = OperatorStore(path)
    n = reopened.get(created["session_id"])
    assert n.state(created["tokens"]["operator"])["workspace"]["plan_aligned"]
    assert n.suggest(created["tokens"]["operator"])["expected_revision"] == 1
    with pytest.raises(OperatorError) as e:
        replan_workspace(
            reopened, m.connection["session_id"], m.connection["tokens"]["operator"], p
        )
    assert e.value.code == "replan_already_created"
    assert len(reopened.sessions) == 2


def test_failed_branch_install_keeps_source_unchanged(monkeypatch):
    m = mission()
    wait(m)
    before = m.operator.session.checkpoint(compact=True)

    def fail(_):
        raise OSError("disk failure")

    monkeypatch.setattr(m.store, "restore", fail)
    with pytest.raises(OSError, match="disk"):
        m.replan("Ошибка сохранения", idempotency_key="failure")
    assert m.operator.session.checkpoint(compact=True) == before
    assert len(m.store.sessions) == 1


def test_http_route_uses_existing_capability_and_returns_new_session(monkeypatch):
    m = mission()
    wait(m)
    monkeypatch.setattr(operator, "STORE", m.store)
    endpoint = f"/api/operator/sessions/{m.connection['session_id']}/replan"
    status, body, _ = handle_post(
        endpoint,
        json.dumps(payload(m)).encode(),
        token=m.connection["tokens"]["operator"],
    )
    result = json.loads(body)
    assert status == 200 and result["state"]["time_sec"] == 5.0
    assert (
        result["state"]["workspace"]["replan"]["source_session_id"]
        == m.connection["session_id"]
    )
    assert "workspace" not in m.store.get(result["session_id"]).state(
        result["tokens"]["agent"]
    )
