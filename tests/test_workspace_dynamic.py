import copy
import json

import pytest

from space_env.api import operator
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_post
from space_env.api.workspace_dynamic import evolve_workspace
from space_env.workspace import MissionWorkspace, default_configuration


def mission(store=None):
    c = default_configuration()
    c["horizon_sec"] = 180.0
    for r in c["requests"]:
        r["deadline_sec"] = 180.0
    return MissionWorkspace(c, store=store)


@pytest.mark.parametrize("decisions,phase", [(1, "fresh"), (20, "raw"), (21, "final")])
def test_phase_recovery_preserves_source_products_and_delivers(decisions, phase):
    m = mission()
    m.run(decisions=decisions)
    before = m.operator.session.checkpoint(compact=True)
    n = m.evolve("Восстановить этапы", idempotency_key="recover")
    w = n.state()["workspace"]
    assert {s["phase"] for s in w["dynamic"]["states"].values()} == {phase}
    assert w["plan_aligned"]
    assert n.run()["workspace"]["fulfilled"] == 2
    assert m.operator.session.checkpoint(compact=True) == before
    if phase != "fresh":
        stages = n.operator.session._scenario["metadata"]["workspace"]["schedule"][
            "steps"
        ][decisions:]
        assert not any("/imaging" in a for s in stages for a in s["action_ids"])


def test_new_request_is_native_event_not_preloaded_and_cancelled_stays_in_denominator():
    m = mission()
    replacement = copy.deepcopy(default_configuration()["requests"][0])
    replacement.update(id="replacement", release_sec=0.0, deadline_sec=180.0)
    n = m.evolve(
        "Заменить заявку",
        idempotency_key="replace",
        additions=[replacement],
        events=[
            {"type": "request_cancelled", "payload": {"request_id": "survey-west"}}
        ],
    )
    assert len(n.operator.session._scenario["definition"]["lifecycle"]["requests"]) == 2
    assert len(n.state()["workspace"]["requests"]) == 3
    result = n.run()["workspace"]
    assert result["total"] == 3 and result["fulfilled"] == 2
    assert (
        next(r for r in result["requests"] if r["id"] == "survey-west")["status"]
        == "cancelled"
    )
    assert (
        next(r for r in result["requests"] if r["id"] == "replacement")["status"]
        == "completed"
    )


def test_priority_deadline_station_outage_and_recovery():
    m = mission()
    n = m.evolve(
        "Переоценка и отказ",
        idempotency_key="events",
        events=[
            {
                "type": "request_priority_changed",
                "payload": {"request_id": "survey-west", "priority": 99},
            },
            {
                "type": "request_deadline_changed",
                "payload": {"request_id": "survey-west", "deadline_sec": 170.0},
            },
            {
                "type": "station_outage_added",
                "payload": {"station": "West", "start_sec": 0.0, "end_sec": 180.0},
            },
        ],
    )
    assert n.state()["workspace"]["planned"] == 1
    west = next(
        r for r in n.state()["workspace"]["requests"] if r["id"] == "survey-west"
    )
    assert west["priority"] == 99 and west["deadline_sec"] == 170.0
    recovered = n.evolve(
        "Станция восстановлена",
        idempotency_key="station-back",
        events=[{"type": "station_recovered", "payload": {"station": "West"}}],
    )
    assert recovered.run()["workspace"]["fulfilled"] == 2


def test_fault_blocks_remaining_work_without_fabricating_native_failure_and_clear_recovers():
    m = mission()
    m.run(decisions=20)
    n = m.evolve(
        "Отказ радио",
        idempotency_key="fault",
        events=[
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "a-00", "subsystem": "radio"},
            }
        ],
    )
    w = n.state()["workspace"]
    assert w["dynamic"]["states"]["survey-west"]["phase"] == "blocked"
    assert w["planned"] == 1
    assert next(r for r in w["requests"] if r["id"] == "survey-west")["status"] not in {
        "failed",
        "cancelled",
    }
    recovered = n.evolve(
        "Радио восстановлено",
        idempotency_key="clear",
        events=[
            {
                "type": "spacecraft_fault_cleared",
                "payload": {"actor": "a-00", "subsystem": "radio"},
            }
        ],
    )
    assert recovered.run()["workspace"]["fulfilled"] == 2


def test_durable_continuation_and_export(tmp_path):
    path = tmp_path / "mission.sqlite3"
    m = mission(OperatorStore(path))
    m.run(decisions=20)
    n = m.evolve("Остатки", idempotency_key="durable")
    reopened = OperatorStore(path).get(n.connection["session_id"])
    assert reopened.state(n.connection["tokens"]["operator"])["workspace"][
        "plan_aligned"
    ]
    n.run()
    n.export(tmp_path / "dynamic.zip")
    assert n.operator.verify(n.connection["tokens"]["admin"])


@pytest.mark.parametrize("fault", ["unknown_target", "backdated", "invalid_event"])
def test_invalid_mutation_is_atomic(fault):
    m = mission()
    m.run(decisions=1)
    before = m.operator.session.checkpoint(compact=True)
    kwargs = {}
    if fault == "invalid_event":
        kwargs["events"] = [
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "a-00", "subsystem": "unknown"},
            }
        ]
    else:
        r = copy.deepcopy(default_configuration()["requests"][0])
        r.update(id="new", release_sec=10.0, deadline_sec=180.0)
        if fault == "unknown_target":
            r["lat_deg"] = 33.0
        else:
            r["release_sec"] = 0.0
        kwargs["additions"] = [r]
    with pytest.raises(ValueError):
        m.evolve("Неверное изменение", idempotency_key="bad", **kwargs)
    assert m.operator.session.checkpoint(compact=True) == before
    assert len(m.store.sessions) == 1


def test_http_retry_and_agent_cannot_inject_events(monkeypatch):
    m = mission()
    monkeypatch.setattr(operator, "STORE", m.store)
    body = {
        "reason": "Изменение",
        "expected_revision": 0,
        "control_epoch": 0,
        "idempotency_key": "http",
        "events": [
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "a-00", "subsystem": "radio"},
            }
        ],
    }
    endpoint = f"/api/operator/sessions/{m.connection['session_id']}/evolve"
    status, raw, _ = handle_post(
        endpoint, json.dumps(body).encode(), token=m.connection["tokens"]["operator"]
    )
    assert status == 200
    created = json.loads(raw)
    _, raw2, _ = handle_post(
        endpoint, json.dumps(body).encode(), token=m.connection["tokens"]["operator"]
    )
    assert json.loads(raw2)["session_id"] == created["session_id"]
    n = m.store.get(created["session_id"])
    event = next(a.id for a in n.session._task.actions if a.kind == "inject")
    n.handover(
        created["tokens"]["admin"],
        {
            "controller": "agent",
            "reason": "Проверка",
            "expected_revision": 1,
            "expected_control_epoch": 1,
            "idempotency_key": "handover",
        },
    )
    with pytest.raises(OperatorError) as e:
        n.step(
            created["tokens"]["agent"],
            {
                "action_ids": [event],
                "expected_revision": 1,
                "control_epoch": 2,
                "idempotency_key": "forbidden",
            },
        )
    assert e.value.status == 403


def test_foreign_or_agent_capability_cannot_evolve():
    m = mission()
    payload = {
        "reason": "Запрещено",
        "expected_revision": 0,
        "control_epoch": 0,
        "idempotency_key": "no",
    }
    for role in ("agent", "viewer", "admin", "role:a-00"):
        with pytest.raises(OperatorError):
            evolve_workspace(
                m.store,
                m.connection["session_id"],
                m.connection["tokens"][role],
                payload,
            )


@pytest.mark.parametrize("kind", ["cpu", "tx"])
def test_partial_native_progress_survives_interrupt(kind):
    c = default_configuration()
    c["horizon_sec"] = 240.0
    c["requests"] = c["requests"][:1]
    c["requests"][0]["deadline_sec"] = 240.0
    if kind == "cpu":
        c["requests"][0]["processing_sec"] = 20.0
        decisions = 21
    else:
        c["requests"][0]["output_bits"] = 20000000
        decisions = 22
    m = MissionWorkspace(c)
    m.run(decisions=decisions)
    n = m.evolve("Частичное выполнение", idempotency_key="partial")
    state = n.state()["workspace"]["dynamic"]["states"]["survey-west"]
    if kind == "cpu":
        assert state["phase"] == "raw" and 0 < state["remaining_cpu_sec"] < 20.0
    else:
        assert state["phase"] == "final" and 0 < state["remaining_bits"] < 20000000
    assert n.run()["workspace"]["fulfilled"] == 1


def test_role_capability_cannot_inject_operator_fault_even_before_handover():
    m = mission()
    n = m.evolve(
        "Отказ",
        idempotency_key="fault-role",
        events=[
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "a-00", "subsystem": "radio"},
            }
        ],
    )
    action = next(a.id for a in n.operator.session._task.actions if a.kind == "inject")
    with pytest.raises(OperatorError) as e:
        n.operator.step(
            n.connection["tokens"]["role:a-00"],
            {
                "action_ids": [action],
                "expected_revision": 1,
                "control_epoch": 1,
                "idempotency_key": "no-fault",
            },
        )
    assert e.value.status == 403


def test_operator_can_cancel_and_reprioritize_declared_unreleased_demand():
    c = default_configuration()
    c["requests"][0]["release_sec"] = 120.0
    m = MissionWorkspace(c)
    n = m.evolve(
        "Изменить будущую заявку",
        idempotency_key="pending",
        events=[
            {
                "type": "request_priority_changed",
                "payload": {"request_id": "survey-west", "priority": 9},
            },
            {"type": "request_cancelled", "payload": {"request_id": "survey-west"}},
        ],
    )
    row = next(
        r for r in n.state()["workspace"]["requests"] if r["id"] == "survey-west"
    )
    assert row["status"] == "cancelled" and row["priority"] == 9
    assert not any(
        e["id"] == "survey-west"
        for e in n.operator.state(n.connection["tokens"]["role:a-00"])["observation"][
            "entities"
        ]
    )


def test_demand_authority_read_requires_operator_access():
    from space_env.runtime import AccessPolicy, MissionRuntime
    from space_env.workspace import compile_configuration

    _, scenario, _ = compile_configuration(default_configuration())
    runtime = MissionRuntime(scenario, access=AccessPolicy())
    with pytest.raises(ValueError, match="authority"):
        runtime.request_authority("survey-west")


@pytest.mark.parametrize("request_id", [[], {}, None, 3, ""])
def test_invalid_cancel_payload_is_validation_error_and_preserves_source(request_id):
    m = mission()
    before = m.operator.session.checkpoint(compact=True)
    with pytest.raises(ValueError):
        m.evolve(
            "Неверная отмена",
            idempotency_key="invalid-id",
            events=[
                {"type": "request_cancelled", "payload": {"request_id": request_id}}
            ],
        )
    assert m.operator.session.checkpoint(compact=True) == before
