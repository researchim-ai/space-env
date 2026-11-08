import copy
import json

import pytest

from space_env.api import operator
from space_env.api.network_dynamic import evolve_network
from space_env.api.network_workspace import default_network_configuration
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_post
from space_env.simulation.snapshot import digest
from space_env.workspace import MissionWorkspace


def payload(m, key="change", **kw):
    s = m.state()
    return {
        "reason": "Проверка продолжения",
        "idempotency_key": key,
        "expected_revision": s["revision"],
        "control_epoch": s["control"]["epoch"],
        **kw,
    }


@pytest.mark.parametrize(
    "decisions,phase,owner",
    [
        (1, "fresh", "fleet-00"),
        (20, "raw", "fleet-00"),
        (21, "final", "fleet-00"),
        (22, "final", "fleet-01"),
        (24, "final", "fleet-02"),
        (27, "completed", "fleet-02"),
    ],
)
def test_phase_and_custody_continuation_preserve_source(decisions, phase, owner):
    m = MissionWorkspace.network()
    m.run(decisions=decisions)
    before = m.operator.session.checkpoint(compact=True)
    n = m.evolve("Продолжение", idempotency_key="recover")
    w = n.state()["network"]
    st = w["dynamic"]["states"]["survey"]
    assert st["phase"] == phase and st["owner"] == owner
    assert w["plan_aligned"]
    if phase != "fresh":
        assert all(
            s["kind"] not in {"slew", "imaging"}
            for r in w["requests"]
            if r["plan"]
            for s in r["plan"]["stages"]
        )
    assert n.run()["network"]["fulfilled"] == 1
    assert m.operator.session.checkpoint(compact=True) == before
    report = n.operator.session._runtime._backend.report()
    assert report["lifecycle"]["requests"]["survey"]["attempts"] == 1
    assert len(report["lifecycle"]["acknowledgements"]) == 1


def test_partial_cpu_keeps_completed_work_and_raw_artifact():
    cfg = default_network_configuration()
    cfg["mission"]["requests"][0]["processing_sec"] = 12.0
    m = MissionWorkspace.network(cfg)
    m.run(decisions=21)
    previous = m.operator.session._runtime._backend.report()["lifecycle"]
    assert 0 < previous["jobs"]["cpu-survey"]["remaining_sec"] < 12
    n = m.evolve("Остаток CPU", idempotency_key="cpu")
    state = n.state()["network"]["dynamic"]["states"]["survey"]
    assert state["phase"] == "raw" and state["remaining_cpu_sec"] < 12
    assert n.run()["network"]["fulfilled"] == 1
    assert (
        n.operator.session._runtime._backend.report()["lifecycle"]["artifacts"][
            "raw-survey"
        ]["content_sha256"]
        == previous["artifacts"]["raw-survey"]["content_sha256"]
    )


def test_outage_and_recovery_replan_owned_final_without_recapture():
    m = MissionWorkspace.network()
    m.run(decisions=22)
    n = m.evolve(
        "Станция отказала",
        idempotency_key="outage",
        events=[
            {
                "type": "station_outage_added",
                "payload": {"station": "Ground", "start_sec": 115.0, "end_sec": 150.0},
            }
        ],
    )
    assert n.state()["network"]["planned"] == 0
    recovered = n.evolve(
        "Восстановлена",
        idempotency_key="back",
        events=[{"type": "station_recovered", "payload": {"station": "Ground"}}],
    )
    assert recovered.run()["network"]["fulfilled"] == 1
    assert n.run()["network"]["fulfilled"] == 0
    assert (
        recovered.operator.session._runtime._backend.report()["lifecycle"]["requests"][
            "survey"
        ]["attempts"]
        == 1
    )


def test_receiver_radio_fault_blocks_route_and_clear_restores_it():
    m = MissionWorkspace.network()
    m.run(decisions=22)
    n = m.evolve(
        "Отказ приёмника",
        idempotency_key="fault",
        events=[
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "fleet-02", "subsystem": "radio"},
            }
        ],
    )
    assert n.state()["network"]["planned"] == 0
    n = n.evolve(
        "Восстановить радио",
        idempotency_key="clear",
        events=[
            {
                "type": "spacecraft_fault_cleared",
                "payload": {"actor": "fleet-02", "subsystem": "radio"},
            }
        ],
    )
    assert n.run()["network"]["fulfilled"] == 1


def test_receiver_fault_selects_alternate_custody_route():
    cfg = default_network_configuration()
    cfg["mission"]["planes"][0]["phases_deg"].append(40.0)
    cfg["links"].append(
        {"source": "fleet-01", "destination": "fleet-03", "max_range_m": 5000000.0}
    )
    m = MissionWorkspace.network(cfg)
    assert m.state()["network"]["requests"][0]["plan"]["path"] == [
        "fleet-00",
        "fleet-01",
        "fleet-02",
    ]
    m.run(decisions=22)
    n = m.evolve(
        "Обойти неисправный приёмник",
        idempotency_key="reroute",
        events=[
            {
                "type": "spacecraft_fault",
                "payload": {"actor": "fleet-02", "subsystem": "radio"},
            }
        ],
    )
    assert n.state()["network"]["requests"][0]["plan"]["path"] == [
        "fleet-01",
        "fleet-03",
    ]
    assert n.run()["network"]["fulfilled"] == 1


def test_cancel_and_new_native_request_keep_full_denominator():
    m = MissionWorkspace.network()
    addition = copy.deepcopy(default_network_configuration()["mission"]["requests"][0])
    addition.update(id="replacement", actor_id="fleet-00")
    n = m.evolve(
        "Замена",
        idempotency_key="replace",
        additions=[addition],
        events=[{"type": "request_cancelled", "payload": {"request_id": "survey"}}],
    )
    assert len(n.operator.session._scenario["definition"]["lifecycle"]["requests"]) == 1
    result = n.run()["network"]
    assert result["total"] == 2 and result["fulfilled"] == 1
    report = n.operator.session._runtime._backend.report()["lifecycle"]
    assert report["requests"]["survey"]["status"] == "cancelled"
    assert report["requests"]["replacement"]["raw_size_bits"] == 1000000


def test_priority_deadline_are_native_and_source_unmodified():
    m = MissionWorkspace.network()
    n = m.evolve(
        "Приоритет",
        idempotency_key="priority",
        events=[
            {
                "type": "request_priority_changed",
                "payload": {"request_id": "survey", "priority": 99},
            },
            {
                "type": "request_deadline_changed",
                "payload": {"request_id": "survey", "deadline_sec": 145.0},
            },
        ],
    )
    row = n.operator.session._runtime._backend.report()["lifecycle"]["requests"][
        "survey"
    ]
    assert row["priority"] == 99 and row["deadline_sec"] == 145
    assert n.run()["network"]["fulfilled"] == 1
    assert m.operator.session.revision == 0


def test_deadline_extension_does_not_renew_existing_bundle_expiry():
    cfg = default_network_configuration()
    cfg["mission"]["requests"][0]["deadline_sec"] = 130.0
    m = MissionWorkspace.network(cfg)
    m.run(decisions=22)
    original = m.operator.session._runtime._backend.report()["dtn"][
        "bundle_expiry_sec"
    ]["final-survey"]
    n = m.evolve(
        "Продлить заявку, сохранив TTL",
        idempotency_key="ttl",
        events=[
            {
                "type": "request_deadline_changed",
                "payload": {"request_id": "survey", "deadline_sec": 145.0},
            },
            {
                "type": "station_outage_added",
                "payload": {"station": "Ground", "start_sec": 115.0, "end_sec": 135.0},
            },
        ],
    )
    meta = n.operator.session._scenario["metadata"]["network_workspace"]
    assert meta["configuration"]["mission"]["requests"][0]["deadline_sec"] == 145
    assert meta["forecast"]["demands"][0]["deadline_sec"] == original == 130
    assert n.state()["network"]["planned"] == 0
    assert n.run()["network"]["fulfilled"] == 0


def test_idempotency_authority_stale_control_and_atomic_validation():
    m = MissionWorkspace.network()
    store, sid, tokens = m.store, m.connection["session_id"], m.connection["tokens"]
    p = payload(m)
    for role in ("agent", "viewer", "role:fleet-00"):
        with pytest.raises(OperatorError) as e:
            evolve_network(store, sid, tokens[role], p)
        assert e.value.code == "forbidden"
    n = evolve_network(store, sid, tokens["operator"], p)
    assert (
        evolve_network(store, sid, tokens["operator"], p)["session_id"]
        == n["session_id"]
    )
    with pytest.raises(OperatorError) as e:
        evolve_network(store, sid, tokens["operator"], {**p, "reason": "Другой запрос"})
    assert e.value.code == "idempotency_conflict"
    before = digest(m.operator.session.checkpoint(compact=True))
    count = len(store.sessions)
    with pytest.raises(ValueError):
        evolve_network(
            store,
            sid,
            tokens["operator"],
            payload(
                m,
                "bad",
                events=[
                    {
                        "type": "spacecraft_fault",
                        "payload": {"actor": "missing", "subsystem": "radio"},
                    }
                ],
            ),
        )
    assert digest(m.operator.session.checkpoint(compact=True)) == before
    assert len(store.sessions) == count
    with pytest.raises(OperatorError) as e:
        evolve_network(
            store,
            sid,
            tokens["operator"],
            {**payload(m, "stale"), "expected_revision": 999},
        )
    assert e.value.code == "stale_control"


def test_api_durable_restore_and_role_privacy(tmp_path, monkeypatch):
    path = tmp_path / "network.sqlite3"
    store = OperatorStore(path)
    m = MissionWorkspace.network(store=store)
    m.run(decisions=22)
    monkeypatch.setattr(operator, "STORE", store)
    status, raw, _ = handle_post(
        f"/api/operator/sessions/{m.connection['session_id']}/evolve",
        json.dumps(payload(m)).encode(),
        token=m.connection["tokens"]["operator"],
    )
    assert status == 200, raw
    result = json.loads(raw)
    op = OperatorStore(path).get(result["session_id"])
    assert op.state(result["tokens"]["operator"])["network"]["plan_aligned"]
    assert "network" not in op.state(result["tokens"]["agent"])
    assert "network" not in op.state(result["tokens"]["role:fleet-01"])


def test_partial_isl_cancel_keeps_one_custodian_and_accounts_wasted_bits():
    cfg = default_network_configuration()
    cfg["mission"]["decision_sec"] = 1.0
    m = MissionWorkspace.network(cfg)
    stage = next(
        s
        for s in m.state()["network"]["requests"][0]["plan"]["stages"]
        if s["kind"] == "relay"
    )
    m.run(decisions=stage["cell"] + 1)
    before = m.operator.session._runtime._backend.report()
    assert any(
        t["status"] not in {"completed", "failed", "cancelled", "expired"}
        for t in before["dtn"]["transfers"].values()
    )
    n = m.evolve("Прервать ISL", idempotency_key="partial")
    st = n.state()["network"]["dynamic"]["states"]["survey"]
    assert st["phase"] == "final" and st["owner"] == "fleet-00"
    report = n.operator.session._runtime._backend.report()
    assert sum(report["dtn"]["tx_serialized_bits"].values()) > 0
    assert abs(report["dtn"]["wire_balance_error_bits"]) < 1e-6
    assert any(t["status"] == "cancelled" for t in report["dtn"]["transfers"].values())
    assert (
        sum(
            p["remaining_bits"]
            for a in n.operator.session._scenario["actor_order"]
            for p in n.operator.session._runtime.inventory(a)["products"]
            if p["id"] == "final-survey"
        )
        == 500000
    )
    assert n.run()["network"]["fulfilled"] == 1
    assert (
        n.operator.session._runtime._backend.report()["lifecycle"]["requests"][
            "survey"
        ]["attempts"]
        == 1
    )


def test_partial_ground_tx_resumes_remaining_bits_and_cannot_relay():
    cfg = default_network_configuration()
    cfg["mission"]["decision_sec"] = 1.0
    m = MissionWorkspace.network(cfg)
    stage = next(
        s
        for s in m.state()["network"]["requests"][0]["plan"]["stages"]
        if s["kind"] == "downlink"
    )
    m.run(decisions=stage["cell"] + 1)
    n = m.evolve("Остаток передачи", idempotency_key="partial-ground")
    w = n.state()["network"]
    st = w["dynamic"]["states"]["survey"]
    assert (
        st["phase"] == "final"
        and 0 < st["remaining_bits"] < 500000
        and not st["intact"]
    )
    assert all(s["kind"] == "downlink" for s in w["requests"][0]["plan"]["stages"])
    assert n.run()["network"]["fulfilled"] == 1
    receipts = n.operator.session._runtime._backend.report()["receipts"]
    assert (
        sum(r["bits"] for r in receipts if r["product_id"] == "final-survey") == 500000
    )
