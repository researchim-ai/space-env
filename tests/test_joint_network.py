import copy
import json
import zipfile

import pytest

from space_env.api import operator
from space_env.api.network_workspace import (
    NetworkConfiguration,
    collect_network_forecast,
    default_network_configuration,
)
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_get, handle_post
from space_env.planning.joint_network import schedule_network
from space_env.runtime import MissionRuntime
from space_env.simulation.snapshot import digest
from space_env.workspace import MissionWorkspace


def seal(v):
    v["sha256"] = digest({k: x for k, x in v.items() if k != "sha256"})
    return v


def view():
    r = {
        "id": "survey",
        "owners": ["a", "b"],
        "stations": ["g"],
        "release_sec": 0.0,
        "deadline_sec": 8.0,
        "priority": 5,
        "raw_bits": 100,
        "output_bits": 50,
        "slew_cells": 1,
        "capture_cells": 1,
        "cpu_cells": 1,
        "relay_cells": 1,
        "ground_cells": 1,
        "owner_wh": 1.0,
        "relay_wh": 1.0,
        "receiver_wh": 0.5,
        "ground_wh": 1.0,
    }
    return seal(
        {
            "schema": "space-env.joint-network-view/1",
            "horizon_sec": 8.0,
            "decision_sec": 1.0,
            "actors": {
                a: {"memory_bits": 1e6, "usable_wh": 100.0} for a in ("a", "b", "c")
            },
            "stations": {"g": 1},
            "demands": [r],
            "edges": [
                {"source": a, "destination": b, "cells": [True] * 8}
                for a, b in (("a", "b"), ("b", "c"), ("c", "g"))
            ],
            "captures": [
                {"request": "survey", "actor": a, "cells": [True] * 8}
                for a in ("a", "b")
            ],
            "geometry_sha256": "0" * 64,
        }
    )


def test_joint_owner_route_and_manual_owner_are_selected_together():
    v = view()
    original = copy.deepcopy(v)
    p = schedule_network(v)
    assert p["allocations"][0]["owner"] == "b"
    assert p["allocations"][0]["path"] == ["b", "c"]
    assert v == original
    v["demands"][0]["owners"] = ["a"]
    v["captures"] = v["captures"][:1]
    p = schedule_network(seal(v))
    assert p["allocations"][0]["path"] == ["a", "b", "c"]
    assert not schedule_network(v, max_hops=1)["allocations"]


def test_station_capacity_memory_and_energy_are_shared():
    v = view()
    v["demands"][0].update(owners=["a"], deadline_sec=4.0)
    second = {**v["demands"][0], "id": "second", "owners": ["b"], "priority": 1}
    v["demands"].append(second)
    v["captures"] = [
        {"request": rid, "actor": a, "cells": [True] * 8}
        for rid, a in (("survey", "a"), ("second", "b"))
    ]
    v["edges"] = [
        {"source": a, "destination": "g", "cells": [True] * 8} for a in ("a", "b")
    ]
    assert schedule_network(seal(v))["predicted_fulfilled"] == 1
    v["stations"]["g"] = 2
    assert schedule_network(seal(v))["predicted_fulfilled"] == 2
    v["actors"]["b"]["usable_wh"] = 1.0
    assert schedule_network(seal(v))["predicted_fulfilled"] == 1
    v["actors"]["b"]["usable_wh"] = 100.0
    v["actors"]["b"]["memory_bits"] = 100.0
    assert schedule_network(seal(v))["predicted_fulfilled"] == 1


def test_shared_intermediate_storage_retains_full_request_denominator():
    v = view()
    v["horizon_sec"] = 16.0
    v["demands"][0]["deadline_sec"] = 16.0
    for edge in v["edges"]:
        edge["cells"] = [True] * 16
    v["demands"][0]["owners"] = ["b"]
    v["demands"].append({**v["demands"][0], "id": "second"})
    v["captures"] = [
        {"request": rid, "actor": "b", "cells": [True] * 16}
        for rid in ("survey", "second")
    ]
    assert schedule_network(seal(v))["predicted_fulfilled"] == 2
    v["actors"]["c"]["memory_bits"] = 75.0
    result = schedule_network(seal(v))
    assert result["total"] == 2 and result["predicted_fulfilled"] == 1


def test_budget_is_exact_and_only_complete_chains_can_be_returned():
    p = schedule_network(view(), max_nodes=1)
    assert p["search_budget_exhausted"] and p["search_nodes"] == 1
    assert p["total"] == 1 and not p["allocations"]
    assert p == schedule_network(view(), max_nodes=1)
    with pytest.raises(ValueError):
        schedule_network(view(), max_nodes=True)


@pytest.mark.parametrize(
    "fault",
    ["seal", "boolean_capacity", "integer_cell", "duplicate_owner", "missing_capture"],
)
def test_invalid_forecast_is_rejected(fault):
    v = view()
    if fault == "seal":
        v["sha256"] = "1" * 64
    elif fault == "boolean_capacity":
        v["stations"]["g"] = True
        seal(v)
    elif fault == "integer_cell":
        v["edges"][0]["cells"][0] = 1
        seal(v)
    elif fault == "duplicate_owner":
        v["demands"][0]["owners"] = ["a", "a"]
        seal(v)
    else:
        v["captures"] = []
        seal(v)
    with pytest.raises(ValueError):
        schedule_network(v)


def test_collection_never_submits_or_advances_native_future(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native future rollout during search")

    monkeypatch.setattr(MissionRuntime, "advance", forbidden)
    monkeypatch.setattr(MissionRuntime, "submit", forbidden)
    spec = NetworkConfiguration.model_validate(default_network_configuration())
    result = schedule_network(collect_network_forecast(spec))
    assert result["allocations"][0]["path"] == ["fleet-00", "fleet-01", "fleet-02"]


def test_subsample_outage_cannot_disappear_from_forecast_cells():
    cfg = default_network_configuration()
    cfg["links"][0]["outages"] = [[0.1, 0.2]]
    v = collect_network_forecast(NetworkConfiguration.model_validate(cfg))
    assert not next(
        e
        for e in v["edges"]
        if e["source"] == "fleet-00" and e["destination"] == "fleet-01"
    )["cells"][0]


@pytest.fixture(scope="module")
def finished():
    m = MissionWorkspace.network()
    initial = m.state()["network"]
    result = m.run()
    return m, initial, result


def test_native_capture_cpu_two_custody_hops_and_ground_ack(finished):
    m, initial, result = finished
    assert initial["planned"] == 1 and initial["fulfilled"] == 0
    assert result["network"]["fulfilled"] == result["network"]["total"] == 1
    plan = initial["requests"][0]["plan"]
    assert plan["owner"] == "fleet-00" and plan["path"] == [
        "fleet-00",
        "fleet-01",
        "fleet-02",
    ]
    states = m.operator.session._runtime.command_status()
    assert [r["command"]["kind"] for r in states].count("relay") == 2
    assert all(r["status"] == "completed" for r in states)
    report = m.operator.session._runtime._backend.report()
    assert report["lifecycle"]["acknowledgements"]
    assert report["lifecycle"]["acknowledgements"][0]["bits"] == 500000
    assert report["lifecycle"]["requests"]["survey"]["raw_size_bits"] == 1000000
    assert not m.operator.session._scenario["definition"]["operations"]["products"]


def test_native_export_has_no_forecast_in_learner_rows(finished, tmp_path):
    m, _, _ = finished
    m.export(tmp_path / "network.zip")
    with zipfile.ZipFile(tmp_path / "network.zip") as z:
        for name in z.namelist():
            if name.endswith("rl.jsonl"):
                assert "network_workspace" not in z.read(name).decode()


def test_no_route_and_tiny_search_budget_are_not_fake_success():
    c = default_network_configuration()
    c["max_hops"] = 0
    m = MissionWorkspace.network(c)
    assert m.state()["network"]["total"] == 1 and m.state()["network"]["planned"] == 0
    assert m.run()["network"]["fulfilled"] == 0
    c["max_search_nodes"] = 1
    other = MissionWorkspace.network(c)
    assert other.state()["network"]["budget_exhausted"]


def test_restore_role_isolation_and_stale_plan(tmp_path):
    path = tmp_path / "network.sqlite3"
    m = MissionWorkspace.network(store=OperatorStore(path))
    m.run(decisions=2)
    op = OperatorStore(path).get(m.connection["session_id"])
    assert op.suggest(m.connection["tokens"]["operator"])["expected_revision"] == 2
    for role in ("agent", "role:fleet-00"):
        assert "network" not in op.state(m.connection["tokens"][role])
    token = m.connection["tokens"]["operator"]
    op.step(
        token,
        {
            "action_ids": ["wait"],
            "expected_revision": 2,
            "control_epoch": 0,
            "idempotency_key": "manual",
        },
    )
    # At this epoch wait is already scheduled; choose a different duration.
    op.step(
        token,
        {
            "action_ids": ["wait"],
            "advance_sec": 1.0,
            "expected_revision": 3,
            "control_epoch": 0,
            "idempotency_key": "different",
        },
    )
    with pytest.raises(OperatorError) as e:
        op.suggest(token)
    assert e.value.code == "plan_diverged"


def test_api_create_preset_and_agent_suggest_is_forbidden(monkeypatch):
    store = OperatorStore()
    monkeypatch.setattr(operator, "STORE", store)
    status, body, _ = handle_get("/api/operator/network/preset", {})
    assert status == 200
    status, raw, _ = handle_post("/api/operator/network", body)
    assert status == 200
    result = json.loads(raw)
    status, _, _ = handle_get(
        f"/api/operator/sessions/{result['session_id']}/suggest",
        {},
        token=result["tokens"]["agent"],
    )
    assert status == 403


def test_residual_search_rejects_missing_states_and_uses_current_memory():
    v = view()
    v.update(
        at_sec=4.0,
        recovery={
            "survey": {
                "phase": "final",
                "owner": "b",
                "remaining_bits": 50.0,
                "remaining_cpu_sec": 0.0,
                "intact": True,
                "reason": None,
            }
        },
        stored_bits={"a": 0.0, "b": 50.0, "c": 0.0},
    )
    plan = schedule_network(seal(v))
    assert [s["kind"] for s in plan["allocations"][0]["stages"]] == [
        "relay",
        "downlink",
    ]
    assert min(s["cell"] for s in plan["allocations"][0]["stages"]) >= 4
    v["actors"]["c"]["memory_bits"] = 50.0
    v["stored_bits"]["c"] = 1.0
    assert schedule_network(seal(v))["predicted_fulfilled"] == 0
    v["recovery"]["survey"]["owner"] = "missing"
    with pytest.raises(ValueError):
        schedule_network(seal(v))
