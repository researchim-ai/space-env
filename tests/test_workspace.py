"""Configuration → joint fleet plan → native operator → user result/export."""

import copy
import json
import zipfile

import pytest

from space_env.api import operator
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_get, handle_post
from space_env.api.workspace import create_mission
from space_env.planning.chain import collect_chain_forecast, schedule_chains
from space_env.planning.fleet_scheduler import schedule_fleet
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import digest
from space_env.workspace import (
    MissionWorkspace,
    compile_configuration,
    default_configuration,
)


@pytest.fixture(scope="module")
def forecast():
    _, scenario, task = compile_configuration(default_configuration())
    return collect_chain_forecast(
        MissionRuntime(scenario, access=AccessPolicy(forecast=True)), scenario, task
    )


def packing_view(forecast):
    v = copy.deepcopy(forecast)
    v.update(horizon_sec=60.0, decision_sec=2.0, completed=[])
    c = copy.deepcopy(v["chains"][0])
    a = c["actor"]
    stations = list(v["stations"])
    first, second = copy.deepcopy(c), copy.deepcopy(c)
    for row, name, station, deadline, priority in [
        (first, "first", stations[0], 50.0, 10),
        (second, "second", stations[1], 60.0, 5),
    ]:
        row.update(
            id=name,
            station=station,
            deadline_sec=deadline,
            priority=priority,
            durations_sec=[2.0, 2.0, 2.0, 2.0],
            acquisition_sec=2.0,
        )
        row["actions"] = {
            k: k + "/" + name for k in ("slew", "imaging", "transform", "downlink")
        }
    v["chains"] = [first, second]
    for actor in v["actors"].values():
        actor["base_wh"] = 0.0
    acq = copy.deepcopy(v["acquisitions"][0])
    v["acquisitions"] = []
    for name, windows in [("first", [[0.0, 58.0]]), ("second", [[0.0, 12.0]])]:
        row = copy.deepcopy(acq)
        row.update(request=name, windows=windows)
        v["acquisitions"].append(row)
    for row in v["contacts"]:
        row["windows"] = [[0.0, 60.0]]
        if row["actor"] == a and row["station"] == stations[0]:
            row["windows"] = [[40.0, 60.0]]
    v["sha256"] = digest({k: x for k, x in v.items() if k != "sha256"})
    return v


def test_compact_reservation_frees_early_opportunity(forecast):
    v = packing_view(forecast)
    before = copy.deepcopy(v)
    legacy = schedule_chains(v, {"edge_guard_sec": 0.0})
    compact = schedule_chains(v, {"edge_guard_sec": 0.0}, placement="compact")
    assert legacy["predicted_fulfilled"] == 1
    assert compact["predicted_fulfilled"] == 2
    assert (
        next(a for a in compact["allocations"] if a["request"] == "first")["start_sec"]
        == 34.0
    )
    assert v == before


def test_fleet_search_retains_baseline_and_exact_total_budget(forecast):
    v = packing_view(forecast)
    result = schedule_fleet(
        v, {"edge_guard_sec": 0.0, "max_search_nodes": 5000, "max_orders": 4}
    )
    assert result["baseline"]["predicted_fulfilled"] == 1
    assert result["schedule"]["predicted_fulfilled"] == 2
    assert result["selected_score"][0] == 15
    assert result["search_nodes"] <= 5000
    assert result["search_nodes"] == sum(
        t.get("search_nodes", t.get("charged_nodes", 0)) for t in result["trials"]
    )
    with pytest.raises(ValueError, match="Лимит"):
        schedule_fleet(v, {"max_search_nodes": 1})
    assert "placement" not in schedule_chains(v)


@pytest.mark.parametrize(
    "fault",
    [
        "actor",
        "station",
        "deadline",
        "duplicate_request",
        "actor_limit",
        "memory",
        "grid",
        "duplicate_orbit",
    ],
)
def test_configuration_errors_before_any_session(fault):
    c = default_configuration()
    store = OperatorStore()
    if fault == "actor":
        c["requests"][0]["actor_id"] = "missing"
    elif fault == "station":
        c["requests"][0]["station"] = "missing"
    elif fault == "deadline":
        c["requests"][0]["deadline_sec"] = 601.0
    elif fault == "duplicate_request":
        c["requests"][1]["id"] = c["requests"][0]["id"]
    elif fault == "actor_limit":
        c["planes"][0]["phases_deg"] = [0.0, 60.0, 120.0, 180.0]
    elif fault == "memory":
        c["memory_mbit"] = 1.0
    elif fault == "grid":
        c["horizon_sec"] = 3600.0
    else:
        c["planes"][1].update(inclination_deg=55.0, raan_deg=-0.0, phases_deg=[0.0])
    with pytest.raises(ValueError):
        create_mission(c, store)
    assert not store.sessions


def test_single_satellite_and_repeated_target_supported():
    c = default_configuration()
    c["planes"] = c["planes"][:1]
    c["requests"] = c["requests"][:1]
    repeat = copy.deepcopy(c["requests"][0])
    repeat["id"] = "repeat"
    c["requests"].append(repeat)
    _, scenario, task = compile_configuration(c)
    assert len(scenario["actor_order"]) == 1
    assert len(scenario["definition"]["acquisition_targets"]) == 1
    assert len(scenario["definition"]["lifecycle"]["requests"]) == 2
    assert {a["request_ref"] for a in task["actions"]} == {"repeat", "survey-west"}


@pytest.fixture(scope="module")
def finished():
    mission = MissionWorkspace()
    initial = mission.state()["workspace"]
    assert initial["planned"] == 2 and initial["fulfilled"] == 0
    assert all(not any(r["actual_stages"].values()) for r in initial["requests"])
    state = mission.run()
    return mission, state


def test_default_mission_real_capture_processing_delivery(finished):
    _, state = finished
    w = state["workspace"]
    assert state["done"] and w["all_fulfilled"] and w["fulfilled"] == w["total"] == 2
    assert all(all(r["actual_stages"].values()) for r in w["requests"])
    assert w["plan_aligned"]


def test_export_actual_history_and_no_capabilities(finished, tmp_path):
    mission, _ = finished
    path = tmp_path / "training.zip"
    mission.export(path)
    with zipfile.ZipFile(path) as z:
        assert "role-000/rl.jsonl" in z.namelist()
        text = z.read("role-000/rl.jsonl").decode()
        assert (
            "tokens" not in text and mission.connection["tokens"]["admin"] not in text
        )
        rows = [json.loads(x) for x in text.splitlines()]
        assert len(rows) == 120
        assert rows[-1]["next_observation"]["time_sec"] == 600.0


def test_manual_deviation_refuses_stale_plan_without_advancing():
    m = MissionWorkspace()
    token = m.connection["tokens"]["operator"]
    m.operator.step(
        token,
        {
            "action_ids": ["wait"],
            "expected_revision": 0,
            "idempotency_key": "manual",
            "advance_sec": 5.0,
        },
    )
    with pytest.raises(OperatorError, match="Ручные"):
        m.operator.suggest(token)
    state = m.state()
    assert (
        state["revision"] == 1
        and state["time_sec"] == 5.0
        and not state["workspace"]["plan_aligned"]
    )


def test_plan_survives_durable_reopen(tmp_path):
    path = tmp_path / "operator.sqlite3"
    store = OperatorStore(path)
    m = MissionWorkspace(store=store)
    m.run(decisions=3)
    reopened = OperatorStore(path).get(m.connection["session_id"])
    token = m.connection["tokens"]["operator"]
    assert reopened.state(token)["workspace"]["plan_aligned"]
    assert reopened.suggest(token)["expected_revision"] == 3
    assert (
        reopened.state(token)["workspace"]["configuration"] == default_configuration()
    )


def test_workspace_plan_not_leaked_to_role_or_agent():
    m = MissionWorkspace()
    for role in ("role:a-00", "agent"):
        state = m.operator.state(m.connection["tokens"][role])
        assert "workspace" not in state
        assert "workspace" not in state["observation"]


def test_outage_explained_without_disappearing_request():
    c = default_configuration()
    c["outages"] = [{"station": "West", "start_sec": 0.0, "end_sec": 600.0}]
    m = MissionWorkspace(c)
    w = m.state()["workspace"]
    assert w["total"] == 2 and w["planned"] == 1
    west = next(r for r in w["requests"] if r["station"] == "West")
    assert "Нет окна связи" in west["plan_reason"]


def test_public_api_and_unified_landing(monkeypatch):
    store = OperatorStore()
    monkeypatch.setattr(operator, "STORE", store)
    status, body, _ = handle_get("/", {})
    assert status == 200 and b"mission-form" in body
    status, body, _ = handle_get("/api/operator/workspace/presets", {})
    assert status == 200
    cfg = json.loads(body)["configuration"]
    cfg["requests"][0]["actor_id"] = "missing"
    status, body, _ = handle_post("/api/operator/workspace", json.dumps(cfg).encode())
    assert status == 400
    assert "actor_id" in json.loads(body)["error"]["message"]
    status, body, _ = handle_post(
        "/api/operator/workspace", json.dumps(default_configuration()).encode()
    )
    assert status == 200
    created = json.loads(body)
    assert created["state"]["workspace"]["planned"] == 2
    endpoint = "/api/operator/sessions/" + created["session_id"]
    status, body, _ = handle_get(
        endpoint + "/suggest", {}, token=created["tokens"]["operator"]
    )
    assert status == 200 and json.loads(body)["advance_sec"] == 5.0
    status, _, _ = handle_get(endpoint, {})
    assert status == 401
