"""Assignment search respects pins, budgets, provenance and actual execution."""

import copy

import pytest

from space_env.api.operator import OperatorStore
from space_env.api.workspace import MissionConfiguration
from space_env.planning.assignment import assign_configuration
from space_env.workspace import (
    MissionWorkspace,
    compile_configuration,
    default_configuration,
)


def configuration():
    c = default_configuration()
    c["horizon_sec"] = 120.0
    c["requests"] = [c["requests"][1]]
    c["requests"][0].update(actor_id=None, station=None, deadline_sec=120.0)
    c["max_assignment_trials"] = 4
    return c


def test_auto_requires_planning_instead_of_silent_compilation():
    with pytest.raises(ValueError, match="Автоматическое"):
        compile_configuration(configuration())


@pytest.fixture(scope="module")
def assigned():
    c = configuration()
    original = copy.deepcopy(c)
    mission = MissionWorkspace(c)
    initial = mission.state()["workspace"]
    state = mission.run()
    assert c == original
    yield mission, initial, state
    mission.close()


def test_joint_actor_and_station_selection_with_native_delivery(assigned):
    _, initial, state = assigned
    assert initial["configuration"]["requests"][0]["actor_id"] is None
    assert initial["configuration"]["requests"][0]["station"] is None
    chosen = initial["resolved_configuration"]["requests"][0]
    assert chosen["actor_id"] == "b-00" and chosen["station"] == "East"
    assert initial["planned"] == 1 and initial["fulfilled"] == 0
    assert state["workspace"]["fulfilled"] == state["workspace"]["total"] == 1
    assert state["workspace"]["plan_aligned"]
    audit = initial["assignment"]
    assert audit["automatic"] and len(audit["trials"]) <= 4
    assert audit["search_nodes"] <= audit["max_search_nodes"]
    assert audit["search_nodes"] == sum(t["charged_nodes"] for t in audit["trials"])


def test_explicit_assignments_are_never_changed():
    c = configuration()
    c["requests"][0]["actor_id"] = "a-00"  # wrong orbit, deliberately pinned
    cfg, *_, audit = assign_configuration(MissionConfiguration.model_validate(c))
    assert cfg.requests[0].actor_id == "a-00"
    assert all(t["assignments"][0][0] == "a-00" for t in audit["trials"])
    assert len(audit["trials"]) <= 2


def test_trial_limit_is_hard_and_deterministic():
    c = configuration()
    c["max_assignment_trials"] = 1
    cfg = MissionConfiguration.model_validate(c)
    first = assign_configuration(cfg)
    second = assign_configuration(cfg)
    assert len(first[-1]["trials"]) == 1
    assert first[-1] == second[-1]
    assert first[-2]["sha256"] == second[-2]["sha256"]
    assert first[0].requests[0].actor_id == "a-00"


def test_failed_search_respects_total_budget_and_creates_no_session():
    c = configuration()
    c["planner"]["max_search_nodes"] = 1
    with pytest.raises(ValueError, match="бюджете"):
        assign_configuration(MissionConfiguration.model_validate(c))


def test_assignment_information_not_in_agent_observation(assigned):
    mission, _, _ = assigned
    state = mission.operator.state(mission.connection["tokens"]["agent"])
    assert "workspace" not in state
    assert "assignment" not in state["observation"]


def test_auto_assignment_survives_durable_reopen(tmp_path):
    path = tmp_path / "auto.sqlite3"
    mission = MissionWorkspace(configuration(), store=OperatorStore(path))
    before = mission.run(decisions=2)["workspace"]
    reopened = OperatorStore(path).get(mission.connection["session_id"])
    token = mission.connection["tokens"]["operator"]
    after = reopened.state(token)["workspace"]
    assert after["assignment"] == before["assignment"]
    assert after["resolved_configuration"] == before["resolved_configuration"]
    assert after["configuration"]["requests"][0]["actor_id"] is None
    assert reopened.suggest(token)["expected_revision"] == 2


def test_infeasible_auto_requests_remain_in_denominator():
    c = configuration()
    c["outages"] = [
        {"station": s["name"], "start_sec": 0.0, "end_sec": 120.0}
        for s in c["stations"]
    ]
    mission = MissionWorkspace(c)
    try:
        result = mission.state()["workspace"]
        assert result["total"] == 1 and result["planned"] == 0
        assert result["fulfilled"] == 0
        assert result["requests"][0]["plan_reason"]
    finally:
        mission.close()


@pytest.mark.parametrize("limit", [0, 33, True])
def test_trial_limit_validation(limit):
    c = configuration()
    c["max_assignment_trials"] = limit
    with pytest.raises(ValueError):
        MissionConfiguration.model_validate(c)
