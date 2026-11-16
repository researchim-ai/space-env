"""One native runtime across phases; replay and observability are not bypassed."""

import copy

import pytest

from space_env.navigation.finite_suite import compile_suite
from space_env.simulation.snapshot import digest
from space_env.tasks.campaign import CampaignSession, CampaignSpec


@pytest.fixture(scope="module")
def recipe():
    case = copy.deepcopy(compile_suite()[0])
    goal = case["task"].pop("goal")
    goal["deadline_sec"] = 4.0
    retain = copy.deepcopy(goal)
    retain["deadline_sec"] = 6.0
    retain["conditions"][0]["not_before_sec"] = 4.0
    return case, {
        "id": "deliver-and-retain",
        "phases": [{"id": "deliver", "goal": goal}, {"id": "retain", "goal": retain}],
    }


def new(recipe):
    case, spec = recipe
    return CampaignSession(case["scenario"], case["task"], spec)


def step(session, actions, dt=1.0, key=None):
    return session.step(
        {"a": actions},
        expected_revision=session.revision,
        idempotency_key=key or str(session.revision),
        advance_sec=dt,
    )


def test_live_continuity_and_replay(recipe):
    s = new(recipe)
    step(s, ["burn"])
    step(s, ["deliver"], 2.0)
    assert s.status()["completed_phases"] == ["deliver"]
    before = s._session._runtime._backend.report()
    filtered = copy.deepcopy(s._session._navigation._filters)
    cp = s.checkpoint()
    restored = CampaignSession.from_checkpoint(cp)
    assert restored.checkpoint() == cp
    result = step(s, ["wait"])
    assert result["campaign"]["success"]
    assert step(restored, ["wait"]) == result
    after = s._session._runtime._backend.report()
    assert before["products"] == after["products"]
    assert s._session._navigation._filters["a"]["index"] > filtered["a"]["index"]
    assert (
        before["energy_ledger"]["a"]["consumed_wh"]
        <= after["energy_ledger"]["a"]["consumed_wh"]
    )
    assert "campaign" not in s.observe("a")
    assert "goal" not in s.observe("a")
    assert (
        s.checkpoint() == CampaignSession.from_checkpoint(s.checkpoint()).checkpoint()
    )


def test_deadline_clamp_and_terminal(recipe):
    s = new(recipe)
    response = step(s, ["wait"], 5.0)
    assert response["transition"]["elapsed_sec"] == 4.0
    assert s.status()["terminal"] and s.status()["failed"]
    assert not s.status()["success"]
    with pytest.raises(ValueError, match="terminal"):
        step(s, ["wait"])
    assert CampaignSession.from_checkpoint(s.checkpoint()).status() == s.status()


def test_idempotency_stale_and_defensive(recipe):
    s = new(recipe)
    response = step(s, ["burn"], key="first")
    assert (
        s.step(
            {"a": ["burn"]},
            expected_revision=0,
            idempotency_key="first",
            advance_sec=1.0,
        )
        == response
    )
    response["campaign"]["completed_phases"].append("fake")
    assert not s.status()["completed_phases"]
    with pytest.raises(ValueError, match="conflict"):
        step(s, ["wait"], key="first")
    with pytest.raises(ValueError, match="stale"):
        s.step({"a": ["wait"]}, expected_revision=0, idempotency_key="stale")
    cp = s.checkpoint()
    cp["status"]["success"] = True
    cp["sha256"] = digest({k: v for k, v in cp.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay mismatch"):
        CampaignSession.from_checkpoint(cp)


def test_all_phase_guards_before_execution(recipe):
    case, spec = copy.deepcopy(recipe)
    spec["phases"][1]["goal"]["conditions"] = [
        {
            "id": "forbidden-truth",
            "kind": "momentum",
            "actor": "a",
            "tolerance_nms": 0.1,
        }
    ]
    with pytest.raises(ValueError, match="goal-error"):
        CampaignSession(case["scenario"], case["task"], spec)


def test_schema_and_existing_goal_rejected(recipe):
    case, spec = copy.deepcopy(recipe)
    spec["phases"][1]["id"] = "deliver"
    with pytest.raises(ValueError, match="unique"):
        CampaignSpec.model_validate(spec)
    spec = copy.deepcopy(recipe[1])
    spec["phases"][1]["goal"]["deadline_sec"] = 4.0
    spec["phases"][1]["goal"]["conditions"][0]["not_before_sec"] = 0.0
    with pytest.raises(ValueError, match="strictly increase"):
        CampaignSpec.model_validate(spec)
    case["task"]["goal"] = recipe[1]["phases"][0]["goal"]
    with pytest.raises(ValueError, match="omit goal"):
        CampaignSession(case["scenario"], case["task"], recipe[1])


def test_midburn_restore_carries_live_navigation(recipe):
    s = new(recipe)
    step(s, ["burn"], 0.5)
    restored = CampaignSession.from_checkpoint(s.checkpoint())
    assert step(s, ["wait"], 0.5) == step(restored, ["wait"], 0.5)
    assert step(s, ["deliver"], 2.0) == step(restored, ["deliver"], 2.0)
    assert step(s, ["wait"]) == step(restored, ["wait"])
    assert s.status()["success"]


def test_absolute_energy_limit_not_renewed_in_new_phase(recipe):
    case, spec = copy.deepcopy(recipe)
    spec["phases"][1]["goal"]["limits"] = {"a": {"max_energy_j": 1.0}}
    s = CampaignSession(case["scenario"], case["task"], spec)
    step(s, ["burn"])
    step(s, ["deliver"], 2.0)
    assert s.status()["completed_phases"] == ["deliver"]
    step(s, ["wait"])
    assert s.status()["terminal"] and s.status()["failed"]
    assert not s.status()["success"]


def test_native_truncation_is_campaign_failure(recipe):
    case, spec = copy.deepcopy(recipe)
    case["task"]["max_decisions"] = 1
    s = CampaignSession(case["scenario"], case["task"], spec)
    response = step(s, ["wait"])
    assert response["transition"]["truncated"]
    assert s.status()["failed"] and s.status()["terminal"]


def test_invalid_request_does_not_commit(recipe):
    s = new(recipe)
    before = s.checkpoint()
    with pytest.raises(ValueError, match="unknown task action"):
        step(s, ["not-an-action"])
    assert s.checkpoint() == before


def test_eo_delivery_then_reorientation():
    from examples.modeling.campaign import recipe

    case, spec = recipe("eo")
    s = CampaignSession(case["scenario"], case["task"], spec)
    delivery_ledger = None
    for item in case["campaign_steps"]:
        if s.status()["terminal"]:
            break
        response = s.step(
            {"@central": item["action_ids"]},
            expected_revision=s.revision,
            idempotency_key=str(s.revision),
            advance_sec=item["advance_sec"],
        )
        if response["completed_phase"] == "acquire-process-deliver":
            delivery_ledger = copy.deepcopy(
                s._session._runtime._backend.report()["lifecycle"]["acknowledgements"]
            )
    assert s.status()["success"]
    assert s.status()["completed_phases"] == ["acquire-process-deliver", "reorient"]
    assert (
        delivery_ledger
        == s._session._runtime._backend.report()["lifecycle"]["acknowledgements"]
    )
    assert (
        CampaignSession.from_checkpoint(s.checkpoint()).checkpoint() == s.checkpoint()
    )
