"""Paid chain proposals, explicit forecast access and physically frozen prefixes."""

import copy
from unittest.mock import patch

import pytest

from examples.planning.chain_mission import configuration
from space_env.planning.chain import (
    collect_chain_forecast,
    profile,
    propose_chain,
    replan_chain,
    schedule_chains,
    verify_chain_proposal,
    verify_chain_replan,
)
from space_env.planning.mission_bundle import publish_plan_bundle, verify_plan_bundle
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.simulation.snapshot import digest


def fixture(mode="nominal"):
    v = configuration(mode)
    d = v["scenario"]["definition"]
    d["horizon_sec"] = 180.0
    for r in d["lifecycle"]["requests"]:
        r["deadline_sec"] = 180.0
    for o in d["operations"]["outages"]:
        o["end_sec"] = 180.0
    v["task"].update(decision_sec=2.0, max_decision_sec=2.0, max_decisions=91)
    v["goals"]["coupled"]["native_spec_sha256"] = digest(d)
    return v


def proposal(v=None):
    v = fixture() if v is None else v
    return propose_chain(
        MissionRuntime(v["scenario"], access=AccessPolicy(forecast=True)),
        v["scenario"],
        v["task"],
        goals=v["goals"],
    )


def test_future_pointing_requires_authorization_and_does_not_advance():
    v = fixture()
    r = MissionRuntime(v["scenario"])
    with pytest.raises(RuntimeContractError):
        r.forecast_pointing("a-00", "t02", 93.0)
    r.access = AccessPolicy(forecast=True)
    before = r.revision, r.time_sec
    p = r.forecast_pointing("a-00", "t02", 93.0)
    assert (r.revision, r.time_sec) == before
    assert p["at_sec"] == 93.0 and len(p["goal_quaternion_wxyz"]) == 4
    assert p != r.forecast_pointing("a-00", "t02", 0.0)
    p["goal_quaternion_wxyz"][0] = 10
    assert sum(
        x * x for x in r.forecast_pointing("a-00", "t02", 93.0)["goal_quaternion_wxyz"]
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("at", [True, -1.0, 181.0, float("nan"), float("inf")])
def test_future_pointing_refuses_invalid_coverage(at):
    v = fixture()
    r = MissionRuntime(v["scenario"], access=AccessPolicy(forecast=True))
    with pytest.raises(RuntimeContractError):
        r.forecast_pointing("a-00", "t02", at)


def test_value_only_search_never_uses_admission_rollout_or_truth():
    v = fixture()
    r = MissionRuntime(v["scenario"], access=AccessPolicy(forecast=True))
    with (
        patch.object(r, "report", side_effect=AssertionError("truth oracle")),
        patch.object(r, "advance", side_effect=AssertionError("native search")),
        patch.object(
            r._backend, "validate_command", side_effect=AssertionError("admission")
        ),
    ):
        p = propose_chain(r, v["scenario"], v["task"])
    assert p["schedule"]["predicted_fulfilled"] == 2
    assert verify_chain_proposal(p, native=True)["forecasts_recomputed"]
    assert (
        p["plan"]["origin"] == "declared"
        and p["plan"]["expected_native_sha256"] is None
    )
    assert not v["scenario"]["definition"]["operations"]["products"]
    for c in p["plan"]["task"]["actions"]:
        if (
            c["id"] in {a for s in p["plan"]["steps"] for a in s["action_ids"]}
            and c["commands"][0]["kind"] == "slew"
        ):
            assert c["commands"][0]["duration_sec"] >= 91
            assert "quaternion_wxyz" in c["commands"][0]["slew"]


@pytest.mark.parametrize(
    "kind",
    ["rigid", "rf", "preload", "quality", "missing", "shared", "quota", "memory"],
)
def test_profile_rejects_unsupported_or_ambiguous_inputs(kind):
    v = fixture()
    d = v["scenario"]["definition"]
    if kind == "rigid":
        d["spacecraft_bus"]["a-00"]["attitude"]["radio_tolerance_deg"] = 10.0
    elif kind == "rf":
        d["operations"]["radio_network"] = {}
    elif kind == "preload":
        d["operations"]["products"] = [{}]
    elif kind == "quality":
        d["lifecycle"]["requests"][0]["max_attempts"] = 2
    elif kind == "missing":
        v["task"]["actions"].pop()
    elif kind == "shared":
        d["lifecycle"]["requests"][1]["target_id"] = d["lifecycle"]["requests"][0][
            "target_id"
        ]
    elif kind == "quota":
        d["operations"]["vehicles"]["a-00"]["memory_bits"] = 1.0
    else:
        d["lifecycle"]["jobs"][0]["outputs"][0]["size_bits"] = 1000000000
    with pytest.raises(ValueError):
        profile(v["scenario"], v["task"])


def test_forecast_snapshot_and_budget_are_owned_and_strict():
    v = fixture()
    r = MissionRuntime(v["scenario"], access=AccessPolicy(forecast=True))
    view = collect_chain_forecast(r, v["scenario"], v["task"])
    before = copy.deepcopy(view)
    s = schedule_chains(view)
    s["steps"][0]["action_ids"].clear()
    assert view == before and schedule_chains(view) != s
    with pytest.raises(ValueError, match="budget"):
        schedule_chains(view, {"max_search_nodes": 1})
    view["chains"][0]["release_sec"] = 3.0
    with pytest.raises(ValueError, match="seal"):
        schedule_chains(view)


def test_outage_keeps_full_denominator_and_no_delivery_proposals():
    p = proposal(fixture("outage"))
    assert p["schedule"]["predicted_fulfilled"] == 0 and p["schedule"]["total"] == 24
    assert not p["schedule"]["allocations"]
    assert len(p["schedule"]["unserved"]) == 24


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    p = proposal()
    x = PlanExecution(p["plan"])
    x.run(decisions=50)
    record = x.checkpoint()
    x.run()
    out = tmp_path_factory.mktemp("chain") / "dataset"
    bundle = publish_plan_bundle(p["plan"], x.checkpoint(), out)
    return p, record, x.checkpoint(), out, bundle


def test_native_paid_capture_transform_delivery_and_bundle(native_case):
    p, _, record, out, bundle = native_case
    assert bundle["mission_goal_passed"], (out / "certificate.json").read_text()
    native = record["episode"]["runtime"]["backend_report"]
    life = native["lifecycle"]
    assert (
        life["metrics"]["fulfilled_requests"]
        == p["schedule"]["predicted_fulfilled"]
        == 2
    )
    assert any(a["source"] == "acquired" for a in life["artifacts"].values())
    assert any(a["source"] == "transformed_out" for a in life["artifacts"].values())
    assert native["spacecraft_bus"]["states"]["a-00"]["electrical_load_wh"] > 0
    assert verify_plan_bundle(out, native=True)["native_execution_replayed"]


def test_idle_replan_preserves_native_prefix_and_survives_restore(native_case):
    p, record, _, _, _ = native_case
    updated = replan_chain(p, record)
    assert updated["prefix_steps"] == 50 and updated["prefix_time_sec"] == 100.0
    assert updated["plan"]["steps"][:50] == p["plan"]["steps"][:50]
    assert updated["forecast"]["revision"] > 0
    assert updated["forecast"]["completed"]
    assert verify_chain_replan(updated)["prefix_native_replayed"]
    x = PlanExecution(updated["plan"])
    x.run(decisions=50)
    restored = PlanExecution.from_checkpoint(updated["plan"], x.checkpoint())
    restored.run()
    assert (
        restored.checkpoint()["episode"]["runtime"]["backend_report"]["lifecycle"][
            "metrics"
        ]["fulfilled_requests"]
        == 2
    )


def test_mid_chain_replanning_is_explicitly_refused(native_case):
    p, *_ = native_case
    x = PlanExecution(p["plan"])
    x.run(decisions=1)
    with pytest.raises(ValueError, match="unfinished touched chain"):
        replan_chain(p, x.checkpoint())


def test_resealed_wrong_pointing_or_steps_refused():
    p = proposal()
    rid = next(iter(p["pointings"]))
    p["pointings"][rid]["at_sec"] += 1
    p["sha256"] = digest({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError, match="pointing"):
        verify_chain_proposal(p)


@pytest.mark.parametrize(
    "kind",
    [
        "negative_energy",
        "duplicate",
        "stale",
        "reverse_window",
        "terminals",
        "private",
        "missing_acq",
    ],
)
def test_resealed_invalid_forecast_still_refused(kind):
    v = fixture()
    r = MissionRuntime(v["scenario"], access=AccessPolicy(forecast=True))
    view = collect_chain_forecast(r, v["scenario"], v["task"])
    if kind == "negative_energy":
        view["chains"][0]["energy_wh"] = -1.0
    elif kind == "duplicate":
        view["chains"][1]["id"] = view["chains"][0]["id"]
    elif kind == "stale":
        view["contacts"][0]["binding"]["revision"] = 1
    elif kind == "reverse_window":
        view["contacts"][0]["windows"] = [[3.0, 1.0]]
    elif kind == "terminals":
        view["stations"]["g00"] = True
    elif kind == "private":
        view["actors"]["a-00"]["rng_seed"] = 7
    else:
        view["acquisitions"].pop()
    view["sha256"] = digest({k: v for k, v in view.items() if k != "sha256"})
    with pytest.raises(ValueError):
        schedule_chains(view)


def test_future_releases_energy_and_tx_quota_block_planning():
    v = fixture()
    for r in v["scenario"]["definition"]["lifecycle"]["requests"]:
        r["release_sec"] = 170.0
    v["goals"]["coupled"]["native_spec_sha256"] = digest(v["scenario"]["definition"])
    assert proposal(v)["schedule"]["predicted_fulfilled"] == 0
    v = fixture()
    for a in v["scenario"]["definition"]["operations"]["vehicles"].values():
        a["max_tx_sec"] = 0.0
    v["goals"]["coupled"]["native_spec_sha256"] = digest(v["scenario"]["definition"])
    p = proposal(v)
    assert p["schedule"]["predicted_fulfilled"] == 0
    assert set(p["schedule"]["unserved"].values()) == {
        "conservative_energy_or_tx_quota"
    }


def test_chain_cli_roundtrip_and_no_overwrite(tmp_path, capsys):
    import json

    from space_env.cli import main
    from space_env.simulation.snapshot import canonical_json

    v = fixture()
    for key in ("scenario", "task"):
        (tmp_path / (key + ".json")).write_text(canonical_json(v[key]))
    args = [
        "chain-plan",
        str(tmp_path / "scenario.json"),
        str(tmp_path / "task.json"),
        str(tmp_path / "proposal.json"),
    ]
    assert main(args) is None
    result = json.loads(capsys.readouterr().out)
    assert (
        result["predicted_fulfilled"] == 2
        and result["native_feasibility_validated"] is False
    )
    assert (
        main(["verify-chain-plan", str(tmp_path / "proposal.json"), "--native"]) is None
    )
    assert json.loads(capsys.readouterr().out)["forecasts_recomputed"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
