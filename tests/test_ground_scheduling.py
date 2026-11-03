"""Versioned windows, ACK-only reclamation, native RF and honest qualification."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from space_env.planning import (
    BaselinePolicy,
    PlanningInput,
    PlanningRun,
    PolicyConfig,
    make_input,
)
from space_env.scaling.benchmark import ScaleSpec, compare_semantics
from space_env.scaling.campaign import campaign_workload
from space_env.scaling.ground import ground_window_diagnostic
from space_env.scaling.qualification import qualification_domains, row_domains
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks.observation import STATES
from tests.test_planning import public


def workload(**changes):
    return campaign_workload(
        {
            "cameras": 1,
            "relays": 1,
            "waves": 1,
            "horizon_sec": 20.0,
            "repeats": 1,
            **changes,
        }
    )


def test_default_profile_and_old_frozen_manifest_unchanged():
    w = workload()
    assert w["scenario"]["metadata"]["family"] == "coupled-formation/1"
    assert "ground_control_sec" not in w["scenario"]["metadata"]["configuration"]
    assert "reclaim_acked_downlinks" not in PolicyConfig().model_dump(mode="json")
    assert {
        c["duration_sec"]
        for a in w["task"]["actions"]
        for c in a["commands"]
        if c["kind"] == "downlink"
    } == {4.0}
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/acceptance/B11_NATIVE_COPY_LARGE.manifest.json.gz"
    )
    frozen = json.loads(gzip.decompress(path.read_bytes()))
    normalized = ScaleSpec.model_validate(frozen["manifest"]).model_dump(
        mode="json", by_alias=True
    )
    assert digest(normalized) == frozen["sha256"]


def test_short_windows_change_task_setup_not_physics_or_targets():
    old, short = workload(), workload(ground_control_sec=2.0)
    assert old["scenario"]["definition"] == short["scenario"]["definition"]
    assert old["policy"] == short["policy"]
    assert short["scenario"]["metadata"]["family"] == "coupled-formation/2"
    assert short["min_completion_fraction"] == 1.0
    for action in short["task"]["actions"]:
        command = action["commands"][0]
        if command["kind"] == "downlink":
            command["duration_sec"] = 4.0
    assert short["task"] == old["task"]


@pytest.mark.parametrize(
    "changes",
    [
        {"ground_control_sec": 1.0},
        {"ground_control_sec": 9.0},
        {"ground_control_sec": True},
        {"ack_release_actions": 1},
    ],
)
def test_invalid_window_settings(changes):
    with pytest.raises(ValueError):
        workload(**changes)


def test_preflight_bound_is_conditional_and_ignores_no_failures():
    w = workload(cameras=8, relays=4, waves=5, wave_gap_sec=20.0, horizon_sec=120.0)
    before = copy.deepcopy(w)
    legacy = ground_window_diagnostic(w)
    assert w == before
    assert legacy["applicable"] and legacy["requests_upper_bound"] == 28
    assert (
        legacy["target_exceeds_conditional_bound"]
        and legacy["requires_native_validation"]
    )
    short = ground_window_diagnostic(
        workload(
            cameras=8,
            relays=4,
            waves=5,
            wave_gap_sec=20.0,
            horizon_sec=120.0,
            ground_control_sec=2.0,
        )
    )
    assert short["full_control_windows_upper_bound"] == 56
    assert not short["target_exceeds_conditional_bound"]
    assert not ground_window_diagnostic(workload(ack_release_actions=True))[
        "applicable"
    ]
    for mutate in [
        lambda v: v["scenario"]["definition"]["operations"].update(
            products=[{"id": "preload"}]
        ),
        lambda v: v["task"].update(decision_sec=1.0),
        lambda v: v["scenario"]["definition"]["operations"]["stations"][0].update(
            terminals=2
        ),
    ]:
        variant = copy.deepcopy(w)
        mutate(variant)
        assert not ground_window_diagnostic(variant)["applicable"]


def ack_public(*, status="delivered", remaining=0.0):
    actor = {
        "key": "actor:a",
        "id": "a",
        "kind": "actor",
        "index": 0,
        "values": {"soc": 0.8},
    }
    product = {
        "key": "product:old",
        "id": "old",
        "kind": "product",
        "index": 1,
        "values": {"remaining_bits": remaining, "status_code": STATES[status]},
    }
    value = public(
        [("next", "a", "downlink", "fresh", 1.0, 100.0, "radio", "G")],
        entities=[actor, product],
        active=[
            {
                "id": "old-control",
                "actor": "a",
                "kind": "downlink",
                "product_id": "old",
                "destination": "G",
                "until_sec": 8.0,
                "claims": [["a", "radio"]],
            }
        ],
    ).view()
    value["observation"]["entities"][2]["values"].update(
        remaining_bits=100.0,
        remaining_work_sec=0.0,
        ready_sec=0.0,
        status_code=STATES["resident"],
    )
    value["observation"]["relations"] = [
        {"kind": "owns", "source": 0, "destination": 1},
        {"kind": "owns", "source": 0, "destination": 2},
    ]
    value["observation"]["catalogue"][0].update(enabled=False, reason="actor_busy")
    value["observation"]["catalogue"].append(
        {"id": "release", "kind": "cancel_actor", "enabled": True, "actors": ["a"]}
    )
    value["actions"]["release"] = {"role": "a", "claims": [["a", "actor"]]}
    return value


def decide(value, **config):
    return BaselinePolicy({"reclaim_acked_downlinks": True, **config}).decide(
        PlanningInput(canonical_json(value))
    )["action_ids"]


def test_observed_ack_allows_cancel_and_conditional_replacement():
    assert set(decide(ack_public())) == {"release", "next"}
    assert BaselinePolicy().decide(PlanningInput(canonical_json(ack_public())))[
        "action_ids"
    ] == ["wait"]


@pytest.mark.parametrize(
    "status,remaining", [("await_ack", 0.0), ("resident", 10.0), ("delivered", 10.0)]
)
def test_transmitted_or_incomplete_is_not_observed_ack(status, remaining):
    assert decide(ack_public(status=status, remaining=remaining)) == ["wait"]


@pytest.mark.parametrize(
    "cause", ["ownership", "role", "other_control", "incoming_relay"]
)
def test_cannot_cancel_unowned_or_other_work(cause):
    v = ack_public()
    if cause == "ownership":
        v["observation"]["relations"] = []
    elif cause == "role":
        v["roles"]["a"]["actors"] = []
    elif cause == "other_control":
        v["active"].append(
            {
                "id": "cpu",
                "actor": "a",
                "kind": "transform",
                "destination": "",
                "claims": [["a", "cpu"]],
            }
        )
    else:
        v["active"].append(
            {
                "id": "rx",
                "actor": "b",
                "kind": "relay",
                "destination": "a",
                "claims": [["b", "radio"]],
            }
        )
    assert decide(v) == ["wait"]


def test_role_slots_and_release_cooldown_still_apply():
    value = ack_public()
    value["roles"]["a"]["action_slots"] = 1
    assert decide(value) == ["release"]
    policy = BaselinePolicy({"reclaim_acked_downlinks": True})
    policy._cooldown["release"] = 2
    assert policy.decide(PlanningInput(canonical_json(ack_public())))["action_ids"] == [
        "wait"
    ]


@pytest.fixture(scope="module")
def native_reclaim():
    w = workload(
        cameras=2,
        relays=1,
        horizon_sec=40.0,
        mode="ground_recovery",
        ground_control_sec=8.0,
        ack_release_actions=True,
    )
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    prefix, before = None, None
    while not run.done:
        if len(run.trace) == 11:
            prefix = run.checkpoint(compact=True)
            before = make_input(run.session, conditional_claims=True).view()
        run.step()
    return run, prefix, before


def test_native_ack_reclaim_and_cold_prefix_parity(native_reclaim):
    run, prefix, before = native_reclaim
    assert run.metrics()["fulfilled"] == 2
    assert all(
        row["transition"]["feedback"]["accepted"] for row in run.session._transitions
    )
    from space_env.factory.quality import quality_report

    quality = quality_report(run)
    assert quality["passed"] and quality["conditional_after_cancel"] > 0
    assert any(
        "release-acked-relay-00" in t["decision"]["action_ids"]
        and any("-ground-" in a for a in t["decision"]["action_ids"])
        for t in run.trace
    )
    assert any(
        e["values"].get("status_code") == STATES["delivered"]
        for e in before["observation"]["entities"]
        if e["kind"] == "product"
    )
    restored = PlanningRun.from_checkpoint(prefix)
    restored.run()
    assert restored.trace == run.trace
    assert (
        restored.session._runtime._backend.report()
        == run.session._runtime._backend.report()
    )
    checkpoint = run.checkpoint(compact=True)["episode"]["runtime"]
    commands = {
        row["command"]["command_id"]: row["command"]
        for row in checkpoint["report"]["commands"]
    }
    acks = {
        a["product_id"]: a["acknowledged_sec"]
        for a in checkpoint["backend_report"]["lifecycle"]["acknowledgements"]
    }
    cancelled = [
        op for op in checkpoint["operations"] if op["request"]["op"] == "cancel"
    ]
    assert cancelled
    for op in cancelled:
        command = commands[op["request"]["arguments"]["command_id"]]
        assert command["kind"] == "downlink"
        assert acks[command["product_id"]] <= op["response"]["time_sec"] + 1e-9


def test_conditional_bridge_exposes_only_current_accepted_product_references(
    native_reclaim,
):
    _run, prefix, _ = native_reclaim
    restored = PlanningRun.from_checkpoint(prefix)
    legacy = make_input(restored.session).view()
    conditional = make_input(restored.session, conditional_claims=True).view()
    assert all("product_id" not in c for c in legacy["active"])
    assert all("product_id" in c for c in conditional["active"])
    assert conditional["observation"] == legacy["observation"]
    assert not any(
        k in canonical_json(conditional)
        for k in ['"scenario"', '"episode_seed"', '"ack_loss_probability"']
    )
    assert any(
        a["claims"]
        for id, a in conditional["actions"].items()
        if not next(
            c for c in conditional["observation"]["catalogue"] if c["id"] == id
        )["enabled"]
    )


@pytest.mark.parametrize(
    "condition",
    ["nominal", "delayed_ack", "slow_ground", "slow_setup", "offset_station"],
)
def test_short_windows_native_ack_and_rf_constraints(condition):
    w = workload(ground_control_sec=2.0, ack_release_actions=True)
    receiver = w["scenario"]["definition"]["operations"]["radio_network"]["receivers"][
        "Ground"
    ]
    if condition == "delayed_ack":
        receiver["ack_processing_sec"] = 6.0
    elif condition == "slow_ground":
        receiver["aggregate_rate_bps"] = 10000.0
    elif condition == "slow_setup":
        receiver["acquisition_sec"] = 3.0
    elif condition == "offset_station":
        w["scenario"]["definition"]["operations"]["stations"][0]["lon_deg"] += 4.0
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    while not run.done:
        before = make_input(run.session, conditional_claims=True).view()
        transition = run.step()
        released = any(
            id.startswith("release-acked-")
            for id in run.trace[-1]["decision"]["action_ids"]
        )
        if released:
            assert any(
                e["values"].get("status_code") == STATES["delivered"]
                for e in before["observation"]["entities"]
                if e["kind"] == "product"
            )
        assert transition["feedback"]["accepted"]
    assert run.metrics()["fulfilled"] == (0 if condition == "slow_ground" else 1)
    report = run.session._runtime._backend.report()
    assert all(
        abs(x["balance_error_wh"]) < 1e-6 for x in report["energy_ledger"].values()
    )
    if condition == "delayed_ack":
        assert run.metrics()["completion_latency_sec"][0] >= 6.0


def test_qualification_domains_never_turn_failed_mission_into_pass():
    from space_env.scaling.qualification import INTEGRITY, PERFORMANCE

    w = workload()
    a, b = copy.deepcopy(w), copy.deepcopy(w)
    a.update(id="straight", comparison_group="same")
    b.update(id="resumed", comparison_group="same", resume_after_decisions=2)
    spec = ScaleSpec.model_validate({"id": "domains", "workloads": [a, b]})
    rows = [
        {
            "id": id,
            "passed": False,
            "error": None,
            "gates": {
                **{k: True for k in PERFORMANCE | INTEGRITY},
                "completion": False,
            },
            "native_backend_sha256": "native",
            "trace_sha256": "trace",
        }
        for id in ["straight", "resumed"]
    ]
    formal = compare_semantics(spec, rows)
    result = qualification_domains(spec, rows, formal)
    assert result["domains"]["technical_runtime"]["all_expected_passed"]
    assert not result["domains"]["mission_expectations"]["all_expected_passed"]
    assert result["observed_raw_repeatability"]["same"]["observed_signatures_identical"]
    assert not result["observed_raw_repeatability"]["same"]["formal_group_passed"]
    assert set(row_domains({"error": {"type": "TimeoutExpired"}}).values()) == {
        "not_observed"
    }


def test_invalid_replacement_rolls_back_ack_cancellation(native_reclaim):
    _, prefix, _ = native_reclaim
    restored = PlanningRun.from_checkpoint(prefix)
    session = restored.session
    old = next(
        c
        for c in session._runtime.command_status()
        if c["status"] == "running" and c["command"]["kind"] == "downlink"
    )
    old_id = old["command"]["command_id"]
    action = next(
        a
        for a in session._actions
        if a.commands
        and a.commands[0].kind == "downlink"
        and a.commands[0].product_id == old["command"]["product_id"]
    )
    response = session.step(
        {"@central": ["release-acked-relay-00", action.id]},
        expected_revision=session.revision,
        idempotency_key="reject-after-ack",
    )
    assert not response["feedback"]["accepted"]
    assert response["feedback"]["code"] == "product_not_local"
    assert (
        next(
            c
            for c in session._runtime.command_status()
            if c["command"]["command_id"] == old_id
        )["status"]
        == "running"
    )
    assert not any(
        row["request"]["op"] == "cancel" for row in session._runtime._operations
    )


def test_future_outages_and_loss_seed_do_not_enter_conditional_input():
    from space_env.tasks import TaskSession

    w = workload(waves=2, horizon_sec=40.0, ack_release_actions=True)
    other = copy.deepcopy(w["scenario"])
    ops = other["definition"]["operations"]
    ops["dtn_network"]["seed"] += 1
    ops["outages"] = [{"station": "Ground", "start_sec": 20.0, "end_sec": 30.0}]
    a = make_input(TaskSession(w["scenario"], w["task"]), conditional_claims=True)
    b = make_input(TaskSession(other, w["task"]), conditional_claims=True)
    assert a == b
    assert BaselinePolicy(w["policy"]).decide(a) == BaselinePolicy(w["policy"]).decide(
        b
    )
    assert not any(
        e["id"].startswith("w01-") for e in a.view()["observation"]["entities"]
    )


def test_delayed_ack_never_reclaims_an_active_window_early():
    w = workload(
        cameras=2,
        relays=1,
        horizon_sec=40.0,
        mode="ground_recovery",
        ground_control_sec=8.0,
        ack_release_actions=True,
    )
    w["scenario"]["definition"]["operations"]["radio_network"]["receivers"]["Ground"][
        "ack_processing_sec"
    ] = 4.0
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    decisions = []
    while not run.done:
        now = run.session._runtime.time_sec
        run.step()
        if "release-acked-relay-00" in run.trace[-1]["decision"]["action_ids"]:
            decisions.append(now)
    assert run.metrics()["fulfilled"] == 2
    assert decisions and min(decisions) >= 26.0
    assert all(
        row["transition"]["feedback"]["accepted"] for row in run.session._transitions
    )
