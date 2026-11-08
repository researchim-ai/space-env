"""Causal baselines, bounded search, native execution and teacher evidence."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from space_env.planning import (
    BaselinePolicy,
    PlanningInput,
    PlanningRun,
    PolicyConfig,
    make_input,
)
from space_env.planning.benchmark import BenchmarkSpec, run_benchmark, verify_evaluation
from space_env.planning.cli import run_command
from space_env.planning.oracle import tiny_oracle
from space_env.runtime import RuntimeContractError, reference_scenario
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSession, export_dataset
from space_env.tasks.observation import STATES
from tests.test_custody_network import spec as custody_fixture
from tests.test_runtime_tasks import task as custody_task


@pytest.fixture
def scenario():
    data = copy.deepcopy(custody_fixture.__wrapped__())
    data["horizon_sec"] = 8.0
    data["products"] = [p for p in data["products"] if p["actor"] == "a"]
    for p in data["products"]:
        p["deadline_sec"] = 8.0
    return reference_scenario(data, id="planner-test", lineage="planner-family")


def task():
    value = custody_task()
    value["decision_sec"] = 1.0
    return value


def public(
    actions, *, entities=(), stations=None, revision=0, active=(), feedback=None
):
    catalogue, meta, roles = [], {}, {}
    for id, actor, kind, product, priority, deadline, resource, destination in actions:
        roles[actor] = {"action_slots": 2, "actors": [actor], "stations": []}
        c = {
            "actor_id": actor,
            "kind": kind,
            "duration_sec": 1.0,
            "product_id": product,
            "destination_id": destination,
        }
        catalogue.append({"id": id, "kind": "submit", "enabled": True, "commands": [c]})
        meta[id] = {"role": actor, "claims": [[actor, resource]]}
    nodes = list(copy.deepcopy(entities))
    for _, _, _, p, priority, deadline, _, _ in actions:
        if not any(e["key"] == "product:" + p for e in nodes):
            nodes.append(
                {
                    "key": "product:" + p,
                    "id": p,
                    "kind": "product",
                    "index": len(nodes),
                    "values": {
                        "priority": priority,
                        "deadline_sec": deadline,
                        "size_bits": 100.0,
                    },
                }
            )
    obs = {
        "revision": revision,
        "time_sec": float(revision),
        "catalogue": catalogue,
        "entities": nodes,
        "relations": [],
    }
    return PlanningInput(
        canonical_json(
            {
                "schema": "space-env.planning-input/1",
                "observation": obs,
                "roles": roles,
                "actions": meta,
                "active": list(active),
                "stations": stations or {"G": {"terminals": 1}},
                "limits": {"joint": 8, "decision_sec": 1.0, "max_decision_sec": 5.0},
                "feedback": feedback,
            }
        )
    )


def choice_input(revision=0, feedback=None):
    return public(
        [
            ("urgent", "a", "downlink", "p", 1.0, 2.0, "radio", "G"),
            ("valuable", "b", "downlink", "q", 5.0, 20.0, "radio", "G"),
            ("compute", "a", "processing", "r", 0.5, 30.0, "cpu", ""),
        ],
        revision=revision,
        feedback=feedback,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"seed": True},
        {"max_nodes": 0},
        {"max_nodes": True},
        {"horizon_sec": float("nan")},
        {"name": "oracle"},
        {"decision_sec": 0.0},
        {"private_seed": 9},
    ],
)
def test_invalid_policy_config(config):
    with pytest.raises(ValueError):
        PolicyConfig.model_validate(config)


def test_public_input_is_immutable_and_has_no_trainer_reference(scenario):
    s = TaskSession(scenario, task())
    p = make_input(s)
    with pytest.raises(FrozenInstanceError):
        p.payload = "{}"
    view = p.view()
    view["observation"]["time_sec"] = 900.0
    assert p.view()["observation"]["time_sec"] == 0.0
    assert all(
        k not in p.payload
        for k in ['"episode_seed"', '"scenario"', '"rng"', '"loss_probability"']
    )


@pytest.mark.parametrize(
    "name",
    ["wait", "random_masked", "priority", "earliest_deadline", "route_aware_rolling"],
)
def test_identical_public_inputs_ignore_hidden_future_and_seed(scenario, name):
    other = copy.deepcopy(scenario)
    other["definition"]["dtn_network"]["seed"] = 81
    other["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 4.0, "end_sec": 6.0}
    ]
    left, right = (
        make_input(TaskSession(scenario, task())),
        make_input(TaskSession(other, task())),
    )
    assert left == right
    a, b = BaselinePolicy({"name": name}), BaselinePolicy({"name": name})
    assert a.decide(left) == b.decide(right)


def test_priority_edf_and_wait_use_distinct_public_rules():
    p = choice_input()
    priority = BaselinePolicy({"name": "priority"}).decide(p)
    edf = BaselinePolicy({"name": "earliest_deadline"}).decide(p)
    assert (
        "valuable" in priority["action_ids"] and "urgent" not in priority["action_ids"]
    )
    assert "urgent" in edf["action_ids"] and "valuable" not in edf["action_ids"]
    assert BaselinePolicy({"name": "wait"}).decide(p)["action_ids"] == ["wait"]


def test_search_finds_better_joint_set_than_greedy_with_independent_subset_oracle():
    # radio+cpu bundle conflicts with each single; two singles have higher utility.
    v = public(
        [
            ("bundle", "a", "processing", "b", 2.0, 10.0, "radio", ""),
            ("cpu", "a", "processing", "c", 1.1, 10.0, "cpu", ""),
            ("radio", "a", "processing", "r", 1.1, 10.0, "radio", ""),
        ]
    ).view()
    v["actions"]["bundle"]["claims"].append(["a", "cpu"])
    result = BaselinePolicy().decide(PlanningInput(canonical_json(v)))
    assert result["action_ids"] == ["cpu", "radio"]
    feasible = []
    ids = sorted(v["actions"])
    for n in range(3):
        for subset in itertools.combinations(ids, n):
            claims = [tuple(c) for id in subset for c in v["actions"][id]["claims"]]
            if len(claims) == len(set(claims)):
                weights = {"bundle": 2.0, "cpu": 1.1, "radio": 1.1}
                feasible.append((sum(weights[id] for id in subset), list(subset)))
    assert result["action_ids"] == max(feasible)[1]


def test_search_budget_has_deterministic_greedy_fallback_and_candidate_limit():
    config = {"max_nodes": 1, "max_candidates": 2}
    r = BaselinePolicy(config).decide(choice_input())
    assert r["search_nodes"] == 1 and r["search_budget_exhausted"]
    assert r["omitted_candidates"] == 1 and len(set(r["action_ids"])) == len(
        r["action_ids"]
    )


def test_role_slots_product_exclusion_and_terminal_capacity():
    p = public(
        [
            ("one", "a", "downlink", "p", 2.0, 10.0, "radio", "G"),
            ("two", "b", "downlink", "p", 1.0, 10.0, "radio", "G"),
        ]
    )
    result = BaselinePolicy().decide(p)
    assert len(result["action_ids"]) == 1


def test_unknown_terminal_capacity_is_not_invented():
    v = choice_input().view()
    v["stations"] = {}
    r = BaselinePolicy().decide(PlanningInput(canonical_json(v)))
    assert r["action_ids"] == ["compute"]


def test_receiver_memory_and_half_duplex_claims():
    receiver = {
        "key": "actor:b",
        "id": "b",
        "kind": "actor",
        "index": 0,
        "values": {"capacity_bits": 10.0, "stored_bits": 0.0, "reserved_bits": 0.0},
    }
    p = public(
        [("hop", "a", "relay", "p", 1.0, 10.0, "radio", "b")], entities=[receiver]
    )
    assert BaselinePolicy().decide(p)["action_ids"] == ["wait"]
    v = p.view()
    v["observation"]["entities"][0]["values"]["capacity_bits"] = 200.0
    v["active"] = [
        {
            "id": "busy",
            "actor": "b",
            "kind": "processing",
            "destination": "",
            "claims": [["b", "cpu"]],
        }
    ]
    assert BaselinePolicy().decide(PlanningInput(canonical_json(v)))["action_ids"] == [
        "wait"
    ]


def test_rng_resume_and_same_revision_retry_do_not_draw_again():
    p = BaselinePolicy({"name": "random_masked", "seed": 44})
    first = p.decide(choice_input())
    saved = p.checkpoint()
    assert p.decide(choice_input()) == first and p.checkpoint() == saved
    restored = BaselinePolicy.from_checkpoint(saved)
    assert p.decide(choice_input(1)) == restored.decide(choice_input(1))
    assert p.checkpoint() == restored.checkpoint()


def test_policy_rejection_cooldown_and_revision_guard():
    p = BaselinePolicy({"name": "priority"})
    first = p.decide(choice_input())["action_ids"]
    second = p.decide(choice_input(1, {"accepted": False, "code": "runtime_rejected"}))[
        "action_ids"
    ]
    assert not set(first) & set(second)
    saved = p.checkpoint()
    with pytest.raises(ValueError, match="consecutive"):
        p.decide(choice_input(9))
    assert p.checkpoint() == saved


def test_observable_stall_triggers_cancel_not_free_retry():
    v = public([]).view()
    v["roles"] = {"a": {"actors": ["a"], "action_slots": 1}}
    v["actions"] = {"cancel": {"role": "a", "claims": [["a", "actor"]]}}
    v["observation"]["catalogue"] = [
        {"id": "cancel", "kind": "cancel_actor", "enabled": True, "actors": ["a"]}
    ]
    v["observation"]["entities"] = [
        {"key": "actor:a", "id": "a", "kind": "actor", "index": 0, "values": {}},
        {
            "key": "transfer:t",
            "id": "t",
            "kind": "transfer",
            "index": 1,
            "values": {"progress": 0.0, "status_code": STATES["capacity_blocked"]},
        },
    ]
    v["observation"]["relations"] = [{"kind": "source", "source": 0, "destination": 1}]
    policy = BaselinePolicy({"stall_steps": 1})
    assert policy.decide(PlanningInput(canonical_json(v)))["action_ids"] == ["wait"]
    v["observation"]["revision"] = 1
    v["observation"]["time_sec"] = 1.0
    assert policy.decide(PlanningInput(canonical_json(v)))["action_ids"] == ["cancel"]


def test_failed_transfer_transition_cools_route_once_and_survives_restart():
    v = public([]).view()
    v["observation"]["entities"] = [
        {"key": "actor:a", "id": "a", "kind": "actor", "index": 0, "values": {}},
        {"key": "actor:b", "id": "b", "kind": "actor", "index": 1, "values": {}},
        {
            "key": "transfer:t",
            "id": "t",
            "kind": "transfer",
            "index": 2,
            "values": {
                "progress": 0.0,
                "remaining_bits": 100.0,
                "status_code": STATES["ready"],
            },
        },
    ]
    v["observation"]["relations"] = [
        {"kind": "source", "source": 0, "destination": 2},
        {"kind": "destination", "source": 2, "destination": 1},
    ]
    p = BaselinePolicy({"cooldown_steps": 2})
    p.decide(PlanningInput(canonical_json(v)))
    v["observation"]["revision"] = 1
    v["observation"]["time_sec"] = 1.0
    v["observation"]["entities"][-1]["values"]["status_code"] = STATES["cancelled"]
    p.decide(PlanningInput(canonical_json(v)))
    route = digest(["a", "b"])
    assert p.checkpoint()["route_cooldown"][route] == 3
    restored = BaselinePolicy.from_checkpoint(p.checkpoint())
    for revision in (2, 3):
        v["observation"]["revision"] = revision
        v["observation"]["time_sec"] = float(revision)
        assert p.decide(PlanningInput(canonical_json(v))) == restored.decide(
            PlanningInput(canonical_json(v))
        )
    assert route not in p.checkpoint()["route_cooldown"]


@pytest.mark.parametrize("name", ["priority", "random_masked", "route_aware_rolling"])
def test_runner_cold_resume_preserves_policy_and_runtime(scenario, name):
    r = PlanningRun(scenario, task(), {"name": name, "seed": 17})
    r.run(2)
    record = r.checkpoint()
    restored = PlanningRun.from_checkpoint(record)
    assert restored.checkpoint() == record
    r.run(3)
    restored.run(3)
    assert r.trace == restored.trace
    assert r.session.checkpoint() == restored.session.checkpoint()
    assert r.policy.checkpoint() == restored.policy.checkpoint()


def test_runner_physics_failure_rolls_back_policy_rng_and_task(scenario, monkeypatch):
    r = PlanningRun(scenario, task(), {"name": "random_masked"})
    before = r.checkpoint()

    def fail(*a, **kw):
        raise RuntimeError("physics failed")

    monkeypatch.setattr(r.session, "step", fail)
    with pytest.raises(RuntimeError, match="physics failed"):
        r.step()
    assert r.checkpoint() == before


def test_runner_rejection_is_real_transition_with_no_free_admission_search(
    scenario, monkeypatch
):
    r = PlanningRun(scenario, task(), {"name": "priority"})
    original = r.session._runtime._backend.validate_command

    def reject(c):
        raise RuntimeContractError("invalid_request", "hidden validation reason")

    monkeypatch.setattr(r.session._runtime._backend, "validate_command", reject)
    r.step()
    assert r.trace[0]["feedback"]["code"] == "runtime_rejected"
    assert r.session.observe()["time_sec"] == 1.0 and len(r.trace) == 1
    assert not r.session._runtime.report()["commands"]
    # Instance monkeypatches are a test instrument, not serializable episode state.
    monkeypatch.setattr(r.session._runtime._backend, "validate_command", original)


def test_runner_tampered_trace_cannot_be_rehashed_into_valid_replay(scenario):
    r = PlanningRun(scenario, task())
    r.run(1)
    record = r.checkpoint()
    record["trace"][0]["decision"]["action_ids"] = ["wait"]
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay mismatch"):
        PlanningRun.from_checkpoint(record)


def test_metrics_ack_only_and_sft_uses_existing_verified_export(scenario, tmp_path):
    r = PlanningRun(scenario, task(), {"name": "priority"})
    r.step()
    assert r.metrics()["acked_ground_mbits"] == 0.0
    r.run()
    m = r.metrics()
    assert (
        m["acked_ground_mbits"] == pytest.approx(2.0)
        and m["fulfilled"] == m["total"] == 1
    )
    assert m["terminated"] and not m["truncated"] and m["energy_j"] > 0.0
    result = export_dataset(
        r.session.checkpoint(), tmp_path / "sft.jsonl", format="sft"
    )
    assert result["rows"] > 0
    assert (
        json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])[
            "policy_origin"
        ]
        == "scripted"
    )


def manifest(scenario):
    return {
        "id": "tiny",
        "cases": [
            {
                "id": "one",
                "family": "ground",
                "scenario": scenario,
                "task": task(),
                "episode_seeds": [0],
                "split": "holdout",
            }
        ],
        "policies": [
            {"name": n}
            for n in ["wait", "priority", "earliest_deadline", "route_aware_rolling"]
        ],
        "teacher_gates": {"min_cases_per_family": 1},
    }


def test_benchmark_freezes_manifest_replays_and_retains_blocked_cases(
    scenario, tmp_path
):
    data = manifest(scenario)
    result = run_benchmark(data, artifacts=tmp_path / "runs")
    assert result["passed_execution"] and len(result["rows"]) == 4
    assert result["summary"]["wait"]["completion_fraction_mean_all_episodes"] == 0.0
    assert result["teacher"]["status"] == "passed_declared_gates"
    assert not result["teacher"]["expert_qualified"]
    frozen = json.loads((tmp_path / "runs/manifest.json").read_text())
    assert frozen["sha256"] == result["manifest_sha256"]
    with pytest.raises(ValueError, match="empty"):
        run_benchmark(data, artifacts=tmp_path / "runs")


@pytest.mark.parametrize("mutation", ["split", "ood", "seed", "policy", "duplicate"])
def test_invalid_benchmark_manifest(scenario, mutation):
    data = manifest(scenario)
    if mutation in {"split", "ood", "duplicate"}:
        second = copy.deepcopy(data["cases"][0])
        second["id"] = "two"
        if mutation == "split":
            second["split"] = "train"
        if mutation == "ood":
            second["split"] = "ood"
            second["scenario"]["lineage"] = "other"
        if mutation == "duplicate":
            second["id"] = "one"
        data["cases"].append(second)
    if mutation == "seed":
        data["cases"][0]["episode_seeds"] = [True]
    if mutation == "policy":
        data["policies"].append({"name": "priority"})
    with pytest.raises(ValueError):
        BenchmarkSpec.model_validate(data)


def test_benchmark_errors_stay_in_denominator_and_fail_teacher(scenario):
    data = manifest(scenario)
    bad = copy.deepcopy(data["cases"][0])
    bad["id"] = "bad"
    bad["scenario"]["definition"]["vehicles"] = {}
    data["cases"].append(bad)
    result = run_benchmark(data)
    assert len(result["rows"]) == 8 and not result["passed_execution"]
    assert result["summary"]["priority"]["episodes"] == 2
    assert result["summary"]["priority"]["completion_fraction_mean_all_episodes"] == 0.5
    assert result["teacher"]["status"] == "failed_declared_gates"


def test_teacher_requires_held_out_independent_cases(scenario):
    data = manifest(scenario)
    data["cases"][0]["split"] = "train"
    assert run_benchmark(data)["teacher"]["status"] == "failed_declared_gates"


def test_tiny_native_oracle_is_bounded_and_not_a_policy(scenario):
    t = task()
    t["decision_sec"] = 4.0
    result = tiny_oracle(TaskSession(scenario, t).checkpoint(), depth=1, max_nodes=16)
    assert result["proven_optimal_in_masked_tree"] and result["best_reward"] == 2.0
    bounded = tiny_oracle(TaskSession(scenario, t).checkpoint(), depth=2, max_nodes=1)
    assert bounded["budget_exhausted"] and not bounded["proven_optimal_in_masked_tree"]
    assert "continuous/full-mission" in result["scope"]


def test_cli_run_resume_replay_native_parity(scenario, tmp_path):
    for name, value in [
        ("scenario", scenario),
        ("task", task()),
        ("policy", {"name": "priority"}),
    ]:
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    args = {k: str(tmp_path / (k + ".json")) for k in ["scenario", "task", "policy"]}
    out = tmp_path / "prefix.json"
    run_command(
        SimpleNamespace(
            command="planning-run", **args, output=str(out), seed=None, decisions=2
        )
    )
    final = tmp_path / "final.json"
    run_command(
        SimpleNamespace(
            command="planning-resume",
            record=str(out),
            output=str(final),
            decisions=None,
        )
    )
    assert run_command(SimpleNamespace(command="planning-replay", record=str(final)))[
        "ok"
    ]
    direct = PlanningRun(scenario, task(), {"name": "priority"})
    direct.run()
    assert PlanningRun.load(final).session.checkpoint() == direct.session.checkpoint()


def test_runner_detects_out_of_band_task_changes(scenario):
    r = PlanningRun(scenario, task())
    r.session.step(
        {"@central": ["wait"]}, expected_revision=0, idempotency_key="outside"
    )
    with pytest.raises(ValueError, match="outside"):
        r.step()


def test_operator_booking_replans_without_double_counting_reserved_owner(scenario):
    t = task()
    t["actions"] = [a for a in t["actions"] if a["id"] == "ground-a"] + [
        {
            "id": "book",
            "role": "operator",
            "kind": "inject",
            "event": {
                "type": "terminal_booking_added",
                "payload": {
                    "id": "reserved",
                    "actor": "b",
                    "station": "Ground",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                },
            },
        }
    ]
    run = PlanningRun(scenario, t, {"name": "priority"}, allow_operator=True)
    run.step(["book"])
    before = make_input(run.session).view()["stations"]["Ground"]
    assert before["bookings"][0]["actor"] == "b"
    assert run.step()["feedback"]["accepted"]
    assert run.trace[-1]["decision"]["action_ids"] == ["wait"]
    resumed = PlanningRun.from_checkpoint(run.checkpoint())
    run.run()
    resumed.run()
    assert run.trace == resumed.trace and run.metrics()["fulfilled"] == 1
    assert run.session._task.policy_origin == "external"
    assert run.trace[0]["override"] == ["book"]


def test_operator_override_requires_explicit_capability(scenario):
    run = PlanningRun(scenario, task())
    before = run.checkpoint()
    with pytest.raises(PermissionError):
        run.step(["wait"])
    assert run.checkpoint() == before


def test_operator_station_recovery_is_native_and_replayed(scenario):
    scenario["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 0.0, "end_sec": 6.0}
    ]
    run = PlanningRun(scenario, task())
    run.step()
    assert "recover-ground" in run.trace[0]["decision"]["action_ids"]
    assert not run.session._runtime.station_observation("Ground")["outage"]
    assert PlanningRun.from_checkpoint(run.checkpoint()).trace == run.trace


def test_known_reservation_owner_does_not_consume_two_terminals():
    v = public([("send", "a", "downlink", "p", 1.0, 10.0, "radio", "G")]).view()
    v["stations"]["G"].update(bookings=[{"actor": "a"}], reservation_required=True)
    assert BaselinePolicy().decide(PlanningInput(canonical_json(v)))["action_ids"] == [
        "send"
    ]
    v["stations"]["G"]["bookings"] = [{"actor": "b"}]
    assert BaselinePolicy().decide(PlanningInput(canonical_json(v)))["action_ids"] == [
        "wait"
    ]


def test_batch_internal_resource_conflict_is_not_emitted():
    v = choice_input().view()
    v["actions"]["valuable"]["command_claims"] = [[["b", "radio"]], [["b", "radio"]]]
    assert (
        "valuable"
        not in BaselinePolicy().decide(PlanningInput(canonical_json(v)))["action_ids"]
    )


def test_teacher_seeds_or_relabelled_variants_do_not_establish_independent_support(
    scenario,
):
    data = manifest(scenario)
    data["teacher_gates"]["min_cases_per_family"] = 2
    second = copy.deepcopy(data["cases"][0])
    second["id"] = "relabelled"
    data["cases"].append(second)
    result = run_benchmark(data)
    assert any(
        f["reason"] == "insufficient_cases" for f in result["teacher"]["failures"]
    )


def test_future_booking_does_not_appear_as_current_station_state(scenario):
    from space_env.runtime import MissionRuntime

    payload = {
        "id": "future",
        "actor": "b",
        "station": "Ground",
        "start_sec": 4.0,
        "end_sec": 6.0,
    }
    m = MissionRuntime(scenario)
    m.inject(
        [
            {
                "id": "book",
                "type": "terminal_booking_added",
                "time_sec": 0.0,
                "source": "test",
                "reason": "published future allocation",
                "payload": payload,
            }
        ],
        expected_revision=0,
        idempotency_key="book",
    )
    assert m.station_observation("Ground")["bookings"] == []


def test_unreleased_templates_cannot_break_static_claim_projection(scenario):
    t = task()
    # Unsupported controls must remain masked; computing their dynamic admission is forbidden.
    t["actions"].append(
        {
            "id": "unavailable",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "x",
                    "actor_id": "a",
                    "kind": "imaging",
                    "duration_sec": 1.0,
                    "request_id": "unreleased",
                }
            ],
        }
    )
    p = make_input(TaskSession(scenario, t))
    assert p.view()["actions"]["unavailable"]["claims"] == []


def test_oracle_candidates_and_policy_durations_fail_explicitly(scenario):
    with pytest.raises(ValueError, match="candidate budget"):
        tiny_oracle(TaskSession(scenario, task()).checkpoint(), max_candidates=1)
    p = BaselinePolicy({"decision_sec": 6.0})
    before = p.checkpoint()
    with pytest.raises(ValueError, match="maximum"):
        p.decide(choice_input())
    assert p.checkpoint() == before


def test_evaluation_recomputation_ignores_measurements_but_not_semantics(scenario):
    data = manifest(scenario)
    data["policies"] = [{"name": "priority"}]
    data.pop("teacher_gates")
    result = run_benchmark(data)
    assert verify_evaluation(result)["episodes"] == 1
    result["rows"][0]["metrics"]["fulfilled"] = 999
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    with pytest.raises(ValueError, match="semantic replay"):
        verify_evaluation(result)


def test_ci_full_shards_cover_all_files_and_split_only_the_two_long_cases():
    from space_env.eval.ci_profiles import PROFILES, command, main

    other = command("full", "other")[4:]
    assert set(other) | {"tests/test_campaign.py", "tests/test_tasking.py"} == set(
        PROFILES["full"]
    )
    assert not set(other) & {"tests/test_campaign.py", "tests/test_tasking.py"}
    assert command("full", "campaign")[4:] == ["tests/test_campaign.py"]
    assert command("full", "tasking-base")[-2:] == ["-k", "not sixteen_vehicles"]
    assert command("full", "tasking-sixteen")[-2:] == ["-k", "sixteen_vehicles"]
    assert main(["reference", "--shard", "other"]) == 2
    assert main([]) == 2


def test_cli_cannot_overwrite_the_frozen_artifact_manifest(scenario, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest(scenario)))
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="outside"):
        run_command(
            SimpleNamespace(
                command="planning-evaluate",
                manifest=str(path),
                output=str(artifacts / "manifest.json"),
                artifacts=str(artifacts),
            )
        )
    assert not artifacts.exists()


@pytest.mark.parametrize("kind", ["trace", "episode", "operator"])
def test_malformed_control_record_has_contract_error(scenario, kind):
    run = PlanningRun(scenario, task())
    run.run(1)
    record = run.checkpoint()
    if kind == "trace":
        record["trace"][0].pop("override")
    if kind == "episode":
        record["episode"] = []
    if kind == "operator":
        record["operator_enabled"] = 1
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        PlanningRun.from_checkpoint(record)


def test_cli_failed_teacher_gates_exit_nonzero_and_preserve_report(
    scenario, tmp_path, capsys
):
    from space_env.cli import main

    data = manifest(scenario)
    data["cases"][0]["split"] = "train"
    path, output = tmp_path / "manifest.json", tmp_path / "report.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as exc:
        main(["planning-evaluate", str(path), str(output)])
    assert exc.value.code == 1
    assert (
        json.loads(output.read_text())["teacher"]["status"] == "failed_declared_gates"
    )
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_manifest_nested_inputs_are_detached_before_any_episode_runs(
    scenario, monkeypatch
):
    data = manifest(scenario)
    data["policies"] = [{"name": "priority"}]
    data.pop("teacher_gates")
    second = copy.deepcopy(data["cases"][0])
    second["id"] = "second"
    data["cases"].append(second)
    original = PlanningRun.run

    def mutate_caller(run, *a, **kw):
        data["cases"][1]["scenario"]["definition"]["vehicles"] = {}
        return original(run, *a, **kw)

    monkeypatch.setattr(PlanningRun, "run", mutate_caller)
    result = run_benchmark(data)
    assert result["passed_execution"]
    assert result["manifest"]["cases"][1]["scenario"]["definition"]["vehicles"]
