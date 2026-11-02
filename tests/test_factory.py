"""Synthetic draws, geometry-grouped splits, native gates and atomic corpora."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from space_env.factory import (
    FactorySpec,
    build_dataset,
    compile_factory,
    mission_fingerprint,
    parameter_registry,
    split_group,
    verify_bundle,
)
from space_env.factory.cli import run_command
from space_env.factory.quality import public_view_gate, quality_report
from space_env.planning import PlanningRun
from space_env.simulation.snapshot import canonical_json, digest
from tests.test_planning import scenario as planning_fixture
from tests.test_planning import task as planning_task


@pytest.fixture
def specification():
    scenario = planning_fixture.__wrapped__()
    task = planning_task()
    task["decision_sec"] = 2.0
    return {
        "id": "factory-test",
        "seed": 13,
        "templates": [
            {
                "id": "ground",
                "family": "delivery",
                "kind": "ground",
                "source": "synthetic unit fixture",
                "scenario": scenario,
                "task": task,
                "split": "train",
            }
        ],
        "policies": [{"name": "route_aware_rolling"}],
        "quality": {"require_teacher_gates": False},
        "export_roles": ["@central", "a"],
    }


def distribution(
    parameter="product_bits_scale", values=(0.75, 1.0), target="*", unit="ratio"
):
    return {
        "parameter": parameter,
        "target": target,
        "unit": unit,
        "values": list(values),
        "source": "synthetic bounded variation",
    }


def renamed(scenario, task):
    # Actor/product 'a' deliberately share a spelling; their namespaces differ.
    def actor_name(value, key=""):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                nk = (
                    {"a": "satellite-X", "b": "satellite-Y"}.get(k, k)
                    if key
                    in {
                        "vehicles",
                        "ephemerides",
                        "transmitters",
                        "endpoints",
                        "orbits",
                        "payloads",
                        "actors",
                    }
                    else k
                )
                result[nk] = actor_name(v, k)
            return result
        if isinstance(value, list):
            return [actor_name(v, key) for v in value]
        if isinstance(value, tuple):
            return [actor_name(v, key) for v in value]
        if isinstance(value, str) and key in {
            "actor",
            "actor_id",
            "source",
            "actor_order",
            "eligible_actors",
            "actors",
        }:
            return {"a": "satellite-X", "b": "satellite-Y"}.get(value, value)
        if isinstance(value, str) and key in {"destination", "destination_id"}:
            return {"b": "satellite-Y"}.get(value, value)
        return value

    s, t = actor_name(copy.deepcopy(scenario)), actor_name(copy.deepcopy(task))
    s.update(id="another-id", lineage="another-lineage", seed=999)
    t["id"] = "renamed-task"
    for i, a in enumerate(t["actions"]):
        a["id"] = "choice-" + str(i)
        for c in a.get("commands", []):
            c["command_id"] = "unimportant-" + str(i)
    return s, t


@pytest.mark.parametrize(
    "change",
    [
        "unit",
        "empty",
        "bool",
        "nan",
        "uniform",
        "count",
        "calibrated",
        "duplicate",
        "wildcard",
    ],
)
def test_distribution_contract_rejects_invalid_values(specification, change):
    p = distribution()
    if change == "unit":
        p["unit"] = "s"
    if change == "empty":
        p["values"] = []
    if change == "bool":
        p["values"] = [True]
    if change == "nan":
        p["values"] = [float("nan")]
    if change == "uniform":
        p.update(mode="uniform", low=2.0, high=1.0, values=[])
    if change == "count":
        p = distribution("station_terminals", [1.5], unit="count")
    if change == "calibrated":
        p["parameter_status"] = "calibrated"
    if change == "duplicate":
        p["values"] = [1.0, 1.0]
    templates = specification["templates"]
    templates[0]["distributions"] = [p]
    if change == "wildcard":
        templates[0]["distributions"].append(distribution(target="a"))
    with pytest.raises(ValueError):
        FactorySpec.model_validate(specification)


def test_parameter_registry_cannot_be_mutated():
    registry = parameter_registry()
    registry["memory_scale"] = "meters"
    assert parameter_registry()["memory_scale"] == "ratio"


def test_generation_is_deterministic_detached_and_global_rng_independent(specification):
    import numpy as np

    specification["templates"][0].update(samples=3, distributions=[distribution()])
    before = copy.deepcopy(specification)
    np.random.seed(7)
    first = compile_factory(specification)
    np.random.seed(81)
    assert compile_factory(specification) == first and specification == before
    assert first["coverage"]["draws"] == 3
    assert all(c["configuration_error"] is None for c in first["cases"])
    assert {c["scenario"]["lineage"] for c in first["cases"]} == {
        "mission-" + first["cases"][0]["group_sha256"]
    }


def test_ids_seeds_and_actor_aliases_do_not_escape_semantic_group(specification):
    template = specification["templates"][0]
    s, t = renamed(template["scenario"], template["task"])
    assert split_group(s) == split_group(template["scenario"])
    assert mission_fingerprint(s, t) == mission_fingerprint(
        template["scenario"], template["task"]
    )
    other = copy.deepcopy(template)
    other.update(id="renamed", scenario=s, task=t, split="holdout")
    specification["templates"].append(other)
    with pytest.raises(ValueError, match="geometry cannot cross"):
        compile_factory(specification)


def test_role_labels_are_not_actor_or_command_kind_namespaces(specification):
    s = specification["templates"][0]["scenario"]
    t = specification["templates"][0]["task"]
    other = copy.deepcopy(t)
    other["roles"]["controller-a"] = other["roles"].pop("a")
    for a in other["actions"]:
        if a["role"] == "a":
            a["role"] = "controller-a"
    assert mission_fingerprint(s, t) == mission_fingerprint(s, other)


def test_resource_workload_changes_do_not_make_independent_geometry(specification):
    a = specification["templates"][0]["scenario"]
    b = copy.deepcopy(a)
    b["definition"]["vehicles"]["a"]["memory_bits"] *= 2
    b["definition"]["products"][0]["size_bits"] *= 0.5
    b["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 0.0, "end_sec": 8.0}
    ]
    assert split_group(a) == split_group(b)
    assert mission_fingerprint(a) != mission_fingerprint(b)


def test_geometry_change_is_distinct_and_order_is_irrelevant(specification):
    s = specification["templates"][0]["scenario"]
    other = copy.deepcopy(s)
    other["actor_order"] = list(reversed(other["actor_order"]))
    assert split_group(s) == split_group(other)
    other["definition"]["stations"][0]["lat_deg"] += 0.1
    assert split_group(s) != split_group(other)


def test_uniform_draws_and_template_order_do_not_change_samples(specification):
    first = specification["templates"][0]
    first.update(
        samples=2,
        distributions=[
            {**distribution(), "values": [], "mode": "uniform", "low": 0.5, "high": 1.0}
        ],
    )
    second = copy.deepcopy(first)
    second["id"] = "copy"
    specification["templates"].append(second)
    a = compile_factory(specification)
    specification["templates"].reverse()
    assert compile_factory(specification)["cases"] == a["cases"]
    assert a["cases"][0]["parameters"] == a["cases"][2]["parameters"]


@pytest.mark.parametrize(
    "parameter,unit,value",
    [
        ("memory_scale", "ratio", 2.0),
        ("battery_scale", "ratio", 0.8),
        ("rate_scale", "ratio", 0.8),
        ("deadline_scale", "ratio", 0.8),
        ("processing_scale", "ratio", 1.0),
        ("station_terminals", "count", 2.0),
        ("relay_ttl_sec", "s", 2.0),
    ],
)
def test_supported_parameter_adapters_use_native_validation(
    specification, parameter, unit, value
):
    specification["templates"][0]["distributions"] = [
        distribution(parameter, [value], unit=unit)
    ]
    c = compile_factory(specification)["cases"][0]
    assert c["configuration_error"] is None


def test_invalid_draw_is_retained_instead_of_resampled(specification):
    specification["templates"][0].update(
        samples=2, distributions=[distribution("memory_scale", [0.0001])]
    )
    compiled = compile_factory(specification)
    assert len(compiled["cases"]) == compiled["coverage"]["configuration_errors"] == 2


def test_missing_target_and_family_capability_are_explicit_errors(specification):
    t = specification["templates"][0]
    t["distributions"] = [distribution(target="not-a-product")]
    assert compile_factory(specification)["cases"][0]["configuration_error"]
    t.pop("distributions")
    t["kind"] = "eo_processing"
    assert compile_factory(specification)["cases"][0]["configuration_error"]


def test_byte_and_case_budgets_fail_without_silent_truncation(specification):
    specification["max_generated_bytes"] = 1024
    with pytest.raises(ValueError, match="byte budget"):
        compile_factory(specification)
    specification.pop("max_generated_bytes")
    specification["templates"][0]["samples"] = 33
    with pytest.raises(ValueError):
        FactorySpec.model_validate(specification)


def test_ood_family_is_not_renamed_into_training(specification):
    second = copy.deepcopy(specification["templates"][0])
    second.update(id="ood", split="ood")
    second["scenario"]["definition"]["stations"][0]["lat_deg"] += 0.05
    specification["templates"].append(second)
    with pytest.raises(ValueError, match="OOD family"):
        compile_factory(specification)


def test_bundle_build_replay_split_files_and_public_data(specification, tmp_path):
    path = tmp_path / "corpus"
    result = build_dataset(specification, path)
    assert result["ok"] and result["rows"]["rl"] == 8 and result["rows"]["sft"] == 4
    verified = verify_bundle(path)
    assert (
        verified["ok"]
        and verified["replay_verified"]
        and verified["content_sha256"] == result["content_sha256"]
    )
    rl = [
        json.loads(s) for s in (path / "data/train/rl.jsonl").read_text().splitlines()
    ]
    assert {r["view_role"] for r in rl} == {"@central", "a"}
    assert all(r["split"] == "train" and not r["expert_qualified"] for r in rl)
    assert (path / "data/holdout/rl.jsonl").read_text() == ""
    for r in rl:
        public_view_gate(r["observation"])
        public_view_gate(r["next_observation"])
    sft = [
        json.loads(s) for s in (path / "data/train/sft.jsonl").read_text().splitlines()
    ]
    assert all(
        r["view_role"] == "@central" and r["policy_origin"] == "scripted" for r in sft
    )
    assert "episode_seed" not in json.dumps([r["messages"] for r in sft])


def test_renamed_or_seed_only_case_variants_share_splits(specification):
    template = specification["templates"][0]
    other = copy.deepcopy(template)
    other["id"] = "new-name"
    other["scenario"]["seed"] = 500
    other["scenario"]["id"] = "new"
    specification["templates"].append(other)
    compiled = compile_factory(specification)
    assert (
        compiled["coverage"]["geometry_groups"]
        == compiled["coverage"]["unique_missions"]
        == 1
    )


def test_identical_policy_trajectories_are_deduplicated_not_counted_as_diversity(
    specification, tmp_path
):
    specification["policies"] = [{"name": "priority"}, {"name": "route_aware_rolling"}]
    path = tmp_path / "dedup"
    result = build_dataset(specification, path)
    assert result["ok"] and result["rows"] == {"rl": 8, "sft": 4}
    report = json.loads((path / "report.json").read_text())
    assert report["episodes"] == 2
    assert any(
        "duplicate_trajectory" in e["excluded"]
        for a in report["audit"]
        for e in a["exports"]
    )


def test_dataset_bytes_repeat_even_when_runtime_timings_change(specification, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    x, y = build_dataset(specification, a), build_dataset(specification, b)
    assert x["content_sha256"] == y["content_sha256"]
    assert (a / "data/train/rl.jsonl").read_bytes() == (
        b / "data/train/rl.jsonl"
    ).read_bytes()
    assert (a / "bundle.json").read_bytes() != (
        b / "bundle.json"
    ).read_bytes()  # Audit timings are measured.


def test_strict_teacher_requirement_quarantines_unqualified_sft(
    specification, tmp_path
):
    specification["quality"]["require_teacher_gates"] = True
    p = tmp_path / "unqualified"
    result = build_dataset(specification, p)
    assert (
        not result["ok"]
        and result["status"] == "quarantined"
        and result["rows"]["sft"] == 0
    )
    assert verify_bundle(p)["status"] == "quarantined"


def test_blocked_valid_episode_is_rl_data_not_sft(specification, tmp_path):
    specification["templates"][0]["task"]["actions"] = [
        a
        for a in specification["templates"][0]["task"]["actions"]
        if a["kind"] != "inject"
    ]
    specification["templates"][0]["scenario"]["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 0.0, "end_sec": 8.0}
    ]
    specification["quality"]["min_sft_rows"] = 0
    result = build_dataset(specification, tmp_path / "blocked")
    assert result["ok"] and result["rows"] == {"rl": 8, "sft": 0}
    assert (
        json.loads((tmp_path / "blocked/report.json").read_text())["episode_errors"]
        == 0
    )


def test_invalid_samples_stay_in_audit_and_quarantine(specification, tmp_path):
    specification["templates"][0]["distributions"] = [
        distribution("memory_scale", [0.0001])
    ]
    path = tmp_path / "invalid"
    result = build_dataset(specification, path)
    report = json.loads((path / "report.json").read_text())
    assert not result["ok"] and report["episodes"] == report["episode_errors"] == 1
    assert report["coverage"]["draws"] == 1 and result["rows"] == {"rl": 0, "sft": 0}
    assert verify_bundle(path)["status"] == "quarantined"


@pytest.mark.parametrize(
    "field", ["seed", "scenario", "runtime", "checkpoint", "truth"]
)
def test_private_fields_cannot_enter_model_views(field):
    with pytest.raises(ValueError, match="privileged"):
        public_view_gate({"nested": [{field: 0}]})


def test_native_accounting_gate_detects_broken_ledger(specification, monkeypatch):
    t = specification["templates"][0]
    run = PlanningRun(t["scenario"], t["task"])
    run.run()
    original = run.session._runtime.observe

    def corrupt(*a, **kw):
        result = original(*a, **kw)
        if kw.get("view") == "truth":
            result["backend"]["report"]["inventory_ledger"]["a"][
                "balance_error_bits"
            ] = 1.0
        return result

    monkeypatch.setattr(run.session._runtime, "observe", corrupt)
    with pytest.raises(ValueError, match="conservation"):
        quality_report(run)


def rehash(path):
    receipt = json.loads((path / "bundle.json").read_text())
    for name in receipt["files"]:
        data = (path / name).read_bytes()
        receipt["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    (path / "bundle.json").write_text(canonical_json(receipt))


def test_rehashed_training_row_tampering_fails_semantic_verification(
    specification, tmp_path
):
    p = tmp_path / "tampered"
    build_dataset(specification, p)
    data = p / "data/train/rl.jsonl"
    rows = [json.loads(s) for s in data.read_text().splitlines()]
    rows[0]["reward"] = 900.0
    data.write_text("".join(canonical_json(r) + "\n" for r in rows))
    rehash(p)
    assert verify_bundle(p, replay=False)["replay_verified"] is False
    with pytest.raises(ValueError, match="training row replay"):
        verify_bundle(p)


@pytest.mark.parametrize("kind", ["extra", "missing", "symlink", "unsafe"])
def test_bundle_file_inventory_and_paths(specification, tmp_path, kind):
    p = tmp_path / "files"
    build_dataset(specification, p)
    if kind == "extra":
        (p / "unlisted.txt").write_text("noise")
    if kind == "missing":
        (p / "data/train/rl.jsonl").unlink()
    if kind == "symlink":
        (p / "foreign").symlink_to(tmp_path)
    if kind == "unsafe":
        r = json.loads((p / "bundle.json").read_text())
        r["files"]["../foreign"] = {"bytes": 0, "sha256": "0" * 64}
        r["sha256"] = digest({k: v for k, v in r.items() if k != "sha256"})
        (p / "bundle.json").write_text(canonical_json(r))
    with pytest.raises(ValueError):
        verify_bundle(p, replay=False)


@pytest.mark.parametrize("budget", ["rows", "bytes"])
def test_failed_build_has_no_partially_published_directory(
    specification, tmp_path, budget
):
    specification["quality"]["max_rows" if budget == "rows" else "max_bundle_bytes"] = (
        1 if budget == "rows" else 1024
    )
    p = tmp_path / "atomic"
    with pytest.raises(ValueError, match="budget"):
        build_dataset(specification, p)
    assert not p.exists() and not list(tmp_path.glob(".factory-*"))


def test_existing_output_and_cli_input_are_preserved(specification, tmp_path):
    p = tmp_path / "existing"
    p.mkdir()
    (p / "mine").write_text("keep")
    with pytest.raises(ValueError):
        build_dataset(specification, p)
    assert (p / "mine").read_text() == "keep"
    spec = tmp_path / "input.json"
    spec.write_text(json.dumps(specification))
    with pytest.raises(ValueError):
        run_command(
            SimpleNamespace(command="factory-compile", spec=str(spec), output=str(spec))
        )


def test_cli_schema_compile_build_verify_and_quarantine_exit(
    specification, tmp_path, capsys
):
    from space_env.cli import main

    spec, schema, catalog, bundle = [
        tmp_path / p for p in ["spec.json", "schema.json", "catalogue.json", "bundle"]
    ]
    spec.write_text(json.dumps(specification))
    main(["factory-schema", str(schema)])
    assert "templates" in json.loads(schema.read_text())["properties"]
    main(["factory-compile", str(spec), str(catalog)])
    assert json.loads(catalog.read_text())["coverage"]["draws"] == 1
    main(["factory-build", str(spec), str(bundle)])
    main(["factory-verify", str(bundle)])
    assert all(json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines())
    specification["quality"]["require_teacher_gates"] = True
    spec.write_text(json.dumps(specification))
    bad = tmp_path / "quarantine"
    with pytest.raises(SystemExit) as exc:
        main(["factory-build", str(spec), str(bad)])
    assert exc.value.code == 1 and (bad / "report.json").exists()


def test_indistinguishable_training_trajectories_cannot_cross_splits(
    specification, tmp_path
):
    base = specification["templates"][0]
    base["scenario"]["definition"].pop("radio_network")
    base["scenario"]["definition"].pop("dtn_network")
    other = copy.deepcopy(base)
    other.update(id="heldout", split="holdout")
    other["scenario"]["definition"]["stations"][0]["lat_deg"] += 0.01
    specification["templates"].append(other)
    result = build_dataset(specification, tmp_path / "leak")
    report = json.loads((tmp_path / "leak/report.json").read_text())
    assert not result["ok"]
    assert any(
        "duplicate trajectory crosses" in a["error"]["message"]
        for a in report["audit"]
        if a["error"]
    )


def test_late_role_export_failure_rolls_back_the_whole_episode(
    specification, tmp_path, monkeypatch
):
    import space_env.factory.bundle as module

    original = module.VerifiedEpisode.export

    def fail(self, output, **kw):
        if kw["role"] == "a":
            raise ValueError("deliberate local export failure")
        return original(self, output, **kw)

    monkeypatch.setattr(module.VerifiedEpisode, "export", fail)
    result = build_dataset(specification, tmp_path / "rollback")
    assert not result["ok"] and result["rows"] == {"rl": 0, "sft": 0}


def test_actor_station_and_role_renaming_cannot_change_geometry_fold(specification):
    template = specification["templates"][0]
    s = copy.deepcopy(template["scenario"])

    def change(v):
        if isinstance(v, dict):
            return {
                ("Receiver" if k == "Ground" else k): change(x) for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [change(x) for x in v]
        return "Receiver" if v == "Ground" else v

    assert split_group(change(s)) == split_group(s)


def test_sft_rows_contain_the_actual_tool_and_action_contract(specification, tmp_path):
    p = tmp_path / "schema"
    build_dataset(specification, p)
    row = json.loads((p / "data/train/sft.jsonl").read_text().splitlines()[0])
    call = row["messages"][1]["tool_calls"][0]["function"]
    assert (
        row["tools"][0]["function"]["name"]
        == call["name"]
        == "space_env_task_joint_step"
    )
    assert row["action_map"]["ids"][0] == "wait"
    assert set(json.loads(call["arguments"])["choices"]) == {"@central"}


def test_eo_cpu_relay_parameter_adapters_use_the_coupled_profile():
    import importlib.util

    p = Path(__file__).resolve().parents[1] / "examples/factory/demo.py"
    loader = importlib.util.spec_from_file_location("factory_example", p)
    m = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(m)
    spec = m.eo_spec()
    original_work = spec["templates"][0]["scenario"]["definition"]["lifecycle"]["jobs"][
        0
    ]["work_sec"]
    spec["templates"][0]["kind"] = "eo_processing"
    spec["templates"][0]["distributions"] += [
        distribution("processing_scale", [0.5]),
        distribution("acquisition_scale", [0.5]),
    ]
    c = compile_factory(spec)["cases"][0]
    assert c["configuration_error"] is None
    assert (
        c["scenario"]["definition"]["lifecycle"]["jobs"][0]["work_sec"]
        == original_work * 0.5
    )
    assert (
        c["scenario"]["definition"]["lifecycle"]["requests"][0]["acquisition_sec"]
        == 1.0
    )


def test_outage_duration_and_recovery_family_are_native(specification):
    t = specification["templates"][0]
    t["kind"] = "recovery"
    t["scenario"]["definition"]["outages"] = [
        {"station": "Ground", "start_sec": 0.0, "end_sec": 1.0}
    ]
    t["distributions"] = [distribution("outage_duration_sec", [2.0], unit="s")]
    c = compile_factory(specification)["cases"][0]
    assert (
        c["configuration_error"] is None
        and c["scenario"]["definition"]["outages"][0]["end_sec"] == 2.0
    )


def test_teacher_gate_cannot_qualify_another_policy(specification):
    from space_env.factory.quality import sft_reasons

    specification["policies"] = [
        {"name": "priority"},
        {"name": "earliest_deadline"},
        {"name": "route_aware_rolling"},
    ]
    specification["sft_policies"] = ["priority"]
    specification["teacher_gates"] = {"candidate": "route_aware_rolling"}
    specification["quality"]["require_teacher_gates"] = True
    spec = FactorySpec.model_validate(specification)
    reasons = sft_reasons(
        spec,
        {"teacher": {"status": "passed_declared_gates"}},
        {
            "policy": "priority",
            "metrics": {"completion_fraction": 1.0, "rejections": 0},
        },
    )
    assert "teacher_gates_not_passed_for_policy" in reasons


def test_catalogue_mutation_by_the_caller_does_not_change_predeclared_spec(
    specification,
):
    c = compile_factory(specification)
    specification["templates"][0]["scenario"]["definition"]["vehicles"]["a"][
        "memory_bits"
    ] = 0.0
    assert (
        c["spec"]["templates"][0]["scenario"]["definition"]["vehicles"]["a"][
            "memory_bits"
        ]
        > 0.0
    )


def test_generated_small_case_can_be_checked_by_the_native_bounded_oracle(
    specification,
):
    from space_env.planning.oracle import tiny_oracle
    from space_env.tasks import TaskSession

    specification["templates"][0]["distributions"] = [distribution(values=[0.75])]
    c = compile_factory(specification)["cases"][0]
    session = TaskSession(c["scenario"], c["task"], seed=c["episode_seed"])
    proof = tiny_oracle(session.checkpoint(), depth=1, max_nodes=16)
    assert proof["proven_optimal_in_masked_tree"] and proof[
        "best_reward"
    ] == pytest.approx(1.5)


@pytest.mark.parametrize("kind", ["not_object", "files_not_object"])
def test_malformed_receipts_raise_contract_errors(specification, tmp_path, kind):
    p = tmp_path / "malformed"
    build_dataset(specification, p)
    receipt = json.loads((p / "bundle.json").read_text())
    if kind == "not_object":
        receipt = []
    else:
        receipt["files"] = []
        receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    (p / "bundle.json").write_text(canonical_json(receipt))
    with pytest.raises(ValueError):
        verify_bundle(p)


def test_nonboolean_verification_mode_is_rejected(tmp_path):
    with pytest.raises(TypeError):
        verify_bundle(tmp_path, replay="false")
