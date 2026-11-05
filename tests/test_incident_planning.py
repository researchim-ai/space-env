"""Phase-aware disruption recovery; native conservation and retained failures."""

import copy
import json
from unittest.mock import patch

import pytest

from examples.planning.incident_mission import configuration, run
from space_env.planning.incident import (
    IncidentSpec,
    recover_incident,
    schedule_recovery,
    verify_incident,
)
from space_env.planning.mission_bundle import verify_plan_bundle
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    result = {}
    for mode in ("partial_tx", "paused_cpu", "lost_capture", "blocked"):
        root = tmp_path_factory.mktemp("incident") / "run"
        metrics = run(configuration(mode), root)
        result[mode] = (
            mode,
            root,
            metrics,
            json.loads((root / "recovery.json").read_text()),
        )
    return result


@pytest.fixture(
    scope="module", params=["partial_tx", "paused_cpu", "lost_capture", "blocked"]
)
def native_case(request, cases):
    return cases[request.param]


@pytest.fixture(scope="module")
def partial_case(cases):
    return cases["partial_tx"]


def test_native_results_and_full_denominator(native_case):
    mode, root, result, p = native_case
    success = mode in ("partial_tx", "paused_cpu")
    assert result["total"] == 24
    assert result["fulfilled"] == result["predicted_fulfilled"] == (2 if success else 0)
    assert (
        result["mission_goal_passed"] is success and result["sft_exported"] is success
    )
    assert (root / "dataset/sft.jsonl").exists() is success
    assert verify_plan_bundle(root / "dataset", native=True)[
        "native_execution_replayed"
    ]
    assert not result["expert_teacher_qualified"]
    assert (
        p["plan"]["steps"][: len(p["source_record"]["episode"]["transitions"])]
        == p["source_proposal"]["plan"]["steps"][
            : len(p["source_record"]["episode"]["transitions"])
        ]
    )


def test_partial_state_and_no_duplicate_acquisition(native_case):
    mode, root, _, p = native_case
    native = json.loads((root / "execution.json").read_text())["episode"]["runtime"][
        "backend_report"
    ]
    life = native["lifecycle"]
    if mode in ("partial_tx", "blocked"):
        assert sum(x["phase"] == "final" for x in p["states"].values()) == 2
        assert all(
            x["remaining_bits"] == pytest.approx(400000.0)
            for x in p["states"].values()
            if x["phase"] == "final"
        )
        assert all(a["phase"] == "final" for a in p["schedule"]["allocations"])
    if mode == "paused_cpu":
        assert sum(x["phase"] == "raw" for x in p["states"].values()) == 2
        assert all(
            x["remaining_cpu_sec"] == pytest.approx(6.0)
            for x in p["states"].values()
            if x["phase"] == "raw"
        )
        assert all(
            x["work_sec"] == pytest.approx(8.0)
            for x in life["jobs"].values()
            if x["status"] == "completed"
        )
    if mode == "lost_capture":
        assert sum(x["phase"] == "terminal" for x in p["states"].values()) == 2
    if mode in ("partial_tx", "paused_cpu", "blocked"):
        assert sum(x["source"] == "acquired" for x in life["artifacts"].values()) == 2
        assert (
            sum(x["source"] == "transformed_out" for x in life["artifacts"].values())
            == 2
        )
    source_actors = p["source_proposal"]["plan"]["task"]["roles"]
    assert all(a in p["plan"]["task"]["roles"] for a in source_actors)


def test_value_only_recovery_search_and_native_recomputation(native_case):
    _, _, _, p = native_case
    with patch(
        "space_env.planning.incident.PlanExecution",
        side_effect=AssertionError("rollout oracle"),
    ):
        assert (
            schedule_recovery(p["forecast"], p["states"], p["schedule"]["settings"])
            == p["schedule"]
        )
    assert verify_incident(p)["committed_prefix_native_replayed"]


@pytest.mark.parametrize(
    "payload",
    [
        {"disruptions": [{"station": "g02", "end_sec": True}]},
        {"disruptions": [{"station": "g02", "end_sec": float("nan")}]},
        {"disruptions": [{"station": "g02", "end_sec": -1.0}]},
        {"disruptions": [], "new_demand": {}},
    ],
)
def test_strict_bounded_incident_inputs(payload):
    with pytest.raises(ValueError):
        IncidentSpec.model_validate(payload)


def test_task_outage_authority_is_station_scoped():
    task = {
        "id": "incident-authority",
        "roles": {"operator": {"stations": ["g02"]}},
        "actions": [
            {
                "id": "outage",
                "kind": "inject",
                "role": "operator",
                "event": {
                    "type": "station_outage_added",
                    "payload": {"station": "g02", "start_sec": 1.0, "end_sec": 2.0},
                },
            }
        ],
    }
    assert TaskSpec.model_validate(task)
    task["actions"][0]["event"]["payload"]["station"] = "foreign"
    with pytest.raises(ValueError, match="authority"):
        TaskSpec.model_validate(task)


def test_unknown_station_and_backdated_end_refused(partial_case):
    _, _, _, p = partial_case
    for incident in (
        {"disruptions": [{"station": "foreign", "end_sec": 120.0}]},
        {"disruptions": [{"station": "g02", "end_sec": 1.0}]},
    ):
        with pytest.raises(ValueError):
            recover_incident(p["source_proposal"], p["source_record"], incident)


def test_resealed_false_progress_is_refused(partial_case):
    _, _, _, p = partial_case
    p = copy.deepcopy(p)
    first = next(iter(p["states"]))
    p["states"][first]["remaining_bits"] = 1.0
    p["sha256"] = digest({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError, match="recomputation"):
        verify_incident(p)


def test_cli_native_audit_and_no_overwrite(partial_case, tmp_path, capsys):
    from space_env.cli import main

    _, _, _, p = partial_case
    path = tmp_path / "incident.json"
    path.write_text(canonical_json(p))
    assert main(["verify-chain-incident", str(path)]) is None
    assert json.loads(capsys.readouterr().out)["native_feasibility_validated"] is False


@pytest.fixture(scope="module")
def role_bundles(cases, tmp_path_factory):
    from space_env.planning.incident_data import publish_incident_data

    result = {}
    for mode in ("partial_tx", "blocked"):
        _, root, _, recovery = cases[mode]
        execution = json.loads((root / "execution.json").read_text())
        output = tmp_path_factory.mktemp("role-bundle") / "data"
        manifest = publish_incident_data(recovery, execution, output)
        result[mode] = (output, manifest, recovery)
    return result


def test_actor_suffix_data_excludes_operator_fault_targets(role_bundles):
    for mode, (root, manifest, recovery) in role_bundles.items():
        assert not manifest["operator_trace_trainable_as_agent"]
        assert any(
            x["format"] == "sft" for x in manifest["training_files"].values()
        ) is (mode == "partial_tx")
        owners = {a["id"]: a["role"] for a in recovery["plan"]["task"]["actions"]}
        for name, meta in manifest["training_files"].items():
            rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
            assert rows and all(
                r["index"] >= recovery["committed_prefix_steps"] for r in rows
            )
            assert all(r["reward_scope"] == "joint_team" for r in rows)
            if meta["format"] == "rl":
                for row in rows:
                    assert all(
                        a["id"] == "wait" or owners[a["id"]] == meta["role"]
                        for a in row["observation"]["catalogue"]
                    )
                    assert len(row["observation"]["catalogue"]) == len(
                        row["observation"]["action_mask"]
                    )
                    assert (
                        row["observation"]["schema"]
                        == "space-env.incident-actor-observation/1"
                    )
                    assert all(
                        a == "wait" or owners[a] == meta["role"]
                        for actions in row["action"]["choices"].values()
                        for a in actions
                    )


def test_actor_data_native_regeneration_and_corruption(role_bundles):
    from space_env.planning.incident_data import verify_incident_data

    for root, _, _ in role_bundles.values():
        assert verify_incident_data(root, native=True)["native_rows_recomputed"]
    root, manifest, _ = role_bundles["partial_tx"]
    path = root / next(iter(manifest["training_files"]))
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_incident_data(root)


@pytest.mark.parametrize(
    "payload",
    [
        {"station": "g02"},
        {"station": "g02", "start_sec": True, "end_sec": 2.0},
        {"station": "g02", "start_sec": 3.0, "end_sec": 2.0},
        {"station": "g02", "start_sec": 0.0, "end_sec": float("inf")},
    ],
)
def test_task_outage_rejects_malformed_declaration_early(payload):
    with pytest.raises(ValueError, match="unsupported"):
        TaskSpec.model_validate(
            {
                "id": "invalid-outage",
                "roles": {"operator": {"stations": ["g02"]}},
                "actions": [
                    {
                        "id": "outage",
                        "kind": "inject",
                        "role": "operator",
                        "event": {"type": "station_outage_added", "payload": payload},
                    }
                ],
            }
        )
