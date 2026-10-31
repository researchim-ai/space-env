"""B01 cross-backend scenario, command, access, checkpoint and CLI acceptance."""

import copy
import json

import pytest

from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    Runtime,
    RuntimeContractError,
    legacy_scenario,
    reference_scenario,
    scenario_json_schema,
)
from space_env.simulation.snapshot import canonical_json, digest
from tests import test_reference_operations as reference_fixtures
from tests.test_execution_integrity import world


@pytest.fixture(scope="module")
def spec():
    return reference_fixtures.spec.__wrapped__()


@pytest.fixture(params=["reference", "legacy"])
def scenario(request, spec):
    if request.param == "reference":
        return reference_scenario(spec, id="reference", lineage="runtime-family")
    return legacy_scenario(world(), id="legacy", lineage="runtime-family")


def privileged(scenario):
    return MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))


def downlink(scenario, *, id="cmd", duration=2, actor=None):
    ref = scenario["backend"] == "reference-fixed"
    return {
        "command_id": id,
        "actor_id": actor or scenario["actor_order"][0],
        "kind": "downlink",
        "duration_sec": duration,
        "destination_id": "Ground" if ref else "Shared",
    }


def test_common_contract_command_duration_checkpoint_and_continuation(
    scenario, tmp_path
):
    runtime = privileged(scenario)
    assert isinstance(runtime, Runtime)
    runtime.submit([downlink(scenario)], expected_revision=0, idempotency_key="submit")
    runtime.advance(5, expected_revision=1, idempotency_key="advance")
    assert runtime.report()["commands"][0]["status"] == "completed"
    assert runtime.report()["metrics"][scenario["objective"]] == pytest.approx(2e6)
    assert runtime.time_sec == 5 and runtime.revision == 2
    path = tmp_path / "checkpoint.json"
    runtime.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    assert restored.report() == runtime.report()
    assert restored.observe() == runtime.observe()
    runtime.advance(1, expected_revision=2, idempotency_key="continue")
    restored.advance(1, expected_revision=2, idempotency_key="continue")
    assert restored.checkpoint() == runtime.checkpoint()


def test_idempotency_after_later_revision_never_spends_time_or_changes_history(
    scenario,
):
    runtime = privileged(scenario)
    response = runtime.submit(
        [downlink(scenario)], expected_revision=0, idempotency_key="s"
    )
    runtime.advance(1, expected_revision=1, idempotency_key="a")
    before = runtime.checkpoint()
    assert (
        runtime.submit([downlink(scenario)], expected_revision=0, idempotency_key="s")
        == response
    )
    assert runtime.checkpoint() == before
    with pytest.raises(RuntimeContractError, match="another body"):
        runtime.submit(
            [downlink(scenario, duration=3)], expected_revision=2, idempotency_key="s"
        )
    with pytest.raises(RuntimeContractError) as e:
        runtime.advance(1, expected_revision=0, idempotency_key="stale")
    assert e.value.code == "stale_revision"
    assert runtime.checkpoint() == before


def test_batch_conflict_and_duplicate_id_are_atomic(scenario):
    runtime = privileged(scenario)
    before = runtime.checkpoint()
    with pytest.raises(RuntimeContractError) as e:
        runtime.submit(
            [downlink(scenario), downlink(scenario, id="other")],
            expected_revision=0,
            idempotency_key="batch",
        )
    assert e.value.code == "actor_busy" and runtime.checkpoint() == before
    runtime.submit([downlink(scenario)], expected_revision=0, idempotency_key="first")
    before = runtime.checkpoint()
    with pytest.raises(RuntimeContractError) as e:
        runtime.submit(
            [downlink(scenario)], expected_revision=1, idempotency_key="duplicate"
        )
    assert e.value.code == "duplicate_command" and runtime.checkpoint() == before


def test_cancel_running_and_not_started_control_windows(scenario):
    runtime = privileged(scenario)
    runtime.submit(
        [downlink(scenario, duration=5)], expected_revision=0, idempotency_key="s"
    )
    runtime.advance(1, expected_revision=1, idempotency_key="a")
    before = runtime.report()["metrics"].copy()
    runtime.cancel("cmd", expected_revision=2, idempotency_key="c")
    runtime.advance(4, expected_revision=3, idempotency_key="idle")
    assert runtime.report()["metrics"] == before
    assert runtime.report()["commands"][0]["status"] == "cancelled"
    runtime.submit(
        [downlink(scenario, id="never-started", duration=1)],
        expected_revision=4,
        idempotency_key="next",
    )
    runtime.cancel(
        "never-started", expected_revision=5, idempotency_key="cancel-pending"
    )
    runtime.advance(1, expected_revision=6, idempotency_key="idle2")
    assert runtime.report()["metrics"] == before


def test_actor_view_cannot_access_truth_artifacts_or_other_actor(spec):
    runtime = MissionRuntime(reference_scenario(spec, id="x", lineage="family"))
    actor = runtime.agent("a")
    view = actor.observe()
    assert set(view["actors"]) == {"a"}
    assert "ephemerides" not in canonical_json(
        view
    ) and "definition" not in canonical_json(view)
    for action in (
        lambda: runtime.observe(view="truth"),
        runtime.checkpoint,
        runtime.iter_events,
        lambda: runtime.forecast("a", "Ground"),
    ):
        with pytest.raises(RuntimeContractError) as e:
            action()
        assert e.value.code == "permission_denied"
    with pytest.raises(RuntimeContractError):
        actor.submit(
            {
                **downlink(reference_scenario(spec, id="x", lineage="f")),
                "actor_id": "b",
            },
            expected_revision=0,
            idempotency_key="bad",
        )
    assert runtime.revision == 0


def test_privileged_views_do_not_change_state_and_forecast_is_explicit(spec):
    runtime = privileged(reference_scenario(spec, id="x", lineage="family"))
    before = runtime.checkpoint()
    assert runtime.observe(view="truth")["backend"]["states"]["a"]["frame"] == "GCRS"
    assert runtime.forecast("a", "Ground")["windows"]
    assert runtime.checkpoint() == before


def test_unsupported_commands_and_capabilities_never_fall_back(scenario):
    runtime = privileged(scenario)
    before = runtime.checkpoint()
    kind = "imaging" if scenario["backend"] == "reference-fixed" else "processing"
    with pytest.raises(RuntimeContractError) as e:
        runtime.submit(
            [{**downlink(scenario), "kind": kind, "destination_id": ""}],
            expected_revision=0,
            idempotency_key="bad",
        )
    assert e.value.code == "unsupported_capability" and runtime.checkpoint() == before
    bad = copy.deepcopy(scenario)
    bad["required_capabilities"] = ["commands.interplanetary"]
    with pytest.raises(RuntimeContractError) as e:
        MissionRuntime(bad)
    assert e.value.code == "unsupported_capability"


def test_explicit_cpu_commands_idle_and_event_boundaries(spec):
    data = copy.deepcopy(spec)
    data["products"][0].update(processing_sec=1)
    scenario = reference_scenario(data, id="cpu", lineage="family")
    runtime = privileged(scenario)
    runtime.advance(1, expected_revision=0, idempotency_key="idle")
    assert runtime._backend.mission.processing_remaining["a"] == 1
    runtime.submit(
        [
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "processing",
                "product_id": "a",
                "duration_sec": 1,
            }
        ],
        expected_revision=1,
        idempotency_key="cpu",
    )
    runtime.advance(1, expected_revision=2, idempotency_key="compute")
    assert runtime._backend.mission.processing_remaining["a"] == 0
    runtime.submit([downlink(scenario)], expected_revision=3, idempotency_key="tx")
    runtime.advance(2, expected_revision=4, idempotency_key="tx-end")
    event = {
        "id": "fault",
        "type": "station_outage_added",
        "time_sec": 4,
        "source": "operator",
        "reason": "fault",
        "payload": {"station": "Ground", "start_sec": 4, "end_sec": 6},
    }
    runtime.inject([event], expected_revision=5, idempotency_key="inject")
    runtime.submit(
        [downlink(scenario, id="b", duration=3, actor="b")],
        expected_revision=6,
        idempotency_key="b",
    )
    runtime.advance(3, expected_revision=7, idempotency_key="b-time")
    assert runtime.report()["metrics"]["on_time_bits"] == 2e6
    assert runtime._backend.mission.remaining["b"] == 1e6
    restored = MissionRuntime.from_checkpoint(
        runtime.checkpoint(), access=AccessPolicy(truth=True)
    )
    assert restored.checkpoint() == runtime.checkpoint()


def test_page_cursor_and_copies_and_restore_requires_fresh_instance(scenario):
    runtime = privileged(scenario)
    initial = runtime.checkpoint()
    runtime.advance(1, expected_revision=0, idempotency_key="one")
    runtime.advance(1, expected_revision=1, idempotency_key="two")
    first = runtime.iter_events(limit=1)
    assert first["next_cursor"] == 1 and first["has_more"]
    first["events"][0]["response"]["ok"] = False
    assert runtime.iter_events(limit=1)["events"][0]["response"]["ok"]
    with pytest.raises(RuntimeContractError, match="fresh"):
        runtime.restore(initial, expected_revision=2)
    target = privileged(scenario)
    target.restore(runtime.checkpoint(), expected_revision=0)
    assert target.report() == runtime.report()


@pytest.mark.parametrize("edit", ["state", "operation", "runtime", "extra"])
def test_replay_detects_forgery_even_when_resigned(scenario, edit):
    runtime = privileged(scenario)
    runtime.advance(1, expected_revision=0, idempotency_key="a")
    payload = runtime.checkpoint()
    if edit == "state":
        payload["report"]["time_sec"] = 9
    elif edit == "operation":
        payload["operations"][0]["next_observation"]["time_sec"] = 9
    elif edit == "runtime":
        payload["runtime"]["python"] = "changed"
    else:
        payload["unsupported_field"] = True
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    with pytest.raises(RuntimeContractError):
        MissionRuntime.from_checkpoint(payload)


def test_reference_actor_order_and_configuration_survive_canonical_json(spec):
    data = copy.deepcopy(spec)
    data["ephemerides"] = {a: data["ephemerides"][a] for a in ("b", "a")}
    scenario = reference_scenario(data, id="order", lineage="family")
    runtime = privileged(scenario)
    payload = json.loads(canonical_json(runtime.checkpoint()))
    restored = MissionRuntime.from_checkpoint(payload)
    assert restored.describe_capabilities()["actors"] == ["b", "a"]
    assert restored.observe() == runtime.observe()
    assert (
        payload["effective_scenario"]["definition"]["vehicles"]["a"][
            "min_processing_soc"
        ]
        == 0.1
    )


def test_transaction_failure_rewinds_whole_advance(scenario, monkeypatch):
    runtime = privileged(scenario)
    runtime.submit([downlink(scenario)], expected_revision=0, idempotency_key="s")
    before = runtime.checkpoint()
    original = runtime._backend.advance
    count = 0

    def fail(dt, active):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected")
        return original(dt, active)

    monkeypatch.setattr(runtime._backend, "advance", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.advance(5, expected_revision=1, idempotency_key="failure")
    assert runtime.checkpoint() == before
    monkeypatch.setattr(runtime._backend, "advance", original)
    runtime.advance(5, expected_revision=1, idempotency_key="failure")
    assert runtime.report()["metrics"][scenario["objective"]] == 2e6


@pytest.mark.parametrize("parts", [[5], [1, 1, 3]])
def test_control_coarse_fine_outcomes(scenario, parts):
    runtime = privileged(scenario)
    runtime.submit([downlink(scenario)], expected_revision=0, idempotency_key="s")
    for i, dt in enumerate(parts):
        runtime.advance(dt, expected_revision=i + 1, idempotency_key=f"a{i}")
    assert runtime.report()["metrics"][scenario["objective"]] == 2e6
    assert runtime.report()["commands"][0]["status"] == "completed"


def test_cli_validate_run_deleted_source_resume_replay(scenario, tmp_path, capsys):
    from space_env.cli import main

    source, plan, record, resume, out = [
        tmp_path / name
        for name in (
            "scenario.json",
            "plan.json",
            "run.json",
            "resume.json",
            "continued.json",
        )
    ]
    source.write_text(json.dumps(scenario))
    plan.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": "submit",
                        "commands": [downlink(scenario)],
                        "expected_revision": 0,
                        "idempotency_key": "s",
                    },
                    {
                        "op": "advance",
                        "dt_sec": 1,
                        "expected_revision": 1,
                        "idempotency_key": "a",
                    },
                ]
            }
        )
    )
    main(["mission-validate", str(source)])
    assert json.loads(capsys.readouterr().out)["ok"]
    main(["mission-run", str(source), str(plan), str(record)])
    assert json.loads(capsys.readouterr().out)["report"]["time_sec"] == 1
    source.unlink()
    plan.unlink()
    resume.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": "advance",
                        "dt_sec": 4,
                        "expected_revision": 2,
                        "idempotency_key": "continue",
                    }
                ]
            }
        )
    )
    main(["mission-resume", str(record), str(resume), str(out)])
    assert (
        json.loads(capsys.readouterr().out)["report"]["metrics"][scenario["objective"]]
        == 2e6
    )
    main(["mission-replay", str(out)])
    assert json.loads(capsys.readouterr().out)["ok"]


def test_schema_and_strict_profile_mismatch(spec):
    schema = scenario_json_schema()
    assert schema["additionalProperties"] is False
    scenario = reference_scenario(spec, id="x", lineage="family")
    scenario["physics_profile"] = "legacy-training/1"
    with pytest.raises(RuntimeContractError) as e:
        MissionRuntime(scenario)
    assert e.value.code == "invalid_scenario"


def test_concurrent_revision_admission_is_serialized(scenario):
    from concurrent.futures import ThreadPoolExecutor

    runtime = privileged(scenario)

    def attempt(key):
        try:
            return runtime.advance(1, expected_revision=0, idempotency_key=key)
        except RuntimeContractError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, ["one", "two"]))
    assert results.count("stale_revision") == 1
    assert runtime.revision == 1 and runtime.time_sec == 1


def test_operation_budget_and_invalid_time_leave_state_unchanged(scenario):
    data = copy.deepcopy(scenario)
    data["limits"] = {"max_operations": 1}
    runtime = privileged(data)
    for value in [True, float("nan"), -1]:
        with pytest.raises(RuntimeContractError):
            runtime.advance(value, expected_revision=0, idempotency_key="invalid")
        assert runtime.revision == 0
    runtime.advance(1, expected_revision=0, idempotency_key="one")
    before = runtime.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        runtime.advance(1, expected_revision=1, idempotency_key="two")
    assert error.value.code == "operation_budget"
    assert runtime.checkpoint() == before


def test_checkpoint_restores_dag_cpu_booking_energy_and_thermal(spec, tmp_path):
    data = reference_fixtures._thermal_spec(reference_fixtures._dag_spec(spec))
    data["vehicles"]["a"]["memory_bits"] = 20e6
    scenario = reference_scenario(data, id="coupled", lineage="runtime-coupled")
    runtime = privileged(scenario)
    runtime.inject(
        [
            {
                "id": "book",
                "type": "terminal_booking_added",
                "time_sec": 0,
                "source": "operator",
                "reason": "test",
                "payload": {
                    "id": "pass",
                    "actor": "a",
                    "station": "Ground",
                    "start_sec": 0,
                    "end_sec": 20,
                },
            }
        ],
        expected_revision=0,
        idempotency_key="booking",
    )
    runtime.submit(
        [
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "processing",
                "duration_sec": 1,
                "product_id": "a",
            }
        ],
        expected_revision=1,
        idempotency_key="cpu",
    )
    runtime.advance(1, expected_revision=2, idempotency_key="cpu-time")
    assert runtime._backend.mission.retained_for("a") == ["derived"]
    runtime.submit(
        [
            {
                "command_id": "derive",
                "actor_id": "a",
                "kind": "processing",
                "duration_sec": 2,
                "product_id": "derived",
            }
        ],
        expected_revision=3,
        idempotency_key="derive",
    )
    runtime.advance(1, expected_revision=4, idempotency_key="partial")
    path = tmp_path / "coupled.json"
    runtime.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    for item in (runtime, restored):
        item.advance(1, expected_revision=5, idempotency_key="finish-cpu")
        item.submit(
            [downlink(scenario, duration=5)], expected_revision=6, idempotency_key="tx"
        )
        item.advance(5, expected_revision=7, idempotency_key="heat")
    assert restored.checkpoint() == runtime.checkpoint()
    assert restored._backend.mission.thermal_locked["a"]
    assert restored._backend.mission.processing_remaining["derived"] == 0
    assert restored._backend.mission.bookings["pass"].actor == "a"
    restricted = MissionRuntime.load(path)
    with pytest.raises(RuntimeContractError, match="truth"):
        restricted.checkpoint()
