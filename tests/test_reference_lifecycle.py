"""B03 request-to-delivery contracts, physical resource failures and provenance."""

from __future__ import annotations

import copy

import pytest

from space_env.reference.lifecycle import MaterializationError
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    lifecycle_scenario,
)
from tests import test_reference_geometry_events as geometry_fixtures


@pytest.fixture(scope="module")
def scenario():
    base = geometry_fixtures.scenario.__wrapped__()["definition"]
    definition = copy.deepcopy(base)
    definition["operations"]["products"] = []
    definition["lifecycle"] = {
        "payloads": {"a": {"power_w": 10.0}, "b": {"power_w": 10.0}},
        "requests": [
            {
                "id": "request",
                "target_id": "site",
                "eligible_actors": ["a"],
                "deadline_sec": 20.0,
                "acquisition_sec": 2.0,
                "raw_product_id": "raw",
                "raw_size_bits": 1000000,
                "delivery_product_id": "final",
                "destination_id": "Ground",
            }
        ],
        "jobs": [
            {
                "id": "compress",
                "inputs": ["raw"],
                "outputs": [{"id": "final", "size_bits": 500000}],
                "work_sec": 2.0,
                "workspace_bits": 100000,
                "power_w": 5.0,
            }
        ],
    }
    return lifecycle_scenario(definition, id="eo", lineage="B03")


def runtime(scenario):
    return MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))


def submit(model, kind, duration, key, **fields):
    command = {
        "command_id": key,
        "actor_id": "a",
        "kind": kind,
        "duration_sec": duration,
        **fields,
    }
    return model.submit(
        [command], expected_revision=model.revision, idempotency_key=key
    )


def advance(model, seconds, key):
    return model.advance(seconds, expected_revision=model.revision, idempotency_key=key)


def lifecycle(model):
    return model._backend.report()["lifecycle"]


def acquire(model):
    submit(model, "imaging", 2.0, "image", request_id="request")
    advance(model, 2.0, "acquire")


def transform(model):
    submit(model, "transform", 2.0, "cpu", job_id="compress")
    advance(model, 2.0, "compute")


def test_request_acquisition_transform_delivery_without_preloaded_products(
    scenario, tmp_path
):
    model = runtime(scenario)
    assert not model._backend.mission.products and not lifecycle(model)["artifacts"]
    assert model.describe_capabilities()["frame"] == "GCRS"
    assert model.checkpoint()["runtime"]["frame_contract"] == "GCRS/ITRS"
    before = lifecycle(model)["metrics"]
    model.forecast_acquisition("a", "site")
    assert lifecycle(model)["metrics"] == before
    acquire(model)
    report = lifecycle(model)
    assert set(report["artifacts"]) == {"raw"}
    assert report["artifacts"]["raw"]["source"] == "acquired"
    assert report["metrics"]["fulfilled_requests"] == 0
    assert model._backend.mission.consumed_wh["a"] > 0.005
    transform(model)
    report = lifecycle(model)
    assert report["artifacts"]["raw"]["status"] == "transformed_in"
    assert report["artifacts"]["final"]["size_bits"] == 500000
    assert report["artifacts"]["final"]["request_ids"] == ["request"]
    assert report["metrics"]["fulfilled_requests"] == 0
    submit(model, "downlink", 1.0, "radio", destination_id="Ground", product_id="final")
    advance(model, 1.0, "deliver")
    report = lifecycle(model)
    assert report["requests"]["request"]["status"] == "completed"
    assert report["requests"]["request"]["delivered_sec"] == 4.5
    assert report["metrics"] == {"fulfilled_requests": 1, "total_requests": 1}
    assert report["ledger"]["a"]["balance_error_bits"] == 0
    path = tmp_path / "eo.json"
    model.save(path)
    assert (
        MissionRuntime.load(path, access=AccessPolicy(truth=True)).checkpoint()
        == model.checkpoint()
    )


def test_partial_acquisition_checkpoint_and_cancel_do_not_create_data(scenario):
    model = runtime(scenario)
    submit(model, "imaging", 2.0, "image", request_id="request")
    advance(model, 1.0, "half")
    assert not lifecycle(model)["artifacts"]
    restored = MissionRuntime.from_checkpoint(
        model.checkpoint(), access=AccessPolicy(truth=True)
    )
    for item in (model, restored):
        item.cancel("image", expected_revision=item.revision, idempotency_key="cancel")
    assert model.checkpoint() == restored.checkpoint()
    assert not lifecycle(model)["artifacts"]
    assert lifecycle(model)["requests"]["request"]["status"] == "failed"


@pytest.mark.parametrize("failure", ["geometry", "power", "memory"])
def test_impossible_acquisition_never_materializes_and_stays_in_denominator(
    scenario, failure
):
    data = copy.deepcopy(scenario)
    if failure == "geometry":
        data["definition"]["acquisition_sensors"]["a"]["max_range_m"] = 1.0
    elif failure == "power":
        data["definition"]["operations"]["vehicles"]["a"]["initial_soc"] = 0.0
    else:
        data["definition"]["operations"]["vehicles"]["a"]["memory_bits"] = 500000.0
    model = runtime(data)
    acquire(model)
    report = lifecycle(model)
    assert report["requests"]["request"]["reason"] == failure
    assert not report["artifacts"] and report["metrics"] == {
        "fulfilled_requests": 0,
        "total_requests": 1,
    }


def test_mid_transform_workspace_pause_resume_and_switching_cost(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["jobs"][0]["switching_sec"] = 1.0
    model = runtime(data)
    acquire(model)
    submit(model, "transform", 0.5, "setup-half", job_id="compress")
    advance(model, 0.5, "setup-time")
    row = lifecycle(model)["jobs"]["compress"]
    assert (
        row["reserved"] and row["work_sec"] == 0 and row["setup_remaining_sec"] == 0.5
    )
    submit(model, "transform", 1.0, "work-half", job_id="compress")
    advance(model, 1.0, "work-time")
    row = lifecycle(model)["jobs"]["compress"]
    assert row["remaining_sec"] == 1.5 and row["switching_sec"] == 1.0
    restored = MissionRuntime.from_checkpoint(
        model.checkpoint(), access=AccessPolicy(truth=True)
    )
    for item in (model, restored):
        submit(item, "transform", 1.5, "finish", job_id="compress")
        advance(item, 1.5, "finish-time")
    assert restored.checkpoint() == model.checkpoint()
    assert lifecycle(model)["jobs"]["compress"]["status"] == "completed"


def test_raw_cannot_be_transmitted_or_injected_as_a_final_product(scenario):
    model = runtime(scenario)
    acquire(model)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="deliverable"):
        submit(
            model,
            "downlink",
            1.0,
            "raw-radio",
            destination_id="Ground",
            product_id="raw",
        )
    assert model.checkpoint() == before
    submit(model, "downlink", 1.0, "auto-radio", destination_id="Ground")
    advance(model, 1.0, "no-raw-radio")
    assert not model._backend.mission.receipts
    with pytest.raises(RuntimeContractError, match="managed"):
        model.inject(
            [
                event(
                    model,
                    "onboard_product_added",
                    {
                        "id": "final",
                        "actor": "a",
                        "size_bits": 500000,
                        "deadline_sec": 20.0,
                    },
                )
            ],
            expected_revision=model.revision,
            idempotency_key="inject-final",
        )


def event(model, kind, payload, id="event"):
    return {
        "id": id,
        "type": kind,
        "time_sec": model.time_sec,
        "source": "customer",
        "reason": "test",
        "payload": payload,
    }


def test_release_cancel_priority_and_new_demand_events_are_causal(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["requests"][0]["release_sec"] = 2.0
    model = runtime(data)
    assert not model.observe("a")["actors"]["a"]["lifecycle"]["requests"]
    with pytest.raises(RuntimeContractError, match="released"):
        submit(model, "imaging", 2.0, "early", request_id="request")
    advance(model, 2.0, "release")
    model.inject(
        [
            event(
                model,
                "request_priority_changed",
                {"request_id": "request", "priority": 5},
            )
        ],
        expected_revision=model.revision,
        idempotency_key="priority",
    )
    assert lifecycle(model)["requests"]["request"]["priority"] == 5
    request = copy.deepcopy(data["definition"]["lifecycle"]["requests"][0])
    request.update(
        id="next",
        raw_product_id="next-raw",
        delivery_product_id="next-raw",
        release_sec=2.0,
    )
    model.inject(
        [event(model, "request_added", {"request": request, "jobs": []}, "new")],
        expected_revision=model.revision,
        idempotency_key="new",
    )
    model.inject(
        [event(model, "request_cancelled", {"request_id": "next"}, "cancel")],
        expected_revision=model.revision,
        idempotency_key="cancel-next",
    )
    assert lifecycle(model)["metrics"]["total_requests"] == 2
    assert lifecycle(model)["requests"]["next"]["status"] == "cancelled"
    assert (
        MissionRuntime.from_checkpoint(
            model.checkpoint(), access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )


def test_expiry_and_pinned_input_policy(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["requests"][0]["raw_ttl_sec"] = 1.0
    model = runtime(data)
    acquire(model)
    submit(model, "transform", 0.5, "partial", job_id="compress")
    advance(model, 0.5, "partial-time")
    advance(model, 1.0, "pinned-time")
    assert lifecycle(model)["artifacts"]["raw"]["status"] == "resident"
    with pytest.raises(RuntimeContractError, match="pinned"):
        model.inject(
            [event(model, "artifact_discarded", {"product_id": "raw"})],
            expected_revision=model.revision,
            idempotency_key="discard",
        )
    model.inject(
        [event(model, "job_cancelled", {"job_id": "compress"}, "job-cancel")],
        expected_revision=model.revision,
        idempotency_key="cancel-job",
    )
    assert lifecycle(model)["artifacts"]["raw"]["status"] == "expired"
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == pytest.approx(
        0, abs=1e-8
    )


def test_declared_materialization_failure_spends_work_without_partial_outputs(
    scenario, monkeypatch
):
    model = runtime(scenario)
    acquire(model)

    class Failing:
        version = "metadata-surrogate/1"

        def materialize(self, inputs, output):
            raise MaterializationError("declared plugin failure")

    monkeypatch.setattr(model._backend.lifecycle, "transformer", Failing())
    transform(model)
    report = lifecycle(model)
    assert (
        report["jobs"]["compress"]["status"] == "failed"
        and "final" not in report["artifacts"]
    )
    assert report["jobs"]["compress"]["work_sec"] == 2.0
    assert report["ledger"]["a"]["balance_error_bits"] == 0


def test_unexpected_failure_rolls_back_entire_advance_including_energy_and_artifacts(
    scenario, monkeypatch
):
    model = runtime(scenario)
    acquire(model)
    submit(model, "transform", 2.0, "cpu", job_id="compress")
    before = model.checkpoint()

    def fail(*args):
        raise RuntimeError("unexpected materializer bug")

    monkeypatch.setattr(model._backend.lifecycle.transformer, "materialize", fail)
    with pytest.raises(RuntimeError, match="unexpected"):
        advance(model, 2.0, "compute")
    assert model.checkpoint() == before


def test_deadline_failure_and_explicit_final_discard_policy(scenario):
    model = runtime(scenario)
    acquire(model)
    transform(model)
    with pytest.raises(RuntimeContractError, match="policy"):
        model.inject(
            [event(model, "artifact_discarded", {"product_id": "final"})],
            expected_revision=model.revision,
            idempotency_key="discard",
        )
    advance(model, 16.0, "deadline")
    assert lifecycle(model)["requests"]["request"]["status"] == "expired"
    assert lifecycle(model)["metrics"]["fulfilled_requests"] == 0


def test_profile_dag_and_provenance_guards(scenario):
    from space_env.runtime import dynamic_reference_scenario

    with pytest.raises(RuntimeContractError, match="explicit"):
        MissionRuntime(
            dynamic_reference_scenario(
                scenario["definition"], id="wrong", lineage="B03"
            )
        )
    cyclic = copy.deepcopy(scenario)
    cyclic["definition"]["lifecycle"]["jobs"][0]["inputs"] = ["final"]
    with pytest.raises(RuntimeContractError, match="cycle"):
        runtime(cyclic)


def test_fanout_consumes_shared_input_only_after_last_consumer(scenario):
    data = copy.deepcopy(scenario)
    jobs = data["definition"]["lifecycle"]["jobs"]
    jobs.append(
        {
            "id": "preview",
            "inputs": ["raw"],
            "outputs": [
                {"id": "preview-out", "size_bits": 100000, "stage": "intermediate"}
            ],
            "work_sec": 1.0,
            "workspace_bits": 100000,
        }
    )
    model = runtime(data)
    acquire(model)
    transform(model)
    assert lifecycle(model)["artifacts"]["raw"]["status"] == "resident"
    submit(model, "transform", 1.0, "preview", job_id="preview")
    advance(model, 1.0, "preview-time")
    report = lifecycle(model)
    assert report["artifacts"]["raw"]["status"] == "transformed_in"
    assert report["ledger"]["a"]["transformed_in"] == 1000000
    assert report["ledger"]["a"]["transformed_out"] == 600000
    assert report["ledger"]["a"]["balance_error_bits"] == 0


def test_fanin_multiple_outputs_and_request_lineage(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="second-raw")
    spec["requests"].append(second)
    spec["jobs"][0].update(
        inputs=["raw", "second-raw"],
        outputs=[
            {"id": "final", "size_bits": 500000},
            {"id": "aux", "size_bits": 100000, "stage": "intermediate"},
        ],
    )
    model = runtime(data)
    acquire(model)
    submit(model, "imaging", 2.0, "image-second", request_id="second")
    advance(model, 2.0, "acquire-second")
    transform(model)
    report = lifecycle(model)
    assert report["artifacts"]["final"]["request_ids"] == ["request", "second"]
    assert report["ledger"]["a"]["transformed_in"] == 2000000
    submit(model, "downlink", 1.0, "radio", destination_id="Ground", product_id="final")
    advance(model, 1.0, "delivery")
    assert lifecycle(model)["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == pytest.approx(
        0, abs=1e-8
    )


def test_memory_deadlock_is_reported_and_workspace_reservation_blocks_injection(
    scenario,
):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["memory_bits"] = 1500000.0
    model = runtime(data)
    acquire(model)
    submit(model, "transform", 1.0, "blocked", job_id="compress")
    advance(model, 1.0, "blocked-time")
    assert lifecycle(model)["jobs"]["compress"]["reason"] == "memory"
    assert "final" not in lifecycle(model)["artifacts"]
    data["definition"]["operations"]["vehicles"]["a"]["memory_bits"] = 2000000.0
    model = runtime(data)
    acquire(model)
    submit(model, "transform", 0.5, "partial", job_id="compress")
    advance(model, 0.5, "partial-time")
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="reserved memory"):
        model.inject(
            [
                event(
                    model,
                    "onboard_product_added",
                    {
                        "id": "external",
                        "actor": "a",
                        "size_bits": 500000,
                        "ready_sec": model.time_sec,
                        "deadline_sec": 20.0,
                    },
                )
            ],
            expected_revision=model.revision,
            idempotency_key="injection",
        )
    assert model.checkpoint() == before


def test_partial_radio_and_expired_raw_do_not_fulfill_request(scenario):
    model = runtime(scenario)
    acquire(model)
    transform(model)
    submit(
        model, "downlink", 0.1, "partial", destination_id="Ground", product_id="final"
    )
    advance(model, 0.1, "partial-time")
    assert lifecycle(model)["metrics"]["fulfilled_requests"] == 0
    assert lifecycle(model)["ledger"]["a"]["transmitted"] == pytest.approx(100000)
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == pytest.approx(
        0, abs=1e-8
    )
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["requests"][0]["raw_ttl_sec"] = 1.0
    model = runtime(data)
    acquire(model)
    advance(model, 1.0, "expire")
    assert lifecycle(model)["artifacts"]["raw"]["status"] == "expired"
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == pytest.approx(
        0, abs=1e-8
    )


def test_batch_event_failure_rolls_back_prior_priority_update(scenario):
    model = runtime(scenario)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError):
        model.inject(
            [
                event(
                    model,
                    "request_priority_changed",
                    {"request_id": "request", "priority": 9},
                    "priority",
                ),
                event(model, "request_cancelled", {"request_id": "missing"}, "bad"),
            ],
            expected_revision=0,
            idempotency_key="batch",
        )
    assert model.checkpoint() == before


def test_duplicate_commands_and_coarse_fine_execution_do_not_duplicate_outputs(
    scenario,
):
    coarse, fine = runtime(scenario), runtime(scenario)
    for model in (coarse, fine):
        submit(model, "imaging", 2.0, "image", request_id="request")
    advance(coarse, 2.0, "whole")
    for i in range(4):
        advance(fine, 0.5, f"slice{i}")
    for key in ("artifacts", "ledger", "requests", "charges"):
        assert lifecycle(coarse)[key] == lifecycle(fine)[key]
    before = coarse.checkpoint()
    submit(coarse, "imaging", 2.0, "image", request_id="request")
    assert coarse.checkpoint() == before


def test_two_satellites_acquire_transform_and_compete_for_one_station(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    data["definition"]["acquisition_sensors"]["b"] = data["definition"][
        "acquisition_sensors"
    ]["a"]
    request = copy.deepcopy(spec["requests"][0])
    request.update(
        id="second",
        eligible_actors=["b"],
        raw_product_id="b-raw",
        delivery_product_id="b-final",
    )
    spec["requests"].append(request)
    spec["jobs"].append(
        {
            "id": "b-cpu",
            "inputs": ["b-raw"],
            "outputs": [{"id": "b-final", "size_bits": 500000}],
            "work_sec": 2.0,
        }
    )
    model = runtime(data)
    for kind, extra, duration in (
        ("imaging", [{"request_id": "request"}, {"request_id": "second"}], 2.0),
        ("transform", [{"job_id": "compress"}, {"job_id": "b-cpu"}], 2.0),
        (
            "downlink",
            [
                {"product_id": "final", "destination_id": "Ground"},
                {"product_id": "b-final", "destination_id": "Ground"},
            ],
            1.0,
        ),
    ):
        model.submit(
            [
                {
                    "command_id": f"{kind}-{actor}",
                    "actor_id": actor,
                    "kind": kind,
                    "duration_sec": duration,
                    **fields,
                }
                for actor, fields in zip(("a", "b"), extra)
            ],
            expected_revision=model.revision,
            idempotency_key=kind,
        )
        advance(model, duration, f"{kind}-time")
    assert lifecycle(model)["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}
    assert all(
        abs(r["balance_error_bits"]) < 1e-8 for r in lifecycle(model)["ledger"].values()
    )


def test_request_cancellation_stops_acquisition_and_multioutput_failure_is_atomic(
    scenario, monkeypatch
):
    model = runtime(scenario)
    submit(model, "imaging", 2.0, "image", request_id="request")
    advance(model, 0.5, "partial")
    model.inject(
        [event(model, "request_cancelled", {"request_id": "request"})],
        expected_revision=model.revision,
        idempotency_key="cancel-request",
    )
    advance(model, 1.5, "finish")
    assert (
        not lifecycle(model)["artifacts"]
        and lifecycle(model)["requests"]["request"]["status"] == "cancelled"
    )
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["jobs"][0]["outputs"].append(
        {"id": "second", "size_bits": 100000}
    )
    model = runtime(data)
    acquire(model)
    original = model._backend.lifecycle.transformer.materialize

    def second_fails(inputs, output):
        if output.id == "second":
            raise MaterializationError("second output failed")
        return original(inputs, output)

    monkeypatch.setattr(
        model._backend.lifecycle.transformer, "materialize", second_fails
    )
    transform(model)
    assert set(lifecycle(model)["artifacts"]) == {"raw"}
    assert lifecycle(model)["jobs"]["compress"]["status"] == "failed"


def test_zero_hidden_cpu_work_and_workspace_owner_guards(scenario):
    model = runtime(scenario)
    acquire(model)
    advance(model, 1.0, "idle")
    assert lifecycle(model)["jobs"]["compress"]["work_sec"] == 0
    with pytest.raises(RuntimeContractError):
        model.submit(
            [
                {
                    "command_id": "other",
                    "actor_id": "b",
                    "kind": "transform",
                    "duration_sec": 1.0,
                    "job_id": "compress",
                }
            ],
            expected_revision=model.revision,
            idempotency_key="other",
        )


def test_power_exhaustion_discards_partial_acquisition_without_raw(scenario):
    data = copy.deepcopy(scenario)
    v = data["definition"]["operations"]["vehicles"]["a"]
    v.update(battery_wh=1.0, initial_soc=0.105, base_power_w=10.0)
    data["definition"]["lifecycle"]["payloads"]["a"]["power_w"] = 20.0
    model = runtime(data)
    acquire(model)
    assert not lifecycle(model)["artifacts"]
    assert lifecycle(model)["operations"]["image"]["progress_sec"] == pytest.approx(0.6)
    assert lifecycle(model)["requests"]["request"]["reason"] == "power"


def test_acknowledgement_is_full_receipt_only_and_no_double_count_for_shared_final(
    scenario,
):
    model = runtime(scenario)
    acquire(model)
    transform(model)
    submit(
        model, "downlink", 0.1, "partial", destination_id="Ground", product_id="final"
    )
    advance(model, 0.1, "partial-time")
    assert lifecycle(model)["acknowledgements"] == []
    assert lifecycle(model)["delivery_receipts"][0]["request_ids"] == ["request"]
    submit(model, "downlink", 1.0, "rest", destination_id="Ground", product_id="final")
    advance(model, 1.0, "rest-time")
    assert len(lifecycle(model)["acknowledgements"]) == 1
    assert lifecycle(model)["ledger"]["a"]["acked"] == 500000
    assert lifecycle(model)["acknowledgements"][0]["fulfilled_request_ids"] == [
        "request"
    ]


def test_horizon_expiration_is_housekeeping_not_invalid_late_operator_event(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["requests"][0]["raw_ttl_sec"] = 28.0
    model = runtime(data)
    acquire(model)
    advance(model, 28.0, "horizon")
    assert model.time_sec == 30
    assert lifecycle(model)["artifacts"]["raw"]["status"] == "expired"
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == 0
    assert (
        MissionRuntime.from_checkpoint(
            model.checkpoint(), access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )


def test_external_unmanaged_injection_is_separate_from_acquisition_and_reward(scenario):
    model = runtime(scenario)
    model.inject(
        [
            event(
                model,
                "onboard_product_added",
                {
                    "id": "external",
                    "actor": "a",
                    "size_bits": 100000,
                    "deadline_sec": 20.0,
                },
            )
        ],
        expected_revision=0,
        idempotency_key="external",
    )
    submit(
        model, "downlink", 0.5, "radio", destination_id="Ground", product_id="external"
    )
    advance(model, 0.5, "time")
    report = lifecycle(model)
    assert (
        report["ledger"]["a"]["injected"]
        == report["ledger"]["a"]["transmitted"]
        == 100000
    )
    assert report["ledger"]["a"]["balance_error_bits"] == 0
    assert (
        report["metrics"]["fulfilled_requests"] == 0
        and report["acknowledgements"] == []
    )


def test_geometry_feature_transform_is_separate_from_declared_resource_cost(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["jobs"][0]["transformer"] = "geometry-features/1"
    model = runtime(data)
    acquire(model)
    transform(model)
    product = lifecycle(model)["artifacts"]["final"]
    assert product["representation"] == "json-geometry-features/1"
    assert product["payload"][0]["acquisition"]["validated_dwell_sec"] == 2.0
    assert product["payload"][0]["acquisition"]["geometry_end"]["target_id"] == "site"
    assert product["size_bits"] == 500000 and product["size_model"].startswith(
        "declared resource bits"
    )
    assert (
        MissionRuntime.from_checkpoint(
            model.checkpoint(), access=AccessPolicy(truth=True)
        ).checkpoint()
        == model.checkpoint()
    )


def test_trusted_transformer_registry_requires_matching_provenance_on_restore(
    scenario, monkeypatch
):
    from space_env.reference import lifecycle as module
    from space_env.simulation.snapshot import digest

    monkeypatch.setattr(
        module, "_TRANSFORMER_FACTORIES", dict(module._TRANSFORMER_FACTORIES)
    )

    class Custom:
        version = "custom-test/1"
        source_sha256 = "a" * 64

        def materialize(self, inputs, output):
            return {
                "representation": "custom-json/1",
                "payload": {"count": len(inputs)},
                "content_sha256": digest(inputs),
            }

    module.register_product_transformer(Custom)
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["jobs"][0]["transformer"] = Custom.version
    data["definition"]["lifecycle"]["transformer_manifests"] = {
        Custom.version: Custom.source_sha256
    }
    model = runtime(data)
    acquire(model)
    transform(model)
    assert lifecycle(model)["artifacts"]["final"]["payload"] == {"count": 1}
    checkpoint = model.checkpoint()
    assert (
        MissionRuntime.from_checkpoint(
            checkpoint, access=AccessPolicy(truth=True)
        ).checkpoint()
        == checkpoint
    )
    del module._TRANSFORMER_FACTORIES[Custom.version]
    with pytest.raises(RuntimeContractError, match="registered"):
        MissionRuntime.from_checkpoint(checkpoint)
    with pytest.raises(RuntimeContractError, match="fingerprint"):
        bad = copy.deepcopy(scenario)
        bad["definition"]["lifecycle"]["transformer_manifests"] = {
            "metadata-surrogate/1": "0" * 64
        }
        runtime(bad)


def test_early_completion_releases_control_window_without_double_reward(scenario):
    model = runtime(scenario)
    submit(model, "imaging", 4.0, "long-image", request_id="request")
    advance(model, 2.0, "early")
    assert model.report()["commands"][0]["status"] == "completed"
    transform(model)
    assert set(lifecycle(model)["artifacts"]) == {"raw", "final"}
    assert lifecycle(model)["metrics"]["fulfilled_requests"] == 0


def test_actor_observations_expose_only_eligible_local_job_definitions(scenario):
    model = runtime(scenario)
    actor_a = model.agent("a").observe()["actors"]["a"]["lifecycle"]
    actor_b = model.agent("b").observe()["actors"]["b"]["lifecycle"]
    assert actor_a["jobs"][0]["inputs"] == ["raw"]
    assert actor_a["jobs"][0]["outputs"][0]["size_bits"] == 500000
    assert not actor_b["requests"] and not actor_b["jobs"] and not actor_b["artifacts"]
    data = copy.deepcopy(scenario)
    data["definition"]["acquisition_sensors"]["b"] = data["definition"][
        "acquisition_sensors"
    ]["a"]
    data["definition"]["lifecycle"]["requests"][0]["eligible_actors"] = ["a", "b"]
    shared = runtime(data)
    assert shared.agent("b").observe()["actors"]["b"]["lifecycle"]["jobs"]
    acquire(shared)
    assert not shared.agent("b").observe()["actors"]["b"]["lifecycle"]["jobs"]


def test_external_injection_validates_before_reserved_memory_arithmetic(scenario):
    model = runtime(scenario)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError):
        model.inject(
            [
                event(
                    model,
                    "onboard_product_added",
                    {
                        "id": "external",
                        "actor": "a",
                        "size_bits": "invalid",
                        "deadline_sec": 20.0,
                    },
                )
            ],
            expected_revision=0,
            idempotency_key="bad",
        )
    assert model.checkpoint() == before
    model.inject(
        [
            event(
                model,
                "onboard_product_added",
                {
                    "id": "external",
                    "actor": "a",
                    "size_bits": "100000",
                    "deadline_sec": 20.0,
                },
            )
        ],
        expected_revision=0,
        idempotency_key="valid",
    )
    assert lifecycle(model)["ledger"]["a"]["injected"] == 100000
    assert lifecycle(model)["ledger"]["a"]["balance_error_bits"] == 0
