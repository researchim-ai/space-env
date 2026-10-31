"""Exposure-time revisit/stereo proofs, quality intersection and delivered bundles."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from space_env.core.constants import R_EARTH_EQUATOR
from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.area_coverage import AreaOfInterest, coverage_from_samples
from space_env.reference.observations import (
    ObservationRequirement,
    ObservationSettings,
    capture_observation,
    evaluate_observations,
    merge_observations,
    stereo_angles,
)
from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.reference.time import Epoch
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.simulation.snapshot import digest
from tests import test_area_coverage as area
from tests import test_payload_quality as payload
from tests import test_reference_lifecycle as life


def stereo(**extra):
    return {
        "min_convergence_deg": 0.1,
        "max_convergence_deg": 10.0,
        "max_interval_sec": 10.0,
        **extra,
    }


def revisit(**extra):
    return {"min_interval_sec": 0.5, "max_interval_sec": 10.0, **extra}


def evaluate(product, aoi, requirement, **settings):
    return evaluate_observations(
        ObservationRequirement.model_validate(requirement),
        product,
        aoi,
        None,
        0.95,
        ObservationSettings(**settings),
    )


def analytic_product(
    times=(0.0, 3.0),
    raw_ids=("first", "second"),
    actors=("a", "a"),
    offsets=(-50000.0, 50000.0),
):
    aoi = AreaOfInterest.model_validate(area.rectangle(half=0.0001))
    samples = [area.sample(aoi.shell, t) for t in times]
    observations = []
    for sample, raw_id, actor, offset in zip(samples, raw_ids, actors, offsets):
        provider = SimpleNamespace(
            epoch=Epoch.utc("2024-01-01T00:00:00Z"),
            sha256="0" * 64,
            context=SimpleNamespace(matrix=lambda _: np.eye(3)),
            state=lambda t, offset=offset: np.array(
                [R_EARTH_EQUATOR + 500000, offset, 0.0, 0.0, 7500.0, 0.0]
            ),
        )
        observations.append(
            capture_observation(
                provider, [1.0, 0.0, 0.0, 0.0], aoi, sample, raw_id, actor
            )
        )
    coverage = coverage_from_samples(aoi, samples)
    return {"observations": observations, "area_coverage": {aoi.id: coverage}}, aoi


@pytest.fixture(scope="module")
def scenario():
    data = copy.deepcopy(payload.optical.__wrapped__())
    spec = data["definition"]["lifecycle"]
    spec["requests"][0]["observations"] = {"revisit": revisit(), "stereo": stereo()}
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2")
    second.pop("observations")
    spec["requests"].append(second)
    spec["jobs"][0]["inputs"] = ["raw", "raw-2"]
    return data


@pytest.fixture(scope="module")
def fleet():
    data = copy.deepcopy(payload.optical.__wrapped__())
    template = life.runtime(data)
    state = template._backend.mission.ephemerides["a"].state(3.0)
    initial = data["definition"]["orbits"]["b"]["initial"]
    initial.update(r_m=state[:3].tolist(), v_m_s=state[3:].tolist())
    spec = data["definition"]["lifecycle"]
    spec["jobs"] = []
    first = spec["requests"][0]
    first["delivery_product_id"] = "raw"
    first["observations"] = {
        "stereo": stereo(different_actors=True, max_interval_sec=2.0),
        "support_product_ids": ["raw-b"],
    }
    second = copy.deepcopy(first)
    second.update(
        id="second",
        eligible_actors=["b"],
        raw_product_id="raw-b",
        delivery_product_id="raw-b",
    )
    second.pop("observations")
    spec["requests"].append(second)
    spec["payloads"]["b"] = copy.deepcopy(spec["payloads"]["a"])
    data["definition"]["acquisition_sensors"]["b"] = copy.deepcopy(
        data["definition"]["acquisition_sensors"]["a"]
    )
    target = AcquisitionTarget.model_validate(
        data["definition"]["acquisition_targets"][0]
    )
    template = life.runtime(data)
    data["definition"]["spacecraft_bus"]["b"]["attitude"]["initial_quaternion_wxyz"] = (
        pointing_quaternion(
            template._backend.mission.ephemerides["b"], target, 0.0
        ).tolist()
    )
    return data


def result(model, rid="request"):
    return area.report(model)["observation_requirements"][rid]


def acquire_two(model):
    life.acquire(model)
    life.submit(model, "imaging", 2.0, "second-image", request_id="second")
    life.advance(model, 2.0, "second-finish")


def send(model, pid="final", actor="a", key="radio"):
    model.submit(
        [
            {
                "command_id": key,
                "actor_id": actor,
                "kind": "downlink",
                "duration_sec": 1.0,
                "product_id": pid,
                "destination_id": "Ground",
            }
        ],
        expected_revision=model.revision,
        idempotency_key=key,
    )
    life.advance(model, 1.0, key + "-time")


def acquire_fleet(model):
    model.submit(
        [
            {
                "command_id": "image-a",
                "actor_id": "a",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "request",
            },
            {
                "command_id": "image-b",
                "actor_id": "b",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "second",
            },
        ],
        expected_revision=model.revision,
        idempotency_key="images",
    )
    life.advance(model, 2.0, "acquire-fleet")


def test_independent_convergence_formula_and_baseline():
    product, aoi = analytic_product()
    expected = math.degrees(2 * math.atan(50000 / 500000))
    angle = stereo_angles(
        product["observations"][0]["position_itrs_m"],
        product["observations"][1]["position_itrs_m"],
        [[R_EARTH_EQUATOR, 0.0, 0.0]],
    )[0]
    assert angle == pytest.approx(expected, abs=1e-12)
    actual = evaluate(
        product,
        aoi,
        {
            "stereo": stereo(
                min_convergence_deg=11.0,
                max_convergence_deg=12.0,
                min_baseline_m=99999.0,
                max_baseline_m=100001.0,
            )
        },
    )
    assert actual["satisfied"]
    witness = actual["stereo"]["witnesses"][0]
    assert witness["baseline_itrs_m"] == 100000.0
    assert (
        witness["min_sampled_angle_deg"] - 1e-12
        <= expected
        <= witness["max_sampled_angle_deg"] + 1e-12
    )


@pytest.mark.parametrize(
    "violation", ["angle", "baseline", "time", "actor", "zero-baseline"]
)
def test_stereo_rejects_incompatible_pairs(violation):
    product, aoi = analytic_product()
    config = stereo(min_convergence_deg=10.0, max_convergence_deg=12.0)
    if violation == "angle":
        config.update(min_convergence_deg=20.0, max_convergence_deg=30.0)
    elif violation == "baseline":
        config["max_baseline_m"] = 100.0
    elif violation == "time":
        config["max_interval_sec"] = 1.0
    elif violation == "actor":
        config["different_actors"] = True
    else:
        product, aoi = analytic_product(offsets=(0.0, 0.0))
    assert not evaluate(product, aoi, {"stereo": config})["satisfied"]


def test_one_raw_with_many_samples_or_dag_copies_is_never_a_second_visit():
    product, aoi = analytic_product(raw_ids=("one", "one"))
    copied = {**product, "observations": merge_observations([product, product], 100)}
    answer = evaluate(copied, aoi, {"revisit": revisit(), "stereo": stereo()})
    assert answer["distinct_acquisitions"] == 1
    assert not answer["satisfied"]


def test_revisit_three_acquisitions_and_interval_actor_policy():
    product, aoi = analytic_product(
        times=(0.0, 3.0, 6.0),
        raw_ids=("a", "b", "c"),
        actors=("one", "two", "three"),
        offsets=(-100.0, 0.0, 100.0),
    )
    req = {
        "revisit": revisit(
            min_observations=3,
            min_interval_sec=3.0,
            max_interval_sec=3.0,
            different_actors=True,
        )
    }
    assert evaluate(product, aoi, req)["satisfied"]
    req["revisit"]["max_interval_sec"] = 2.9
    req["revisit"]["min_interval_sec"] = 1.0
    assert not evaluate(product, aoi, req)["satisfied"]


def test_partial_spatial_overlap_does_not_count_full_area():
    product, aoi = analytic_product()
    obs = product["observations"][1]
    polygon = shapely.from_geojson(json.dumps(obs["geometry"]))
    obs["geometry"] = json.loads(
        shapely.to_geojson(polygon.intersection(shapely.box(-1e6, -1e6, 0.0, 1e6)))
    )
    obs["id"] = digest({k: v for k, v in obs.items() if k != "id"})
    value = evaluate(product, aoi, {"revisit": revisit()})
    assert 0.45 < value["joint_fraction"] < 0.55 and not value["satisfied"]


def test_runtime_same_actor_fan_in_needs_delivery_and_actual_exposure_times(scenario):
    model = life.runtime(scenario)
    acquire_two(model)
    raw = area.report(model)["artifacts"]["raw"]
    assert raw["observations"][0]["time_sec"] == 0.0
    assert raw["observations"][-1]["time_sec"] < raw["created_sec"]
    assert all(
        row["observation_id"] in {r["id"] for r in raw["observations"]}
        for row in raw["quality_evidence"]
    )
    life.transform(model)
    assert result(model)["satisfied"] and not result(model)["full_delivery"]
    send(model)
    assert result(model)["delivered_and_fulfilled"]
    assert area.report(model)["metrics"]["fulfilled_requests"] == 2
    for witness in result(model)["revisit"]["witnesses"]:
        assert len(set(witness["raw_ids"])) == 2


def test_constellation_ground_bundle_waits_for_both_full_acks(fleet):
    model = life.runtime(fleet)
    acquire_fleet(model)
    assert result(model)["satisfied"] and not result(model)["full_delivery"]
    send(model, "raw")
    assert area.report(model)["metrics"]["fulfilled_requests"] == 0
    send(model, "raw-b", "b", "radio-b")
    report = area.report(model)
    assert result(model)["delivered_and_fulfilled"]
    assert report["metrics"]["fulfilled_requests"] == 2
    assert report["artifacts"]["raw-b"]["request_ids"] == ["second"]
    assert sum(row["bits"] for row in report["acknowledgements"]) == 2000000
    assert all(row["balance_error_bits"] == 0 for row in report["ledger"].values())


def test_quality_must_hold_in_both_observations_not_just_one_clear_union(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    aoi = spec["areas_of_interest"][0]
    spec["cloud_masks"] = [
        payload.mask(aoi, "clear-first", end=2.0),
        payload.mask(aoi, "cloud-second", start=2.0, state="cloud"),
    ]
    model = life.runtime(data)
    acquire_two(model)
    life.transform(model)
    send(model)
    assert (
        area.report(model)["payload_quality"]["requests"]["request"]["usable_fraction"]
        >= 0.95
    )
    assert result(model)["joint_fraction"] == 0
    assert (
        area.report(model)["requests"]["request"]["reason"] == "observation_requirement"
    )


@pytest.mark.parametrize("kind", ["frames", "pairs", "chains", "cells", "points"])
def test_evaluation_budgets_fail_closed_without_partial_credit(kind):
    product, aoi = analytic_product(
        times=(0.0, 3.0, 6.0),
        raw_ids=("a", "b", "c"),
        actors=("a", "b", "c"),
        offsets=(-1000.0, 0.0, 1000.0),
    )
    args = {
        "frames": {"max_frames_per_evaluation": 2},
        "pairs": {"max_pair_checks": 1},
        "chains": {"max_chain_checks": 1},
        "cells": {"max_cell_checks": 1},
        "points": {"max_points_per_region": 5},
    }[kind]
    answer = evaluate(
        product,
        aoi,
        {"revisit": revisit(), "stereo": stereo(min_convergence_deg=0.01)},
        **args,
    )
    assert answer["status"] == "budget_exceeded"
    assert not answer["satisfied"] and answer["joint_fraction"] == 0


def test_mid_acquisition_bundle_restore_and_partial_radio(fleet, tmp_path):
    model = life.runtime(fleet)
    model.submit(
        [
            {
                "command_id": "image-a",
                "actor_id": "a",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "request",
            },
            {
                "command_id": "image-b",
                "actor_id": "b",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "second",
            },
        ],
        expected_revision=0,
        idempotency_key="images",
    )
    life.advance(model, 0.5, "prefix")
    model.save(tmp_path / "mid.json")
    other = MissionRuntime.load(tmp_path / "mid.json", access=AccessPolicy(truth=True))
    for item in (model, other):
        life.advance(item, 1.5, "finish")
        send(item, "raw")
        send(item, "raw-b", "b", "radio-b")
    assert model.checkpoint() == other.checkpoint()


@pytest.mark.parametrize(
    "invalid",
    [
        "no-area",
        "empty",
        "time",
        "angle",
        "support-self",
        "support-unknown",
        "support-duplicate",
        "destination",
    ],
)
def test_invalid_requirements_and_bundles_reject_without_fallback(scenario, invalid):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    request = spec["requests"][0]
    if invalid == "no-area":
        del request["coverage"]
    elif invalid == "empty":
        request["observations"] = {}
    elif invalid == "time":
        request["observations"]["revisit"]["max_interval_sec"] = 0.1
    elif invalid == "angle":
        request["observations"]["stereo"]["max_convergence_deg"] = 0.01
    elif invalid == "support-self":
        request["observations"]["support_product_ids"] = ["final"]
    elif invalid == "support-unknown":
        request["observations"]["support_product_ids"] = ["unmanaged"]
    elif invalid == "support-duplicate":
        request["observations"]["support_product_ids"] = ["raw-2", "raw-2"]
    else:
        request["observations"]["support_product_ids"] = ["raw-2"]
        spec["requests"][1]["delivery_product_id"] = "raw-2"
        spec["requests"][1]["destination_id"] = "Other"
        data["definition"]["operations"]["stations"].append(
            {**data["definition"]["operations"]["stations"][0], "name": "Other"}
        )
    with pytest.raises(ValueError):
        life.runtime(data)


def test_capture_budget_rollback_preserves_bus_alias_and_observations(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["observation_settings"] = {
        "max_records_per_product": 3
    }
    model = life.runtime(data)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="observation record budget"):
        life.advance(model, 2.0, "over-budget")
    assert model.checkpoint() == before
    assert model._backend.bus is model._backend.mission.bus


def test_clear_halves_from_different_times_do_not_form_two_good_views(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    aoi = spec["areas_of_interest"][0]
    left, right = (-1e6, -1e6, 0.0, 1e6), (0.0, -1e6, 1e6, 1e6)
    spec["cloud_masks"] = [
        payload.mask(aoi, "first-left", bounds=left, end=2.0),
        payload.mask(aoi, "first-right", bounds=right, state="cloud", end=2.0),
        payload.mask(aoi, "second-left", bounds=left, state="cloud", start=2.0),
        payload.mask(aoi, "second-right", bounds=right, start=2.0),
    ]
    model = life.runtime(data)
    acquire_two(model)
    life.transform(model)
    send(model)
    assert (
        area.report(model)["payload_quality"]["requests"]["request"]["usable_fraction"]
        >= 0.95
    )
    assert result(model)["joint_fraction"] < 1e-8
    assert not result(model)["delivered_and_fulfilled"]


def test_partial_support_ack_never_completes_ground_bundle(fleet):
    model = life.runtime(fleet)
    acquire_fleet(model)
    send(model, "raw")
    model.submit(
        [
            {
                "command_id": "radio-b",
                "actor_id": "b",
                "kind": "downlink",
                "duration_sec": 1.0,
                "product_id": "raw-b",
                "destination_id": "Ground",
            }
        ],
        expected_revision=model.revision,
        idempotency_key="support",
    )
    life.advance(model, 0.25, "partial-support")
    assert not result(model)["full_delivery"]
    assert area.report(model)["metrics"]["fulfilled_requests"] == 0
    life.advance(model, 0.75, "complete-support")
    assert result(model)["delivered_and_fulfilled"]


@pytest.mark.parametrize("cause", ["cancel-before-capture", "expire-after-capture"])
def test_unreachable_support_causally_fails_dependent_obligation(fleet, cause):
    data = copy.deepcopy(fleet)
    if cause == "expire-after-capture":
        data["definition"]["lifecycle"]["requests"][1]["raw_ttl_sec"] = 0.25
    model = life.runtime(data)
    if cause == "cancel-before-capture":
        model.inject(
            [life.event(model, "request_cancelled", {"request_id": "second"})],
            expected_revision=model.revision,
            idempotency_key="cancel-demand",
        )
    else:
        acquire_fleet(model)
        life.advance(model, 0.5, "expire-support")
    assert area.report(model)["requests"]["request"]["status"] == "failed"
    assert not result(model)["delivered_and_fulfilled"]


def test_undelivered_global_history_and_copies_cannot_fill_primary_product(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    spec["requests"][0]["delivery_product_id"] = "raw"
    spec["requests"][1]["delivery_product_id"] = "raw-2"
    spec["jobs"] = []
    model = life.runtime(data)
    acquire_two(model)
    send(model, "raw")
    assert result(model)["distinct_acquisitions"] == 1
    assert (
        area.report(model)["requests"]["request"]["reason"] == "observation_requirement"
    )


def test_content_fingerprint_and_conflicting_pose_are_rejected():
    product, _ = analytic_product()
    corrupt = copy.deepcopy(product)
    corrupt["observations"][0]["position_itrs_m"][0] += 1.0
    with pytest.raises(ValueError, match="content fingerprint"):
        merge_observations([corrupt], 100)
    row = corrupt["observations"][0]
    row["id"] = digest({k: v for k, v in row.items() if k != "id"})
    with pytest.raises(ValueError, match="conflicting pose"):
        merge_observations([product, corrupt], 100)


def test_cached_report_is_a_copy_and_does_not_change_execution(scenario):
    model = life.runtime(scenario)
    acquire_two(model)
    life.transform(model)
    first = result(model)
    before = model.checkpoint()
    first["stereo"]["witnesses"].clear()
    first["satisfied"] = False
    assert result(model)["satisfied"]
    assert result(model)["stereo"]["witnesses"]
    assert model.checkpoint() == before


def test_evaluation_budget_failure_does_not_undo_radio_or_fabricate_credit(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["observation_settings"] = {"max_pair_checks": 1}
    model = life.runtime(data)
    acquire_two(model)
    life.transform(model)
    send(model)
    assert area.report(model)["artifacts"]["final"]["status"] == "delivered"
    assert result(model)["status"] == "budget_exceeded"
    assert area.report(model)["requests"]["request"]["reason"] == "observation_budget"


def test_ground_bundle_combination_budget_is_not_silently_truncated(fleet):
    data = copy.deepcopy(fleet)
    data["definition"]["lifecycle"]["observation_settings"] = {
        "max_records_per_product": 12
    }
    model = life.runtime(data)
    acquire_fleet(model)
    send(model, "raw")
    send(model, "raw-b", "b", "radio-b")
    assert result(model)["full_delivery"]
    assert result(model)["status"] == "budget_exceeded"
    assert not result(model)["delivered_and_fulfilled"]


def test_late_support_does_not_revive_expired_obligation(fleet):
    data = copy.deepcopy(fleet)
    data["definition"]["lifecycle"]["requests"][0]["deadline_sec"] = 4.0
    model = life.runtime(data)
    acquire_fleet(model)
    send(model, "raw")
    life.advance(model, 2.0, "miss-deadline")
    send(model, "raw-b", "b", "radio-b")
    assert result(model)["full_delivery"]
    assert area.report(model)["requests"]["request"]["status"] == "expired"
    assert not result(model)["delivered_and_fulfilled"]
