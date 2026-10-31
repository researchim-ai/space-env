"""Spatial quality, incomplete evidence, physical SAR resources and replay."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from space_env.core.frames import geodetic_to_ecef
from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.area_coverage import AreaOfInterest
from space_env.reference.payload_quality import (
    OpticalModel,
    SarModel,
    ground_metrics,
    merge_evidence,
    resolution_metrics,
)
from space_env.reference.spacecraft_bus import pointing_quaternion
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.simulation.snapshot import digest
from tests import test_area_coverage as area
from tests import test_reference_lifecycle as life


def optical_model():
    return {
        "kind": "optical",
        "model_id": "synthetic-camera/1",
        "source_status": "synthetic",
        "source": "test assumptions",
        "instantaneous_fov_rad": 1e-5,
    }


def sar_model():
    return {
        "kind": "sar",
        "model_id": "synthetic-stripmap/1",
        "source_status": "synthetic",
        "source": "test assumptions",
        "bandwidth_hz": 1e8,
        "antenna_length_m": 4.0,
        "min_incidence_deg": 5.0,
        "max_incidence_deg": 60.0,
        "look_side": "either",
        "peak_rf_power_w": 20.0,
        "pulse_duty_fraction": 0.5,
        "amplifier_efficiency": 0.5,
        "rolling_window_sec": 10.0,
        "max_on_sec": 4.0,
    }


def mask(
    aoi,
    id="clear",
    *,
    state="clear",
    start=0.0,
    end=20.0,
    available=0.0,
    bounds=(-1e6, -1e6, 1e6, 1e6),
):
    aoi = AreaOfInterest.model_validate(aoi)
    return {
        "id": id,
        "aoi_id": aoi.id,
        "aoi_sha256": digest(aoi.model_dump(mode="json")),
        "bounds_m": list(bounds),
        "state": state,
        "valid_from_sec": start,
        "valid_until_sec": end,
        "available_from_sec": available,
        "source_status": "synthetic",
        "source": "scripted spatial mask",
    }


@pytest.fixture(scope="module")
def optical():
    data = copy.deepcopy(area.scenario.__wrapped__())
    spec = data["definition"]["lifecycle"]
    target = data["definition"]["acquisition_targets"][0]
    target["lat_deg"] *= -1
    target["lon_deg"] = (target["lon_deg"] + 360) % 360 - 180
    initial = data["definition"]["orbits"]["a"]["initial"]
    initial["r_m"] = (-np.asarray(initial["r_m"])).tolist()
    initial["v_m_s"] = (-np.asarray(initial["v_m_s"])).tolist()
    for station in data["definition"]["operations"]["stations"]:
        station.update(lat_deg=target["lat_deg"], lon_deg=target["lon_deg"])
    spec["areas_of_interest"] = [
        area.rectangle(target["lon_deg"], target["lat_deg"], 0.005)
    ]
    template = life.runtime(data)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        pointing_quaternion(
            template._backend.mission.ephemerides["a"],
            AcquisitionTarget.model_validate(target),
            0.0,
        ).tolist()
    )
    spec["payloads"]["a"]["imaging_model"] = optical_model()
    spec["requests"][0]["quality"] = {
        "kind": "optical",
        "min_sun_elevation_deg": 5.0,
        "max_gsd_m": 100.0,
    }
    spec["cloud_masks"] = [mask(spec["areas_of_interest"][0])]
    return data


@pytest.fixture(scope="module")
def sar():
    data = copy.deepcopy(area.scenario.__wrapped__())
    template = life.runtime(data)
    target = data["definition"]["acquisition_targets"][0]
    target["lat_deg"] += 1.0
    aoi = area.rectangle(target["lon_deg"], target["lat_deg"], 0.005)
    spec = data["definition"]["lifecycle"]
    spec["areas_of_interest"] = [aoi]
    spec["payloads"]["a"]["imaging_model"] = sar_model()
    spec["requests"][0]["quality"] = {
        "kind": "sar",
        "max_ground_range_resolution_m": 100.0,
        "max_azimuth_resolution_m": 10.0,
    }
    q = pointing_quaternion(
        template._backend.mission.ephemerides["a"],
        AcquisitionTarget.model_validate(target),
        0.0,
    )
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        q.tolist()
    )
    return data


def quality(model):
    return area.report(model)["payload_quality"]


def test_clear_optical_area_is_usable_only_after_receipt(optical):
    model = life.runtime(optical)
    assert "products.payload_quality" in model.describe_capabilities()["supported"]
    assert quality(model)["requests"]["request"]["usable_fraction"] == 0
    life.acquire(model)
    raw = area.report(model)["artifacts"]["raw"]
    assert raw["quality_evidence"]
    assert all(row["cloud_state"] == "clear" for row in raw["quality_evidence"])
    assert not quality(model)["requests"]["request"]["delivered_and_fulfilled"]
    life.transform(model)
    life.submit(
        model, "downlink", 1.0, "radio", product_id="final", destination_id="Ground"
    )
    life.advance(model, 1.0, "receipt")
    assert area.report(model)["metrics"]["fulfilled_requests"] == 1
    assert quality(model)["requests"]["request"]["usable_fraction"] >= 0.95


@pytest.mark.parametrize(
    "state", ["cloud", "unknown", "missing", "stale", "unavailable"]
)
def test_cloud_and_unknown_data_never_become_clear_credit(optical, state):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    if state == "missing":
        spec["cloud_masks"] = []
    elif state == "stale":
        spec["cloud_masks"][0]["valid_until_sec"] = 0.1
        spec["cloud_masks"][0]["state"] = "cloud"
    elif state == "unavailable":
        spec["cloud_masks"][0]["available_from_sec"] = 5.0
    else:
        spec["cloud_masks"][0]["state"] = state
    model = life.runtime(data)
    area.deliver(model)
    result = area.report(model)
    assert result["artifacts"]["final"]["status"] == "delivered"
    assert result["requests"]["request"]["reason"] == "quality_requirement"
    assert quality(model)["requests"]["request"]["usable_fraction"] == 0


def test_spatial_cloud_mask_cannot_be_replaced_by_target_center(optical):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    aoi = spec["areas_of_interest"][0]
    spec["cloud_masks"] = [mask(aoi, bounds=(-1e6, -1e6, 50.0, 1e6))]
    model = life.runtime(data)
    area.deliver(model)
    fraction = quality(model)["requests"]["request"]["usable_fraction"]
    assert 0.4 < fraction < 0.7
    assert area.report(model)["metrics"]["fulfilled_requests"] == 0


@pytest.mark.parametrize("criterion", ["night", "resolution", "applicability"])
def test_optical_geometry_and_resolution_change_usable_area(optical, criterion):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    if criterion == "night":
        spec["requests"][0]["quality"]["min_sun_elevation_deg"] = 90.0
    elif criterion == "resolution":
        spec["requests"][0]["quality"]["max_gsd_m"] = 0.01
    else:
        spec["payloads"]["a"]["imaging_model"]["max_incidence_deg"] = 0.0001
    model = life.runtime(data)
    area.deliver(model)
    assert area.report(model)["metrics"]["fulfilled_requests"] == 0
    assert quality(model)["requests"]["request"]["usable_fraction"] == 0


def test_time_release_and_forecast_masks_have_explicit_semantics(optical):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    aoi = spec["areas_of_interest"][0]
    spec["cloud_masks"] = [
        mask(aoi, "cloud", state="cloud", end=1.0),
        mask(aoi, "clear-later", start=1.0),
    ]
    spec["cloud_masks"][1].update(
        source_status="forecast", allow_forecast=True, available_from_sec=1.0
    )
    model = life.runtime(data)
    view = model.observe("a")["actors"]["a"]["lifecycle"]["payload_quality"]
    assert {m["id"] for m in view["cloud_masks"]} == {"cloud"}
    life.acquire(model)
    raw = area.report(model)["artifacts"]["raw"]
    assert any(
        row["time_sec"] < 1 and row["cloud_state"] == "cloud"
        for row in raw["quality_evidence"]
    )
    assert any(
        row["time_sec"] >= 1 and row["cloud_state"] == "clear"
        for row in raw["quality_evidence"]
    )
    life.transform(model)
    assert quality(model)["requests"]["request"]["usable_fraction"] >= 0.95


def test_analytic_resolution_equations_and_geometric_local_normal(sar):
    optical = OpticalModel.model_validate(optical_model())
    radar = SarModel.model_validate(sar_model())
    metrics = {"incidence_deg": 30.0, "slant_range_m": 800000.0}
    assert resolution_metrics(optical, metrics)["gsd_m"] == pytest.approx(
        8 / math.cos(math.pi / 6)
    )
    result = resolution_metrics(radar, metrics)
    assert result["ground_range_resolution_m"] == pytest.approx(299792458 / 1e8)
    assert result["azimuth_resolution_m"] == 2.0
    model = life.runtime(sar)
    provider = model._backend.mission.ephemerides["a"]
    target = model._backend.acquisition_targets["site"]
    value = ground_metrics(provider, 0.0, target.lon_deg, target.lat_deg)
    point = geodetic_to_ecef(target.lat_deg, target.lon_deg, 0.0)
    normal = np.array(
        [
            math.cos(math.radians(target.lat_deg))
            * math.cos(math.radians(target.lon_deg)),
            math.cos(math.radians(target.lat_deg))
            * math.sin(math.radians(target.lon_deg)),
            math.sin(math.radians(target.lat_deg)),
        ]
    )
    line = provider.context.matrix(provider.epoch) @ provider.state(0)[:3] - point
    expected = math.degrees(
        math.atan2(np.linalg.norm(np.cross(normal, line)), normal @ line)
    )
    assert value["incidence_deg"] == pytest.approx(expected, abs=1e-10)


def test_sar_uses_dc_power_once_and_rf_heat_balance_is_separate(sar):
    model = life.runtime(sar)
    initial = model._backend.mission.energy_wh["a"]
    life.acquire(model)
    result = quality(model)
    intervals = result["sar_intervals"]["a"]
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval["dc_input_j"] == 40.0
    assert interval["radiated_rf_j"] == 20.0
    assert interval["amplifier_heat_j"] == 20.0
    assert (
        interval["dc_input_j"]
        == interval["radiated_rf_j"] + interval["amplifier_heat_j"]
    )
    charges = area.report(model)["charges"]
    payload_wh = sum(row["energy_wh"] for row in charges if row["kind"] == "imaging")
    assert payload_wh == pytest.approx((10 + 20) * 2 / 3600)
    bus = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]
    assert bus["states"]["a"]["payload_heat_adjustment_j"] == pytest.approx(-12.0)
    assert abs(bus["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-8
    assert initial - model._backend.mission.energy_wh["a"] >= payload_wh / 0.9


@pytest.mark.parametrize("cause", ["incidence", "side", "capacity", "peak", "duty"])
def test_sar_physical_limits_block_imaging_without_fabricated_raw(sar, cause):
    data = copy.deepcopy(sar)
    radar = data["definition"]["lifecycle"]["payloads"]["a"]["imaging_model"]
    if cause == "incidence":
        radar.update(min_incidence_deg=50.0, max_incidence_deg=60.0)
    elif cause == "side":
        template = life.runtime(data)
        provider = template._backend.mission.ephemerides["a"]
        target = template._backend.acquisition_targets["site"]
        side = ground_metrics(provider, 0.0, target.lon_deg, target.lat_deg)[
            "look_side"
        ]
        radar["look_side"] = "right" if side == "left" else "left"
    elif cause == "peak":
        data["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 40.0
    elif cause == "capacity":
        data["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 15.0
    else:
        radar["max_on_sec"] = 0.75
    model = life.runtime(data)
    life.acquire(model)
    result = area.report(model)
    assert not result["artifacts"]
    assert (
        result["requests"]["request"]["reason"]
        == {
            "incidence": "sar_incidence",
            "side": "sar_look_side",
            "capacity": "bus_power_capacity",
            "peak": "bus_power_capacity",
            "duty": "sar_duty",
        }[cause]
    )
    if cause == "duty":
        assert quality(model)["sar_intervals"]["a"][0]["end_sec"] == pytest.approx(0.75)


def test_sar_works_without_cloud_masks_and_qualifies_delivered_resolution(sar):
    model = life.runtime(sar)
    area.deliver(model)
    assert area.report(model)["metrics"]["fulfilled_requests"] == 1
    assert quality(model)["requests"]["request"]["usable_fraction"] >= 0.95
    assert all(
        row["cloud_state"] == "not_applicable"
        for row in area.report(model)["artifacts"]["raw"]["quality_evidence"]
    )


def test_sar_duty_continuous_rolling_window_does_not_reset_on_command_boundaries(sar):
    model = life.runtime(sar)
    engine = model._backend.lifecycle.quality
    engine.record_sar("a", 0.0, 4.0)
    assert engine.duty_boundary("a", 4.0, 5.0) == 4.0
    assert engine.duty_boundary("a", 10.0, 14.0) == 14.0
    engine.models["a"] = engine.models["a"].model_copy(update={"max_on_sec": 10.0})
    assert engine.duty_boundary("a", 4.0, 25.0) == 25.0


@pytest.mark.parametrize("budget", ["records", "cells", "points"])
def test_quality_budget_rollback_is_atomic(optical, budget):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    spec["quality_sampling"] = (
        {"max_evidence_records": 1}
        if budget == "records"
        else {"max_cells_per_sample": 1}
        if budget == "cells"
        else {"max_points_per_region": 5}
    )
    model = life.runtime(data)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="quality .*budget"):
        life.advance(model, 2.0, "over-budget")
    assert model.checkpoint() == before
    assert model._backend.bus is model._backend.mission.bus


@pytest.mark.parametrize("which", ["optical", "sar"])
def test_quality_and_rf_history_checkpoint_preserve_evidence(
    which, optical, sar, tmp_path
):
    model = life.runtime(optical if which == "optical" else sar)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    life.advance(model, 0.5, "prefix")
    model.save(tmp_path / "quality.json")
    restored = MissionRuntime.load(
        tmp_path / "quality.json", access=AccessPolicy(truth=True)
    )
    for item in (model, restored):
        life.advance(item, 1.5, "finish")
    assert model.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize(
    "invalid",
    [
        "overlap",
        "hash",
        "time",
        "forecast",
        "observed-future",
        "kind",
        "area",
        "fields",
        "duty",
    ],
)
def test_invalid_quality_scenarios_have_no_fallback(optical, invalid):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    if invalid == "overlap":
        spec["cloud_masks"].append({**spec["cloud_masks"][0], "id": "other"})
    elif invalid == "hash":
        spec["cloud_masks"][0]["aoi_sha256"] = "0" * 64
    elif invalid == "time":
        spec["cloud_masks"][0]["valid_until_sec"] = 10000.0
    elif invalid == "forecast":
        spec["cloud_masks"][0]["source_status"] = "forecast"
    elif invalid == "observed-future":
        spec["cloud_masks"][0].update(
            source_status="observed", valid_from_sec=1.0, available_from_sec=0.0
        )
    elif invalid == "kind":
        spec["payloads"]["a"]["imaging_model"] = sar_model()
    elif invalid == "area":
        del spec["requests"][0]["coverage"]
    elif invalid == "fields":
        spec["requests"][0]["quality"]["max_ground_range_resolution_m"] = 10.0
    else:
        spec["payloads"]["a"]["imaging_model"] = {**sar_model(), "max_on_sec": 11.0}
    with pytest.raises(ValueError):
        life.runtime(data)


def test_fan_in_out_reevaluates_each_request_and_deduplicates_evidence(optical):
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2")
    spec["requests"][0]["quality"]["max_gsd_m"] = 0.01
    spec["requests"].append(second)
    spec["jobs"][0]["inputs"] = ["raw", "raw-2"]
    spec["jobs"][0]["outputs"].append({"id": "copy", "size_bits": 500000})
    model = life.runtime(data)
    life.acquire(model)
    life.submit(model, "imaging", 2.0, "second-image", request_id="second")
    life.advance(model, 2.0, "second-finish")
    life.transform(model)
    products = area.report(model)["artifacts"]
    assert products["final"]["quality_evidence"] == products["copy"]["quality_evidence"]
    merged = merge_evidence([products["final"], products["copy"]], 20000)
    assert len(merged) == len(products["final"]["quality_evidence"])
    life.submit(
        model, "downlink", 1.0, "radio", product_id="final", destination_id="Ground"
    )
    life.advance(model, 1.0, "receipt")
    requests = area.report(model)["requests"]
    assert requests["request"]["reason"] == "quality_requirement"
    assert requests["second"]["status"] == "completed"
    assert area.report(model)["metrics"]["fulfilled_requests"] == 1
    corrupt = copy.deepcopy(products["final"])
    corrupt["quality_evidence"][0]["cloud_state"] = "corrupt"
    with pytest.raises(ValueError, match="incompatible quality"):
        merge_evidence([products["final"], corrupt], 20000)


def test_passive_optical_night_is_outside_applicability_even_with_clear_mask():
    data = copy.deepcopy(area.scenario.__wrapped__())
    spec = data["definition"]["lifecycle"]
    spec["payloads"]["a"]["imaging_model"] = optical_model()
    spec["requests"][0]["quality"] = {
        "kind": "optical",
        "min_sun_elevation_deg": 0.0,
        "max_gsd_m": 100.0,
    }
    spec["cloud_masks"] = [mask(spec["areas_of_interest"][0])]
    model = life.runtime(data)
    area.deliver(model)
    assert quality(model)["requests"]["request"]["usable_fraction"] == 0
    assert area.report(model)["metrics"]["fulfilled_requests"] == 0


def test_cancellation_and_safe_recovery_do_not_erase_sar_duty_history(sar):
    model = life.runtime(sar)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    life.advance(model, 0.5, "partial")
    before = copy.deepcopy(quality(model)["sar_intervals"])
    model.cancel("image", expected_revision=model.revision, idempotency_key="cancel")
    for kind in ("spacecraft_safe_mode", "spacecraft_recover"):
        model.inject(
            [life.event(model, kind, {"actor": "a"}, kind)],
            expected_revision=model.revision,
            idempotency_key=kind,
        )
    assert quality(model)["sar_intervals"] == before
    assert before["a"][0]["dc_input_j"] == 10.0


def test_quality_observation_does_not_leak_peer_masks(optical):
    model = life.runtime(optical)
    own = model.observe("a")["actors"]["a"]["lifecycle"]["payload_quality"]
    peer = model.observe("b")["actors"]["b"]["lifecycle"]["payload_quality"]
    assert len(own["cloud_masks"]) == 1
    assert peer["cloud_masks"] == [] and peer["model"] is None


def test_quality_transform_budget_failure_keeps_spent_cpu_and_inputs(optical):
    probe = life.runtime(optical)
    life.acquire(probe)
    count = len(area.report(probe)["artifacts"]["raw"]["quality_evidence"])
    data = copy.deepcopy(optical)
    spec = data["definition"]["lifecycle"]
    spec["quality_sampling"] = {"max_evidence_records": count}
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2")
    spec["requests"].append(second)
    spec["jobs"][0]["inputs"] = ["raw", "raw-2"]
    model = life.runtime(data)
    life.acquire(model)
    life.submit(model, "imaging", 2.0, "second-image", request_id="second")
    life.advance(model, 2.0, "second-finish")
    life.transform(model)
    result = area.report(model)
    assert "final" not in result["artifacts"]
    assert result["artifacts"]["raw"]["status"] == "resident"
    assert result["jobs"]["compress"]["status"] == "failed"
    assert result["jobs"]["compress"]["lost_work_sec"] == 2.0
