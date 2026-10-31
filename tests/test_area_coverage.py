"""B04-B1 polygon geometry, provenance and delivered coverage contracts."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("shapely")
pyproj = pytest.importorskip("pyproj")

from space_env.core.attitude import align_z
from space_env.core.constants import R_EARTH_EQUATOR, R_EARTH_POLAR
from space_env.reference.area_coverage import (
    AreaOfInterest,
    coverage_from_samples,
    domain,
    geometry_manifest,
    merge_coverage,
    ray_ellipsoid,
    rectangular_footprint,
)
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from tests import test_reference_lifecycle as life
from tests import test_spacecraft_bus as bus_fixtures


def rectangle(lon=0.0, lat=0.0, half=0.05, id="area"):
    def longitude(x):
        return (x + 180) % 360 - 180

    return {
        "id": id,
        "anchor_lon_deg": lon,
        "anchor_lat_deg": lat,
        "shell": [
            [longitude(lon - half), lat - half],
            [longitude(lon + half), lat - half],
            [longitude(lon + half), lat + half],
            [longitude(lon - half), lat + half],
            [longitude(lon - half), lat - half],
        ],
    }


def sample(shell, time=0.0):
    return {"status": "ok", "shell": shell, "time_sec": time}


@pytest.fixture(scope="module")
def scenario():
    data = copy.deepcopy(bus_fixtures.scenario.__wrapped__())
    definition = data["definition"]
    target = next(t for t in definition["acquisition_targets"] if t["id"] == "site")
    definition["lifecycle"]["areas_of_interest"] = [
        rectangle(target["lon_deg"], target["lat_deg"], 0.005)
    ]
    definition["lifecycle"]["requests"][0]["coverage"] = {
        "aoi_id": "area",
        "min_fraction": 0.95,
    }
    definition["lifecycle"]["payloads"]["a"]["footprint"] = {
        "half_angle_x_deg": 1.0,
        "half_angle_y_deg": 1.0,
    }
    return data


def report(model):
    return model.observe(view="truth")["backend"]["report"]["lifecycle"]


def deliver(model):
    life.acquire(model)
    life.transform(model)
    life.submit(
        model, "downlink", 1.0, "radio", product_id="final", destination_id="Ground"
    )
    life.advance(model, 1.0, "deliver")


@pytest.mark.parametrize(
    "lon,lat", [(0.0, 0.0), (179.99, 10.0), (-179.99, -30.0), (40.0, 85.0)]
)
def test_equal_area_matches_independent_geodesic_area(lon, lat):
    aoi = AreaOfInterest.model_validate(rectangle(lon, lat))
    polygon, _ = domain(aoi)
    coords = np.asarray(aoi.shell)
    geodesic_area = abs(
        pyproj.Geod(ellps="WGS84").polygon_area_perimeter(coords[:, 0], coords[:, 1])[0]
    )
    assert polygon.area == pytest.approx(geodesic_area, rel=2e-7)
    covered = coverage_from_samples(aoi, [sample(aoi.shell)])
    # FOV edges are sampled straight LAEA chords, AOI edges geodesically densified.
    assert covered["fraction"] == pytest.approx(1.0, abs=2e-5)


def test_polar_polygon_and_hole_are_preserved():
    geod = pyproj.Geod(ellps="WGS84")
    ring = [geod.fwd(0.0, 90.0, bearing, 10000)[:2] for bearing in range(0, 360, 30)]
    ring.append(ring[0])
    hole = [geod.fwd(0.0, 90.0, bearing, 2000)[:2] for bearing in range(0, 360, 30)]
    hole.append(hole[0])
    aoi = AreaOfInterest(
        id="polar", anchor_lon_deg=0.0, anchor_lat_deg=90.0, shell=ring, holes=[hole]
    )
    polygon, _ = domain(aoi)
    outer = AreaOfInterest(
        id="outer", anchor_lon_deg=0.0, anchor_lat_deg=90.0, shell=ring
    )
    assert polygon.area / domain(outer)[0].area == pytest.approx(0.96, abs=2e-6)
    assert coverage_from_samples(aoi, [sample(hole)])["covered_area_m2"] < 1.0


@pytest.mark.parametrize(
    "change", ["open", "bowtie", "outside", "large", "duplicate", "hole-outside"]
)
def test_invalid_aoi_is_rejected_without_repair(change):
    data = rectangle()
    if change == "open":
        data["shell"] = data["shell"][:-1]
    elif change == "bowtie":
        data["shell"][1], data["shell"][2] = data["shell"][2], data["shell"][1]
    elif change == "outside":
        data["shell"][1][0] = 181.0
    elif change == "large":
        data = rectangle(half=20.0)
    elif change == "duplicate":
        data["shell"].insert(1, data["shell"][0])
    else:
        data["holes"] = [rectangle(1.0, 1.0)["shell"]]
    with pytest.raises(ValueError):
        domain(AreaOfInterest.model_validate(data))


def test_union_overlap_disjoint_and_holes_do_not_double_count():
    aoi = AreaOfInterest.model_validate(rectangle(half=0.1))
    left = rectangle(-0.05, 0.0, 0.04)["shell"]
    right = rectangle(0.05, 0.0, 0.04)["shell"]
    a = coverage_from_samples(aoi, [sample(left)])
    b = coverage_from_samples(aoi, [sample(right)])
    both = merge_coverage([a, b])["area"]
    assert both["covered_area_m2"] == pytest.approx(
        a["covered_area_m2"] + b["covered_area_m2"], rel=1e-12
    )
    assert merge_coverage([a, a])["area"]["covered_area_m2"] == pytest.approx(
        a["covered_area_m2"]
    )
    overlap = coverage_from_samples(aoi, [sample(rectangle(-0.03, 0.0, 0.04)["shell"])])
    assert (
        merge_coverage([a, overlap])["area"]["covered_area_m2"]
        < a["covered_area_m2"] + overlap["covered_area_m2"]
    )
    outside = coverage_from_samples(aoi, [sample(rectangle(1.0, 1.0)["shell"])])
    assert outside["covered_area_m2"] == 0


def test_ray_hit_matches_independent_equatorial_solution_and_rejects_misses():
    radius, altitude = R_EARTH_EQUATOR, 500000.0
    r = radius + altitude
    alpha = math.radians(20.0)
    hit = ray_ellipsoid([r, 0, 0], [-math.cos(alpha), math.sin(alpha), 0])
    theta = math.asin(r / radius * math.sin(alpha)) - alpha
    np.testing.assert_allclose(
        hit, [radius * math.cos(theta), radius * math.sin(theta), 0], atol=1e-8
    )
    assert ray_ellipsoid([r, 0, 0], [1, 0, 0]) is None
    assert ray_ellipsoid([r, 0, 0], [0, 1, 0]) is None
    assert ray_ellipsoid([0, 0, 0], [1, 0, 0]) is None
    with pytest.raises(ValueError):
        ray_ellipsoid([r, 0, 0], [0, 0, 0])


def test_rectangular_camera_uses_actual_attitude_and_declares_limb_failure():
    provider = SimpleNamespace(
        state=lambda _: np.array([R_EARTH_EQUATOR + 500000, 0, 0, 0, 7500, 0]),
        context=SimpleNamespace(matrix=lambda _: np.eye(3)),
        epoch=SimpleNamespace(shifted=lambda t: t),
    )
    sensor = {"half_angle_x_deg": 1.0, "half_angle_y_deg": 2.0}
    result = rectangular_footprint(
        provider, align_z(np.array([-1.0, 0.0, 0.0])), sensor, 0.0
    )
    assert result["status"] == "ok" and len(result["shell"]) == 33
    assert np.ptp(np.asarray(result["shell"])[:, 0]) > 0
    away = rectangular_footprint(
        provider, align_z(np.array([1.0, 0.0, 0.0])), sensor, 0.0
    )
    assert away["reason"] == "footprint_horizon"
    polar = ray_ellipsoid([0, 0, R_EARTH_POLAR + 500000], [0, 0, -1])
    assert polar[2] == pytest.approx(R_EARTH_POLAR)


def test_full_area_pipeline_and_receipt_lineage(scenario):
    model = life.runtime(scenario)
    assert "products.area_coverage" in model.describe_capabilities()["supported"]
    assert report(model)["area_coverage"]["acquired"] == {}
    life.acquire(model)
    raw = report(model)["artifacts"]["raw"]
    assert raw["area_coverage"]["area"]["fraction"] >= 0.95
    assert report(model)["metrics"]["fulfilled_requests"] == 0
    assert report(model)["area_coverage"]["delivered"] == {}
    life.transform(model)
    final = report(model)["artifacts"]["final"]
    assert final["area_coverage"] == raw["area_coverage"]
    life.submit(
        model, "downlink", 1.0, "radio", product_id="final", destination_id="Ground"
    )
    life.advance(model, 1.0, "delivery")
    result = report(model)
    assert result["metrics"]["fulfilled_requests"] == 1
    assert result["area_coverage"]["delivered"]["area"]["fraction"] >= 0.95
    assert result["ledger"]["a"]["balance_error_bits"] == 0


def test_partial_coverage_does_not_fulfill_even_after_full_ack(scenario):
    data = copy.deepcopy(scenario)
    aoi = data["definition"]["lifecycle"]["areas_of_interest"][0]
    data["definition"]["lifecycle"]["areas_of_interest"] = [
        rectangle(aoi["anchor_lon_deg"], aoi["anchor_lat_deg"], 0.5)
    ]
    model = life.runtime(data)
    deliver(model)
    result = report(model)
    assert result["artifacts"]["final"]["status"] == "delivered"
    assert result["area_coverage"]["delivered"]["area"]["fraction"] < 0.95
    assert result["requests"]["request"]["reason"] == "coverage_requirement"
    assert result["metrics"]["fulfilled_requests"] == 0
    assert result["acknowledgements"][0]["fulfilled_request_ids"] == []


def test_mid_acquisition_restore_and_atomic_budget_rollback(scenario, tmp_path):
    model = life.runtime(scenario)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    life.advance(model, 0.5, "prefix")
    model.save(tmp_path / "mid.json")
    other = MissionRuntime.load(tmp_path / "mid.json", access=AccessPolicy(truth=True))
    for item in (model, other):
        life.advance(item, 1.5, "finish")
    assert model.checkpoint() == other.checkpoint()
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["payloads"]["a"]["footprint"][
        "max_acquisition_samples"
    ] = 3
    model = life.runtime(data)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="coverage sample budget"):
        life.advance(model, 2.0, "budget")
    assert model.checkpoint() == before
    assert model._backend.bus is model._backend.mission.bus


@pytest.mark.parametrize(
    "change",
    ["missing-aoi", "missing-camera", "duplicate-aoi", "ideal-attitude", "manifest"],
)
def test_incompatible_area_configuration_is_rejected(scenario, change):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    if change == "missing-aoi":
        spec["requests"][0]["coverage"]["aoi_id"] = "unknown"
    elif change == "missing-camera":
        del spec["payloads"]["a"]["footprint"]
    elif change == "duplicate-aoi":
        spec["areas_of_interest"] *= 2
    elif change == "ideal-attitude":
        data["backend"], data["physics_profile"] = (
            "reference-lifecycle",
            "reference-lifecycle/1",
        )
        data["fidelity"] = {
            "trajectory": "controlled-gcrs",
            "resources": "explicit-surrogate",
        }
        del data["definition"]["spacecraft_bus"]
        for vehicle in data["definition"]["operations"]["vehicles"].values():
            vehicle["solar_mode"] = "ideal_sun_tracking"
    else:
        spec["coverage_geometry_manifest"] = {"shapely": "wrong"}
    with pytest.raises(ValueError):
        life.runtime(data)


def test_duplicate_lineage_and_incompatible_projection_rejected():
    aoi = AreaOfInterest.model_validate(rectangle())
    row = coverage_from_samples(aoi, [sample(aoi.shell)])
    other = copy.deepcopy(row)
    other["aoi_sha256"] = "different"
    with pytest.raises(ValueError, match="incompatible AOI"):
        merge_coverage([row, other])
    assert set(geometry_manifest()) == {"pyproj", "proj", "shapely", "geos"}


def test_fan_in_out_preserves_union_unique_samples_and_revisit_history(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    second = copy.deepcopy(spec["requests"][0])
    second.update(id="second", raw_product_id="raw-2")
    spec["requests"].append(second)
    spec["jobs"][0]["inputs"] = ["raw", "raw-2"]
    spec["jobs"][0]["outputs"].append({"id": "unused-copy", "size_bits": 500000})
    model = life.runtime(data)
    life.acquire(model)
    life.submit(model, "imaging", 2.0, "second-image", request_id="second")
    life.advance(model, 2.0, "second-acquisition")
    life.transform(model)
    result = report(model)
    coverage = result["artifacts"]["final"]["area_coverage"]["area"]
    assert set(coverage["sources"]) == {"raw", "raw-2"}
    assert result["artifacts"]["unused-copy"]["area_coverage"]["area"] == coverage
    duplicate = merge_coverage([coverage, coverage])["area"]
    assert duplicate["sample_count"] == coverage["sample_count"]
    assert duplicate["covered_area_m2"] == pytest.approx(coverage["covered_area_m2"])
    history = result["area_coverage"]["revisits"]
    assert len(history) == 1 and history[0]["interval_sec"] == 2.0
    assert history[0]["overlap_area_m2"] > 0
    assert result["area_coverage"]["delivered"] == {}
    life.submit(
        model, "downlink", 1.0, "radio", product_id="final", destination_id="Ground"
    )
    life.advance(model, 1.0, "receipt")
    assert report(model)["metrics"] == {"fulfilled_requests": 2, "total_requests": 2}


def test_unrelated_delivery_cannot_satisfy_another_products_area_requirement(scenario):
    data = copy.deepcopy(scenario)
    spec = data["definition"]["lifecycle"]
    first = spec["requests"][0]
    first["delivery_product_id"] = "raw"
    first["coverage"]["min_fraction"] = 0.99
    spec["jobs"] = []
    aoi = spec["areas_of_interest"][0]
    spec["areas_of_interest"] = [
        rectangle(aoi["anchor_lon_deg"], aoi["anchor_lat_deg"], 0.5)
    ]
    second = copy.deepcopy(first)
    second.update(
        id="other", raw_product_id="other-raw", delivery_product_id="other-raw"
    )
    second["coverage"]["min_fraction"] = 0.001
    spec["requests"].append(second)
    model = life.runtime(data)
    life.acquire(model)
    life.submit(model, "imaging", 2.0, "second-image", request_id="other")
    life.advance(model, 2.0, "second-acquisition")
    life.submit(
        model,
        "downlink",
        1.0,
        "other-radio",
        product_id="other-raw",
        destination_id="Ground",
    )
    life.advance(model, 1.0, "other-receipt")
    result = report(model)
    assert result["requests"]["other"]["status"] == "completed"
    assert result["requests"]["request"]["status"] == "running"
    assert result["metrics"]["fulfilled_requests"] == 1


def test_actor_and_release_scoped_aoi_definitions(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["requests"][0]["release_sec"] = 1.0
    model = life.runtime(data)

    def local(actor):
        return model.observe(actor)["actors"][actor]["lifecycle"]

    assert local("a")["areas_of_interest"] == {}
    life.advance(model, 1.0, "release")
    assert set(local("a")["areas_of_interest"]) == {"area"}
    assert local("b")["areas_of_interest"] == {}
    assert local("b")["requests"] == []


def test_pointing_failure_and_cancellation_do_not_bank_area(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    model = life.runtime(data)
    life.acquire(model)
    assert report(model)["area_coverage"]["acquired"] == {}
    assert not report(model)["artifacts"]
    model = life.runtime(scenario)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    life.advance(model, 0.5, "partial")
    model.cancel("image", expected_revision=model.revision, idempotency_key="cancel")
    assert report(model)["area_coverage"]["acquired"] == {}


def test_missing_optional_geometry_dependency_has_explicit_error(monkeypatch):
    import builtins

    from space_env.reference.area_coverage import _libraries

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "shapely":
            raise ImportError("simulated optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match=r"space-env\[coverage\]"):
        _libraries()


def test_distant_projection_domain_is_rejected_explicitly():
    aoi = AreaOfInterest.model_validate(rectangle())
    with pytest.raises(ValueError, match="2000 km"):
        coverage_from_samples(aoi, [sample(rectangle(170.0, 0.0)["shell"])])


def test_densification_budget_is_checked_before_allocating_unbounded_points():
    data = rectangle(half=0.5)
    data["max_segment_m"] = 10.0
    with pytest.raises(ValueError, match="densification budget"):
        domain(AreaOfInterest.model_validate(data))


def test_limb_crossing_camera_does_not_fabricate_partial_footprint(scenario):
    data = copy.deepcopy(scenario)
    orbit = data["definition"]["orbits"]["a"]["initial"]
    orbit["r_m"] = (np.asarray(orbit["r_m"]) * 2).tolist()
    sensor = data["definition"]["lifecycle"]["payloads"]["a"]["footprint"]
    sensor.update(half_angle_x_deg=45.0, half_angle_y_deg=45.0)
    model = life.runtime(data)
    life.acquire(model)
    assert report(model)["requests"]["request"]["reason"] == "footprint_horizon"
    assert not report(model)["artifacts"]


def test_native_geometry_manifest_drift_prevents_checkpoint_restore(
    scenario, tmp_path, monkeypatch
):
    model = life.runtime(scenario)
    model.save(tmp_path / "area.json")
    import space_env.reference.lifecycle as module

    actual = geometry_manifest()
    monkeypatch.setattr(
        module, "geometry_manifest", lambda: {**actual, "proj": "different-build"}
    )
    with pytest.raises(
        ValueError, match="effective scenario differs|geometry manifest mismatch"
    ):
        MissionRuntime.load(tmp_path / "area.json")
