"""Bounded point-acquisition/eclipse geometry and atomic forecast bindings."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.core.constants import AU, MU_EARTH
from space_env.core.frames import ecef_to_geodetic
from space_env.reference import EOPSeries, Epoch, FrameContext
from space_env.reference.acquisition import (
    AcquisitionSensor,
    AcquisitionTarget,
    acquisition_geometry,
)
from space_env.reference.geometry_events import eclipse_windows, positive_windows
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    dynamic_reference_scenario,
)
from tests import test_reference_dynamic as fixtures


@pytest.fixture(scope="module")
def scenario():
    spec = copy.deepcopy(fixtures.dynamic_spec.__wrapped__())
    initial = spec["orbits"]["a"]["initial"]
    eop = spec["eop"]
    context = FrameContext(
        EOPSeries(
            eop["rows"],
            source=eop["source"],
            allow_predictions=eop["allow_predictions"],
        )
    )
    lat, lon, _ = ecef_to_geodetic(
        context.matrix(Epoch.utc(initial["epoch_utc"])) @ initial["r_m"]
    )
    spec["acquisition_targets"] = [{"id": "site", "lat_deg": lat, "lon_deg": lon}]
    spec["acquisition_sensors"] = {
        "a": {"max_off_nadir_deg": 35.0, "minimum_dwell_sec": 1.0}
    }
    return dynamic_reference_scenario(spec, id="geometry", lineage="B02-Q2")


def runtime(scenario):
    return MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))


def test_window_search_roots_touches_clipping_and_resolution():
    result = positive_windows(
        lambda t: (t - 2) * (8 - t), 0.0, 10.0, settings={"scan_sec": 1.0}
    )
    assert len(result["windows"]) == 1
    assert result["windows"][0]["start_sec"] == pytest.approx(2, abs=0.001)
    assert result["windows"][0]["end_sec"] == pytest.approx(8, abs=0.001)
    assert not positive_windows(
        lambda t: -((t - 5) ** 2), 0.0, 10.0, settings={"scan_sec": 1.0}
    )["windows"]
    assert not positive_windows(lambda t: 0.0, 0.0, 10.0)["windows"]
    touching = positive_windows(
        lambda t: (t - 5) ** 2, 0.0, 10.0, settings={"scan_sec": 1.0}
    )["windows"]
    assert (
        len(touching) == 1
        and touching[0]["clipped_start"]
        and touching[0]["clipped_end"]
    )
    # Demonstrate, rather than hide, the lack of a completeness certificate.
    narrow = lambda t: 0.1 - (t - 5.5) ** 2
    assert not positive_windows(narrow, 0.0, 10.0, settings={"scan_sec": 5.0})[
        "windows"
    ]
    assert positive_windows(narrow, 0.0, 10.0, settings={"scan_sec": 0.1})["windows"]


@pytest.mark.parametrize(
    "options",
    [
        {"scan_sec": True},
        {"scan_sec": 0.0},
        {"scan_sec": 0.1, "root_tolerance_sec": 1.0},
        {"scan_sec": 1e-300},
        {"max_evaluations": 2},
        {"root_tolerance_sec": float("nan")},
        {"extra": 1},
    ],
)
def test_search_rejects_invalid_or_excessive_controls(options):
    with pytest.raises(ValueError):
        positive_windows(lambda t: 1.0, 0.0, 30.0, settings=options)


def test_nonfinite_margin_and_refinement_budget_reject():
    with pytest.raises(ValueError, match="nonfinite"):
        positive_windows(lambda t: float("nan"), 0.0, 1.0)
    with pytest.raises(ValueError, match="budget"):
        positive_windows(
            lambda t: t * t - 0.3,
            0.0,
            1.0,
            settings={"scan_sec": 1.0, "max_evaluations": 2},
        )


def test_acquisition_forecast_is_readonly_and_is_not_an_imaging_operation(scenario):
    model = runtime(scenario)
    before = model.checkpoint()
    result = model.forecast_acquisition("a", "site")
    windows = result["result"]["windows"]
    assert windows and windows[0]["clipped_start"]
    assert result["result"]["opportunities"]
    assert model.checkpoint() == before
    assert "forecast.acquisition.geometry" in model.describe_capabilities()["supported"]
    assert "commands.imaging" not in model.describe_capabilities()["supported"]
    assert result["binding"]["actor_id"] == "a"
    provider = model._backend.mission.ephemerides["a"]
    look = acquisition_geometry(
        provider, model._backend.acquisition_targets["site"], AcquisitionSensor(), 0.0
    )
    assert look["elevation_deg"] > 89 and look["off_nadir_deg"] < 1


def test_solar_range_dwell_and_availability_constraints(scenario):
    data = copy.deepcopy(scenario)
    definition = data["definition"]
    definition["acquisition_targets"][0].update(
        available_from_sec=2.0, available_until_sec=5.0
    )
    model = runtime(data)
    result = model.forecast_acquisition("a", "site")["result"]
    assert (
        result["windows"][0]["start_sec"] == 2 and result["windows"][0]["end_sec"] == 5
    )
    definition["acquisition_sensors"]["a"]["minimum_dwell_sec"] = 4.0
    assert not runtime(data).forecast_acquisition("a", "site")["result"][
        "opportunities"
    ]
    definition["acquisition_sensors"]["a"]["max_range_m"] = 1.0
    assert not runtime(data).forecast_acquisition("a", "site")["result"]["windows"]
    definition["acquisition_sensors"]["a"].pop("max_range_m")
    baseline = runtime(data)
    provider = baseline._backend.mission.ephemerides["a"]
    target = baseline._backend.acquisition_targets["site"]
    sun_elevation = acquisition_geometry(provider, target, AcquisitionSensor(), 0.0)[
        "sun_elevation_deg"
    ]
    definition["acquisition_sensors"]["a"]["min_sun_elevation_deg"] = min(
        90.0, sun_elevation + 10.0
    )
    assert not runtime(data).forecast_acquisition("a", "site")["result"]["windows"]


def test_unknown_targets_sensors_permissions_and_bounds_fail_without_mutation(scenario):
    model = runtime(scenario)
    before = model.checkpoint()
    for call in (
        lambda: model.forecast_acquisition("a", "missing"),
        lambda: model.forecast_acquisition("b", "site"),
        lambda: model.forecast_eclipses("missing"),
        lambda: model.forecast_eclipses("a", end_sec=31),
        lambda: model.forecast_acquisition(
            "a", "site", settings={"max_evaluations": 2}
        ),
    ):
        with pytest.raises(RuntimeContractError):
            call()
        assert model.checkpoint() == before
    limited = MissionRuntime(scenario)
    with pytest.raises(RuntimeContractError) as error:
        limited.forecast_acquisition("a", "site")
    assert error.value.code == "permission_denied"


def test_bound_command_retry_and_stale_forecast_guard_are_atomic(scenario):
    model = runtime(scenario)
    binding = model.forecast_acquisition("a", "site")["binding"]
    command = {
        "command_id": "wait",
        "actor_id": "a",
        "kind": "wait",
        "duration_sec": 1.0,
        "forecast": binding,
    }
    admitted = model.submit([command], expected_revision=0, idempotency_key="bound")
    model.advance(1, expected_revision=1, idempotency_key="time")
    assert (
        model.submit([command], expected_revision=0, idempotency_key="bound")
        == admitted
    )
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        model.submit(
            [{**command, "command_id": "stale"}],
            expected_revision=2,
            idempotency_key="stale",
        )
    assert error.value.code == "stale_forecast" and model.checkpoint() == before


def test_foreign_actor_and_scenario_bindings_are_rejected(scenario):
    model = runtime(scenario)
    binding = model.forecast_acquisition("a", "site")["binding"]
    before = model.checkpoint()
    for altered in (
        {**binding, "actor_id": "b"},
        {**binding, "scenario_sha256": "0" * 64},
        {**binding, "trajectory_sha256": "0" * 64},
    ):
        with pytest.raises(RuntimeContractError, match="binding"):
            model.submit(
                [
                    {
                        "command_id": "bad",
                        "actor_id": "a",
                        "kind": "wait",
                        "duration_sec": 1.0,
                        "forecast": altered,
                    }
                ],
                expected_revision=0,
                idempotency_key="bad",
            )
        assert model.checkpoint() == before


def test_maneuver_changes_opportunities_and_checkpoint_keeps_bindings(
    scenario, tmp_path
):
    data = copy.deepcopy(scenario)
    data["definition"]["horizon_sec"] = 1200.0
    model = runtime(data)
    first = model.forecast_acquisition("a", "site")
    model.submit(
        [fixtures.impulse(dv=(0.0, 100.0, 0.0))],
        expected_revision=0,
        idempotency_key="kick",
    )
    second = model.forecast_acquisition("a", "site")
    assert (
        first["binding"]["trajectory_sha256"] != second["binding"]["trajectory_sha256"]
    )
    assert (
        first["result"]["opportunities"][0]["id"]
        != second["result"]["opportunities"][0]["id"]
    )
    assert (
        abs(
            first["result"]["windows"][0]["end_sec"]
            - second["result"]["windows"][0]["end_sec"]
        )
        > 0.001
    )
    command = {
        "command_id": "guarded",
        "actor_id": "a",
        "kind": "wait",
        "duration_sec": 1.0,
        "forecast": second["binding"],
    }
    model.submit([command], expected_revision=1, idempotency_key="guard")
    path = tmp_path / "geometry.json"
    model.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True, forecast=True))
    assert restored.forecast_acquisition("a", "site") == model.forecast_acquisition(
        "a", "site"
    )
    assert restored.checkpoint() == model.checkpoint()


@pytest.mark.parametrize("kind", ["duplicate", "actor", "coordinates", "availability"])
def test_invalid_geometry_configuration_rejects_scenario(scenario, kind):
    data = copy.deepcopy(scenario)
    definition = data["definition"]
    if kind == "duplicate":
        definition["acquisition_targets"] *= 2
    elif kind == "actor":
        definition["acquisition_sensors"]["missing"] = {}
    elif kind == "coordinates":
        definition["acquisition_targets"][0]["lat_deg"] = 91.0
    else:
        definition["acquisition_targets"][0]["available_until_sec"] = 31.0
    with pytest.raises(RuntimeContractError):
        runtime(data)


class CircularProvider:
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    radius = 7e6
    motion = math.sqrt(MU_EARTH / radius**3)
    duration_sec = 2 * math.pi / motion
    sha256 = "synthetic-circle"

    def state(self, clock):
        phase = self.motion * clock
        return np.array(
            [
                self.radius * math.cos(phase),
                self.radius * math.sin(phase),
                0.0,
                -self.radius * self.motion * math.sin(phase),
                self.radius * self.motion * math.cos(phase),
                0.0,
            ]
        )


def test_eclipse_events_are_nested_and_explicitly_use_existing_conical_model():
    provider = CircularProvider()
    result = eclipse_windows(
        provider,
        sun_gcrs=lambda epoch: np.array([AU, 0.0, 0.0]),
        settings={"scan_sec": 20.0},
    )
    umbra, any_shadow = result["umbra"]["windows"], result["partial_or_full"]["windows"]
    assert len(umbra) == len(any_shadow) == 1
    assert (
        any_shadow[0]["start_sec"]
        < umbra[0]["start_sec"]
        < umbra[0]["end_sec"]
        < any_shadow[0]["end_sec"]
    )
    assert "linear-penumbra" in result["model"]
    assert result["sun_source"].startswith("caller-declared")
    assert not umbra[0]["clipped_start"] and not umbra[0]["clipped_end"]


def test_eclipse_forecast_is_privileged_readonly_and_defaults_to_current_time(scenario):
    model = runtime(scenario)
    before = model.checkpoint()
    forecast = model.forecast_eclipses("a")
    assert forecast["result"]["coverage_sec"] == [0, 30]
    assert model.checkpoint() == before
    model.advance(1, expected_revision=0, idempotency_key="advance")
    assert model.forecast_eclipses("a")["result"]["coverage_sec"] == [1, 30]


def test_analytic_geometry_event_comparison():
    from space_env.validation.geometry_events import geometry_event_report

    report = geometry_event_report()
    assert report["passed"], report["checks"]
    assert len(report["checks"]) == 4
    assert report["qualification_status"] == "needs_external_evidence"


def test_forecast_after_cancel_changes_binding_and_empty_availability_is_explicit(
    scenario,
):
    data = copy.deepcopy(scenario)
    data["definition"]["acquisition_targets"][0]["available_until_sec"] = 1.0
    model = runtime(data)
    model.submit([fixtures.burn()], expected_revision=0, idempotency_key="burn")
    before_cancel = model.forecast_acquisition("a", "site")["binding"]
    model.cancel("burn", expected_revision=1, idempotency_key="cancel")
    after = model.forecast_acquisition("a", "site")
    assert before_cancel["trajectory_sha256"] != after["binding"]["trajectory_sha256"]
    empty = model.forecast_acquisition("a", "site", start_sec=2.0)["result"]
    assert empty["windows"] == empty["opportunities"] == []
    assert empty["search_details"] is empty["search_coverage_sec"] is None


def test_fixed_reference_supports_eclipse_and_binding_but_not_acquisition():
    from space_env.runtime import reference_scenario
    from tests import test_reference_operations as fixed

    model = runtime(
        reference_scenario(fixed.spec.__wrapped__(), id="fixed", lineage="B02-Q2")
    )
    report = model.forecast_eclipses("a")
    assert report["result"]["model"] == "core.spherical-conical-linear-penumbra/1"
    binding = model.forecast("a", "Ground")["binding"]
    model.submit(
        [
            {
                "command_id": "radio",
                "actor_id": "a",
                "kind": "downlink",
                "duration_sec": 1.0,
                "destination_id": "Ground",
                "forecast": binding,
            }
        ],
        expected_revision=0,
        idempotency_key="radio",
    )
    with pytest.raises(RuntimeContractError) as error:
        model.forecast_acquisition("a", "site")
    assert error.value.code == "unsupported_capability"


def test_antimeridian_and_pole_targets_do_not_create_coordinate_discontinuity(scenario):
    model = runtime(scenario)
    provider = model._backend.mission.ephemerides["a"]
    sensor = AcquisitionSensor()
    for latitude in (0.0, 90.0, -90.0):
        west = acquisition_geometry(
            provider,
            AcquisitionTarget(id="west", lat_deg=latitude, lon_deg=-180.0),
            sensor,
            0.0,
        )
        east = acquisition_geometry(
            provider,
            AcquisitionTarget(id="east", lat_deg=latitude, lon_deg=180.0),
            sensor,
            0.0,
        )
        for key in ("range_m", "elevation_deg", "off_nadir_deg", "sun_elevation_deg"):
            assert west[key] == pytest.approx(east[key], abs=1e-7)


def test_eclipse_global_geometry_budget_and_invalid_sun_are_guarded():
    provider = CircularProvider()
    with pytest.raises(ValueError, match="budget"):
        eclipse_windows(provider, settings={"max_evaluations": 2})
    with pytest.raises(ValueError, match="geometry"):
        eclipse_windows(provider, sun_gcrs=lambda e: np.array([1.0, 0.0, 0.0]))


def test_opportunities_reserve_root_precision_margin_before_dwell_acceptance(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["horizon_sec"] = 1200.0
    result = runtime(data).forecast_acquisition("a", "site")["result"]
    row = result["opportunities"][0]
    assert row["usable_end_sec"] == pytest.approx(row["end_sec"] - 0.001)
    assert row["usable_duration_sec"] < row["duration_sec"]
    data["definition"]["acquisition_sensors"]["a"]["minimum_dwell_sec"] = (
        row["duration_sec"] - 0.0005
    )
    assert not runtime(data).forecast_acquisition("a", "site")["result"][
        "opportunities"
    ]


def test_root_at_query_boundary_is_not_mislabeled_as_clipped_interior():
    starts_at_root = positive_windows(lambda t: t, 0.0, 1.0)["windows"][0]
    ends_at_root = positive_windows(lambda t: 1 - t, 0.0, 1.0)["windows"][0]
    assert not starts_at_root["clipped_start"] and starts_at_root["clipped_end"]
    assert ends_at_root["clipped_start"] and not ends_at_root["clipped_end"]
