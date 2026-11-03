"""Q5-C1 evidence provenance, anti-leakage fits and non-resampled sensitivity."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from space_env.calibration import (
    calibrate,
    compile_ensemble,
    hardware_catalog,
    hardware_profile,
    measurement_payload,
    sample_parameters,
    solar_resources,
    verify_source_bytes,
    wheel_model,
)
from space_env.calibration import hardware as hw
from space_env.calibration.schema import EnsembleSpec
from space_env.simulation.snapshot import digest
from space_env.validation import calibration as validation

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return json.loads(validation.FIXTURE.read_text())["cases"]


def seal(spec):
    spec["source"]["content_sha256"] = digest(measurement_payload(spec))
    return spec


def assumptions():
    return {
        "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "efficiency": 0.8,
        "idle_power_w": 0.1,
        "torque_loss_w_per_nm2": 1.0,
        "rationale": "synthetic host/body/efficiency assumptions; no measured power curve",
    }


def parameter(id="battery", target="battery_capacity_scale", **updates):
    return {
        "id": id,
        "actor": "a",
        "target": target,
        "mode": "uniform",
        "low": 0.8,
        "high": 1.1,
        "rationale": "assumed sensitivity range, not an empirical manufacturer tolerance",
        **updates,
    }


def ensemble_spec(**updates):
    return {
        "id": "test-ensemble",
        "seed": 31,
        "samples": 4,
        "parameters": [parameter()],
        **updates,
    }


@pytest.fixture(scope="module")
def scenario():
    from tests import test_rigid_mission

    return test_rigid_mission.scenario.__wrapped__()


def test_curated_catalog_preserves_units_conditions_and_missing_uncertainty():
    catalog = hardware_catalog()
    assert len(catalog["profiles"]) == 6 and len(catalog["sources"]) == 2
    assert all(
        p["status"] == "published_reference_point" and p["uncertainty"]
        for p in catalog["profiles"].values()
    )
    p = hardware_profile("cw0017")
    assert p["values_si"]["rated_momentum_nms"] == pytest.approx(0.00177)
    assert p["values_si"]["torque_at_rated_momentum_nm"] == pytest.approx(0.00023)
    assert p["values_si"]["steady_at_rated_momentum_w"] == pytest.approx(0.3)
    assert p["values_si"]["speed_at_rated_momentum_rad_s"] == pytest.approx(
        8000 * math.pi / 30
    )
    assert (
        hardware_profile("cw0500")["values_si"]["torque_at_rated_momentum_nm"] == 0.01
    )
    solar = hardware_profile("azur-3g30-bare")
    assert solar["values_si"]["efficiency"] == 0.298
    assert solar["values_si"]["temperature_k"] == 301.15
    assert solar["values_si"]["irradiance_w_m2"] == 1353.0


def test_catalog_and_adapters_own_their_data():
    first = hardware_catalog()
    first["profiles"]["cw0017"]["values_si"]["rated_momentum_nms"] = 1000.0
    assert hardware_profile("cw0017")["values_si"]["rated_momentum_nms"] < 0.01
    model = wheel_model(["cw0017"] * 3, assumptions())
    assert model["qualification_status"] == "mixed-published-derived-assumed"
    assert not model["industrial_readiness"]
    assert model["model"]["wheel_inertia_kg_m2"][0] == pytest.approx(
        0.00177 / (8000 * math.pi / 30)
    )
    assert (
        model["model"]["idle_power_w"] == 0.1
    )  # steady-at-speed published power is NOT silently idle power
    assert "not hardware maximum" in model["parameter_evidence"]["max_momentum_nms"]


@pytest.mark.parametrize("id", ["unknown", [], None])
def test_unknown_or_nonstring_profile_rejected(id):
    with pytest.raises(ValueError):
        hardware_profile(id)


@pytest.mark.parametrize(
    "profiles", [[], ["cw0017"], ["cw0017"] * 4, ["azur-3g30-bare"] * 3]
)
def test_wheel_assembly_requires_three_declared_wheels(profiles):
    with pytest.raises(ValueError):
        wheel_model(profiles, assumptions())


@pytest.mark.parametrize(
    "field,value",
    [
        ("efficiency", True),
        ("efficiency", 1.1),
        ("idle_power_w", -1.0),
        ("torque_loss_w_per_nm2", 0.0),
        ("rationale", " " * 20),
        ("inertia_kg_m2", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]]),
    ],
)
def test_invalid_host_assumptions_rejected(field, value):
    value_dict = assumptions()
    value_dict[field] = value
    with pytest.raises(ValueError):
        wheel_model(["cw0017"] * 3, value_dict)


def test_resigned_packaged_catalog_corruption_is_not_accepted(monkeypatch, tmp_path):
    data = hardware_catalog()
    data["profiles"]["cw0500"]["values_si"]["torque_at_rated_momentum_nm"] = 0.016
    data.pop("sha256")
    data["sha256"] = digest(data)
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(hw, "CATALOG", path)
    with pytest.raises(ValueError, match="frozen"):
        hardware_catalog()


def test_source_bytes_integrity_has_no_authentication_claim():
    source = hardware_catalog()["sources"]["cubespace-wheels"]
    raw = b"local source integrity test"
    source["content_sha256"] = hashlib.sha256(raw).hexdigest()
    result = verify_source_bytes(source, raw)
    assert result["passed"] and not result["independent_authentication"]
    assert not verify_source_bytes(source, raw + b"altered")["passed"]
    with pytest.raises(ValueError):
        verify_source_bytes(source, "not bytes")


def test_solar_mapping_keeps_geometry_and_losses_explicit():
    base = {"memory_bits": 1e6, "battery_wh": 20.0, "solar_efficiency": 0.4}
    original = copy.deepcopy(base)
    result = solar_resources(
        "azur-3g30-cic",
        base,
        area_m2=0.2,
        packing_factor=0.9,
        assumption="BOL constant efficiency at declared reference conditions only",
    )
    assert base == original
    assert result["vehicle_resources"]["solar_efficiency"] == pytest.approx(0.295 * 0.9)
    assert result["vehicle_resources"]["solar_area_m2"] == 0.2
    assert result["qualification_status"] == "mixed-published-derived-assumed"
    with pytest.raises(ValueError):
        solar_resources(
            "cw0017",
            base,
            area_m2=0.2,
            packing_factor=0.9,
            assumption="not a solar cell reference point",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"area_m2": True},
        {"area_m2": 0.0},
        {"packing_factor": 1.1},
        {"packing_factor": float("nan")},
        {"assumption": " "},
    ],
)
def test_solar_assembly_invalid_inputs(updates):
    kwargs = {
        "area_m2": 0.1,
        "packing_factor": 0.9,
        "assumption": "explicit synthetic reference condition approximation",
        **updates,
    }
    with pytest.raises(ValueError):
        solar_resources(
            "azur-3g30-bare", {"memory_bits": 1e6, "battery_wh": 20.0}, **kwargs
        )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_synthetic_fit_reports_conditional_uncertainty_without_calibration_promotion(
    index,
):
    case = cases()[index]
    report = calibrate(case["spec"])
    assert report["passed"]
    assert report["fit_status"] == "synthetic_workflow_verified"
    assert (
        not report["independent_source_authentication"]
        and not report["industrial_readiness"]
    )
    assert (
        report["parameter"]["conditional_normal_95_interval"][0]
        < case["truth"]
        < report["parameter"]["conditional_normal_95_interval"][1]
    )
    assert report["train_degrees_of_freedom"] == 5
    assert report["data_sha256"] == case["spec"]["source"]["content_sha256"]
    assert report["sha256"] == digest(
        {k: v for k, v in report.items() if k != "sha256"}
    )
    if index:
        inverse = report["derived_reciprocal"]
        assert inverse["estimate"] == pytest.approx(1 / report["parameter"]["estimate"])
        lo, hi = report["parameter"]["conditional_normal_95_interval"]
        assert inverse["conditional_95_interval"] == pytest.approx([1 / hi, 1 / lo])


@pytest.mark.parametrize("index", [0, 1, 2])
def test_changed_holdout_never_changes_fit_or_standard_error(index):
    spec = cases()[index]["spec"]
    original = calibrate(spec)
    bad = copy.deepcopy(spec)
    for row in bad["observations"]:
        if row["split"] == "holdout":
            row["y"] += 1000.0
            row["sigma_y"] *= 10
    report = calibrate(seal(bad))
    assert report["parameter"] == original["parameter"]
    assert report["train"] == original["train"]
    assert not report["passed"] and not report["gates"]["holdout_absolute_error"]
    assert len(report["holdout"]) == 3


def test_measurement_digest_cannot_be_silently_recomputed():
    spec = cases()[0]["spec"]
    spec["observations"][-1]["y"] += 1.0
    with pytest.raises(ValueError, match="digest"):
        calibrate(spec)


def test_noiseless_known_gain_and_known_noise_information():
    spec = cases()[0]["spec"]
    for row in spec["observations"]:
        row["y"] = row["x"] * 0.27
        row["sigma_y"] = 1.0
    report = calibrate(seal(spec))
    assert report["passed"] and report["parameter"]["estimate"] == pytest.approx(
        0.27, abs=1e-15
    )
    information = sum(
        r["x"] ** 2 for r in spec["observations"] if r["split"] == "train"
    )
    assert report["parameter"]["standard_error"] == pytest.approx(
        1 / math.sqrt(information)
    )


def test_zero_excitation_is_unidentifiable_not_a_zero_uncertainty_fit():
    spec = cases()[0]["spec"]
    for row in spec["observations"]:
        if row["split"] == "train":
            row["x"] = 0.0
    with pytest.raises(ValueError, match="unidentifiable"):
        calibrate(seal(spec))


def test_holdout_extrapolation_not_qualified_by_small_residual():
    spec = cases()[0]["spec"]
    gain = calibrate(spec)["parameter"]["estimate"]
    row = spec["observations"][-1]
    row["x"] = 101.0
    row["y"] = gain * row["x"]
    report = calibrate(seal(spec))
    assert (
        not report["passed"] and not report["gates"]["holdout_within_training_domain"]
    )
    assert report["gates"]["holdout_absolute_error"]


@pytest.mark.parametrize(
    "mutation",
    [
        "model_unit",
        "duplicate_id",
        "experiment_reuse",
        "too_few_train",
        "nan_y",
        "bool_x",
        "zero_sigma",
        "datasheet",
        "unknown_field",
        "invalid_bounds",
        "blank_conditions",
    ],
)
def test_bad_calibration_contract_rejected(mutation):
    spec = cases()[0]["spec"]
    if mutation == "model_unit":
        spec["x_unit"] = "J"
    elif mutation == "duplicate_id":
        spec["observations"][1]["id"] = spec["observations"][0]["id"]
    elif mutation == "experiment_reuse":
        spec["observations"][-1]["experiment_id"] = spec["observations"][0][
            "experiment_id"
        ]
    elif mutation == "too_few_train":
        spec["gates"]["min_train"] = 7
    elif mutation == "nan_y":
        spec["observations"][0]["y"] = float("nan")
    elif mutation == "bool_x":
        spec["observations"][0]["x"] = True
    elif mutation == "zero_sigma":
        spec["observations"][0]["sigma_y"] = 0.0
    elif mutation == "datasheet":
        spec["source"].update(
            kind="manufacturer_specification", uri="https://example.com/datasheet"
        )
    elif mutation == "unknown_field":
        spec["calibrated"] = True
    elif mutation == "invalid_bounds":
        spec["gates"]["parameter_high"] = spec["gates"]["parameter_low"]
    elif mutation == "blank_conditions":
        spec["conditions"] = " " * 20
    with pytest.raises(ValueError):
        calibrate(spec)


def test_declared_measurement_is_not_authenticated_calibration():
    spec = cases()[0]["spec"]
    spec["source"].update(
        kind="laboratory_measurement",
        uri="https://example.invalid/declared-measurements",
    )
    report = calibrate(spec)
    assert (
        report["passed"]
        and report["fit_status"] == "declared_measurements_holdout_checked"
    )
    assert report["independent_source_authentication"] is False


def test_seeded_sampling_prefix_order_invariance_and_rng_ownership():
    first = ensemble_spec(
        parameters=[
            parameter(),
            parameter(
                "inertia",
                "body_inertia_scale",
                mode="truncated_normal",
                mean=1.0,
                sigma=0.05,
                low=0.8,
                high=1.2,
            ),
        ]
    )
    state = random.getstate()
    result = sample_parameters(first)
    assert random.getstate() == state
    assert result == sample_parameters(first)
    reverse = copy.deepcopy(first)
    reverse["parameters"].reverse()
    assert [r["draws"] for r in result["members"]] == [
        r["draws"] for r in sample_parameters(reverse)["members"]
    ]
    larger = copy.deepcopy(first)
    larger["samples"] = 8
    assert result["members"] == sample_parameters(larger)["members"][:4]
    extra = copy.deepcopy(first)
    extra["parameters"].append(parameter("solar", "solar_efficiency_scale"))
    assert all(
        all(new["draws"][k] == v for k, v in old["draws"].items())
        for old, new in zip(result["members"], sample_parameters(extra)["members"])
    )
    assert all(0.8 <= r["draws"]["inertia"] <= 1.2 for r in result["members"])


def test_shared_rank_groups_are_explicit_not_fitted_covariance():
    spec = ensemble_spec(
        parameters=[
            parameter("battery", "battery_capacity_scale", latent_group="aging"),
            parameter("solar", "solar_efficiency_scale", latent_group="aging"),
        ]
    )
    assert all(
        r["draws"]["battery"] == r["draws"]["solar"]
        for r in sample_parameters(spec)["members"]
    )


def test_empty_normal_tail_preserves_attempts_instead_of_resampling():
    spec = ensemble_spec(
        parameters=[parameter(mode="truncated_normal", mean=1e10, sigma=0.01)]
    )
    result = sample_parameters(spec)
    assert result["attempted"] == 4 and len(result["members"]) == 4
    assert all(len(r["errors"]) == 1 for r in result["members"])


@pytest.mark.parametrize(
    "mutation",
    [
        "samples",
        "boolean_seed",
        "duplicate_target",
        "duplicate_id",
        "distribution",
        "normal_missing",
        "uniform_extra",
        "claimed_measured",
        "unordered",
        "unknown_unit",
    ],
)
def test_invalid_ensemble_contract(mutation):
    spec = ensemble_spec()
    p = spec["parameters"][0]
    if mutation == "samples":
        spec["samples"] = 65
    elif mutation == "boolean_seed":
        spec["seed"] = True
    elif mutation == "duplicate_target":
        spec["parameters"].append(parameter("other"))
    elif mutation == "duplicate_id":
        spec["parameters"].append(parameter("battery", "solar_efficiency_scale"))
    elif mutation == "distribution":
        p["mode"] = "gaussian_covariance"
    elif mutation == "normal_missing":
        p["mode"] = "truncated_normal"
    elif mutation == "uniform_extra":
        p["mean"] = 1.0
    elif mutation == "claimed_measured":
        p["status"] = "manufacturer_tolerance"
    elif mutation == "unordered":
        p["high"] = p["low"]
    elif mutation == "unknown_unit":
        p["unit"] = "kg"
    with pytest.raises(ValueError):
        EnsembleSpec.model_validate(spec)


def test_compiler_owns_base_keeps_rng_and_downgrades_inherited_calibration(scenario):
    base = copy.deepcopy(scenario)
    base["definition"]["spacecraft_bus"]["a"]["parameter_status"] = "calibrated"
    original = digest(base)
    result = compile_ensemble(
        base, ensemble_spec(parameters=[parameter("inertia", "body_inertia_scale")])
    )
    assert (
        digest(base) == original
        and result["structural_candidates"] == 4
        and result["rejected"] == 0
    )
    assert not result["native_execution_validated"]
    for member in result["members"]:
        candidate = member["scenario"]
        bus = candidate["definition"]["spacecraft_bus"]["a"]
        assert bus["parameter_status"] == "synthetic"
        assert (
            candidate["seed"] == base["seed"]
            and candidate["metadata"] == base["metadata"]
        )
        assert any("base calibration not transferred" in s for s in bus["sources"])
        assert "Assumed sensitivity" in bus["attitude"]["rigid"]["model"]["source"]
        assert member["scenario_sha256"] == digest(candidate)


def test_all_draws_and_failures_retained_for_invalid_motor_envelope(scenario):
    spec = ensemble_spec(
        parameters=[
            parameter("aa-torque", "wheel_torque_scale", low=2.0, high=3.0),
            parameter(),
        ]
    )
    result = compile_ensemble(scenario, spec)
    assert result["attempted"] == result["rejected"] == len(result["members"]) == 4
    assert result["structural_candidates"] == 0
    assert all(set(r["draws"]) == {"aa-torque", "battery"} for r in result["members"])
    assert all("supply" in r["error"]["message"] for r in result["members"])


def test_unknown_actor_and_active_thermal_network_do_not_silently_ignore_binding(
    scenario,
):
    bad = ensemble_spec(parameters=[parameter(actor="missing")])
    assert compile_ensemble(scenario, bad)["rejected"] == 4
    from tests.test_thermal_network import network

    active = copy.deepcopy(scenario)
    active["definition"]["spacecraft_bus"]["a"]["thermal_network"] = network()
    result = compile_ensemble(
        active,
        ensemble_spec(parameters=[parameter("thermal", "thermal_capacity_scale")]),
    )
    assert result["rejected"] == 4
    assert all(
        "active node network" in r["error"]["message"] for r in result["members"]
    )
    unsupported = copy.deepcopy(scenario)
    unsupported["backend"] = "reference-fixed"
    with pytest.raises(ValueError):
        compile_ensemble(unsupported, ensemble_spec())


def test_fixture_generator_reproduces_bytes(tmp_path):
    script = ROOT / "scripts/generate_calibration_fixtures.py"
    output = tmp_path / "fixtures.json"
    subprocess.run(
        [sys.executable, "-I", str(script), str(output)], cwd=tmp_path, check=True
    )
    assert output.read_bytes() == validation.FIXTURE.read_bytes()


def test_packaged_workflow_reports_synthetic_scope_and_negative_holdout():
    report = validation.calibration_workflow_report()
    assert report["passed"] and len(report["checks"]) == 14
    assert (
        not report["hardware_calibration_complete"]
        and not report["industrial_readiness"]
    )
    assert not report["negative_holdout"]["passed"]
    assert all(r["source"]["kind"] == "synthetic_fixture" for r in report["fits"])
    assert not report["source_bytes_revalidated_in_run"]


def test_resigned_workflow_fixture_not_accepted(monkeypatch, tmp_path):
    data = json.loads(validation.FIXTURE.read_text())
    data["cases"][0]["spec"]["gates"]["max_holdout_absolute_error"] = 1e6
    data.pop("sha256")
    data["sha256"] = digest(data)
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(validation, "FIXTURE", path)
    with pytest.raises(ValueError, match="frozen"):
        validation.calibration_workflow_report()


def test_cli_persists_failed_fit_and_rejects_input_output_alias(tmp_path, capsys):
    from space_env.cli import main

    spec = cases()[0]["spec"]
    for row in spec["observations"]:
        if row["split"] == "holdout":
            row["y"] += 100.0
    seal(spec)
    source = tmp_path / "input.json"
    target = tmp_path / "report.json"
    source.write_text(json.dumps(spec))
    with pytest.raises(SystemExit) as exc:
        main(["calibration-fit", str(source), str(target)])
    assert exc.value.code == 1 and not json.loads(target.read_text())["passed"]
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["calibration-fit", str(source), str(source)])
    assert exc.value.code == 2 and source.read_bytes() == original
    assert "calibration_contract_error" in capsys.readouterr().out


@pytest.mark.parametrize(
    "updates",
    [
        {"retrieved_on": "2026-02-30"},
        {"uri": "https://"},
        {"uri": "https://user:password@example.invalid/source"},
    ],
)
def test_invalid_source_date_host_or_embedded_credentials(updates):
    from space_env.calibration.schema import EvidenceSource

    source = hardware_catalog()["sources"]["cubespace-wheels"]
    source.update(updates)
    with pytest.raises(ValueError):
        EvidenceSource.model_validate(source)


def test_initial_wheel_momentum_is_checked_after_capacity_draw(scenario):
    base = copy.deepcopy(scenario)
    base["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "initial_wheel_momentum_nms"
    ] = [0.05, 0.0, 0.0]
    spec = ensemble_spec(
        parameters=[parameter("capacity", "wheel_momentum_scale", low=0.01, high=0.02)]
    )
    result = compile_ensemble(base, spec)
    assert result["rejected"] == 4
    assert all(
        "initial wheel momentum exceeds capacity" in r["error"]["message"]
        for r in result["members"]
    )


def test_cli_large_input_does_not_create_output(tmp_path, capsys):
    from space_env.cli import main

    source = tmp_path / "large.json"
    source.write_bytes(b" " * 2_000_001)
    output = tmp_path / "output.json"
    with pytest.raises(SystemExit) as exc:
        main(["calibration-fit", str(source), str(output)])
    assert exc.value.code == 2 and not output.exists()
    assert "exceeds 2MB" in capsys.readouterr().out


@pytest.mark.parametrize(
    "updates",
    [
        {"frame": "scalar"},
        {"axis": None},
        {"experiment_model": "constant-effective-efficiency"},
    ],
)
def test_principal_axis_calibration_requires_compatible_frame_and_experiment(updates):
    spec = cases()[1]["spec"]
    spec.update(updates)
    with pytest.raises(ValueError, match="frame/axis"):
        calibrate(spec)
