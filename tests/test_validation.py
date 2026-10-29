"""Эталоны ядра и сценарий миссии, который собирается из файла."""
from pathlib import Path

import gymnasium as gym
import pytest
import yaml

import space_env.envs  # регистрация сред
from space_env.mission.shift_scenario import load_mission_scenario
from space_env.validation.report import applicability_limits, core_report

_EO = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "eo_pass.yaml"


def test_core_report_records_measured_error_against_a_prior_tolerance():
    report = core_report()
    assert report["passed"] is True
    names = {item["name"] for item in report["cases"]}
    assert names == {"two_body_energy", "two_body_angular_momentum", "j2_node", "drag_lowers_radius"}
    energy = next(item for item in report["cases"] if item["name"] == "two_body_energy")
    assert energy["tolerance"] == 1.0e-8
    assert energy["measured"] < energy["tolerance"]
    assert energy["status"] == "verified"
    drag = next(item for item in report["cases"] if item["name"] == "drag_lowers_radius")
    assert drag["measured"] > 0.0
    assert drag["status"] == "approximate"


def test_applicability_names_what_has_no_independent_reference():
    names = {item["name"]: item["status"] for item in applicability_limits()}
    assert names["pc"] == "checked_against_disk_integral"
    assert names["iau_2000a"] == "training_frame"
    assert names["sun_ephemeris"] == "epv00_shared"
    assert names["flux_along_step"] == "sample_average"
    assert names["centered_f107"] == "inside_snapshot"
    assert names["star_1w"] == "evaluated_open"
    assert names["disposal_25y"] == "scale_height_estimate"
    assert names["teme_to_gcrf"] == "vallado_2006_itrs"
    assert names["rk4_stage_epoch"] == "stage_epoch"
    assert names["outcome_error"] == "mission_contract"
    assert names["external_backend"] == "does_not_replace_validation"
    assert names["finite_burn"] == "separate_profile"
    assert names["memory_queue"] == "configured"
    assert names["slew_accel"] == "configured"
    assert names["power_hysteresis"] == "configured"


def test_a_broken_measurement_fails_the_same_tolerance_rule():
    from space_env.validation.report import _case

    broken = _case("broken_two_body", 1.0, 1.0e-8, "испорченная модель", "энергия двух тел")
    assert broken["passed"] is False
    assert core_report()["passed"] is True


def test_rk4_stages_use_distinct_epochs():
    from datetime import datetime, timedelta

    import numpy as np

    from space_env.core.gravity import acceleration_m_s2
    from space_env.core.propagate import propagate_state

    seen = []
    real = acceleration_m_s2

    def spy(*args, **kwargs):
        seen.append(kwargs.get("when"))
        return real(*args, **kwargs)

    import space_env.core.propagate as propagate

    original = propagate.acceleration_m_s2
    propagate.acceleration_m_s2 = spy
    try:
        start = datetime(2024, 1, 1, 0, 0, 0)
        propagate_state(
            np.array([6_800_000.0, 0.0, 0.0]),
            np.array([0.0, 7_500.0, 0.0]),
            10.0,
            mass_kg=100.0,
            drag_area_m2=0.0,
            drag_coefficient=0.0,
            include_j2=False,
            include_drag=False,
            substep_sec=10.0,
            when=start,
        )
    finally:
        propagate.acceleration_m_s2 = original
    assert seen[0] == start
    assert seen[1] == start + timedelta(seconds=5.0)
    assert seen[2] == start + timedelta(seconds=5.0)
    assert seen[3] == start + timedelta(seconds=10.0)


def test_stage_coordinates_move_when_density_reads_them():
    from datetime import datetime

    import numpy as np

    from space_env.core.propagate import propagate_state

    seen: list[tuple[float, float]] = []

    def spy(*_args, **kwargs):
        seen.append((kwargs["lat_deg"], kwargs["lon_deg"]))
        return np.zeros(3)

    import space_env.core.propagate as propagate

    original = propagate.acceleration_m_s2
    propagate.acceleration_m_s2 = spy
    try:
        propagate_state(
            np.array([6_800_000.0, 0.0, 0.0]),
            np.array([0.0, 7_500.0, 0.0]),
            10.0,
            mass_kg=100.0,
            drag_area_m2=1.0,
            drag_coefficient=2.2,
            include_j2=False,
            include_drag=True,
            atmosphere="nrlmsise",
            substep_sec=10.0,
            when=datetime(2024, 1, 1, 0, 0, 0),
        )
    finally:
        propagate.acceleration_m_s2 = original
    assert len(seen) == 4
    assert seen[0][1] != pytest.approx(seen[3][1])


def test_reference_pack_keeps_prior_tolerances_and_rejects_euler():
    import math

    import numpy as np

    from space_env.core.constants import MU_EARTH
    from space_env.core.orbit import Orbit
    from space_env.core.time import Time
    from space_env.validation.profiles import REFERENCE_PROFILE, TRAINING_PROFILE, optional_backends
    from space_env.validation.report import reference_report

    report = reference_report()
    assert report["passed"] is True
    assert TRAINING_PROFILE.position_tolerance_m is None
    assert all(item["replaces_validation"] is False and item["teme_to_gcrf"] is False for item in optional_backends())
    by_name = {item["name"]: item for item in report["cases"]}
    assert by_name["reference_position"]["tolerance"] == REFERENCE_PROFILE.position_tolerance_m == 50.0
    assert by_name["reference_velocity"]["tolerance"] == 1.0
    assert by_name["rk4_step_convergence"]["tolerance"] == 8.0
    assert by_name["rk4_step_convergence"]["passed"] is True
    assert by_name["contact_boundary"]["tolerance"] == 1.0
    assert by_name["moon_distance_envelope"]["status"] == "approximate"
    assert by_name["cannonball_srp"]["measured"] < 1.0e-9

    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6_800_000.0, 0.001, 0.0, 0.2, 0.1, 0.3, epoch)
    kepler = orbit.propagate(600.0)
    r = orbit.r.copy()
    v = orbit.v.copy()
    elapsed = 0.0
    while elapsed < 600.0 - 1e-9:
        step = min(5.0, 600.0 - elapsed)
        radius = float(np.linalg.norm(r))
        acceleration = -MU_EARTH * r / radius**3
        r = r + step * v
        v = v + step * acceleration
        elapsed += step
    euler_error_m = float(np.linalg.norm(r - kepler.r))
    assert euler_error_m > REFERENCE_PROFILE.position_tolerance_m
    assert math.isfinite(euler_error_m)


def test_mission_file_builds_a_different_shift_and_rejects_a_bad_field(tmp_path):
    loaded = load_mission_scenario(_EO)
    assert loaded["keys"] == ("sentinel-2a",)
    assert loaded["horizon_sec"] == 3600.0
    assert loaded["epoch"] == "2026-06-21T12:00:00Z"
    env = gym.make("space_env/OperatorShift-v0", scenario=str(_EO), max_episode_steps=2)
    try:
        assert env.unwrapped._keys == ("sentinel-2a",)
        assert env.unwrapped._horizon_sec == 3600.0
        assert env.unwrapped._nu_deg == 0.0
    finally:
        env.close()
    broken = tmp_path / "bad.yaml"
    broken.write_text(
        yaml.safe_dump({"keys": ["sentinel-2a", "sentinel-2a"], "horizon_sec": 100.0, "extra": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="неизвестные поля"):
        load_mission_scenario(broken)
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump({"keys": ["sentinel-2a", "sentinel-2a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="повторяется"):
        load_mission_scenario(duplicate)
    negative = tmp_path / "negative.yaml"
    negative.write_text(yaml.safe_dump({"keys": ["sentinel-2a"], "dv_budget_m_s": -1}), encoding="utf-8")
    with pytest.raises(ValueError, match="бюджет"):
        load_mission_scenario(negative)
