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
    assert names["pc"] == "not_independently_validated"
    assert names["iau_2000a"] == "not_implemented"
    assert names["teme_to_gcrf"] == "not_implemented"


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
