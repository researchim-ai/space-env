"""Сценарий v2 и снимок общих часов."""
from pathlib import Path

import numpy as np
import pytest

from space_env.catalog.build import make_engine
from space_env.mission.scenario_v2 import load_scenario_v2, scenario_manifest, world_from_scenario
from space_env.simulation.world import MissionWorld

_EO = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "eo_72h.yaml"
_EO16 = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "eo_72h_16.yaml"


def test_seventy_two_hour_file_repeats_one_model_without_a_new_env():
    spec = load_scenario_v2(_EO)
    assert spec["horizon_sec"] == 259200
    assert [item["id"] for item in spec["instances"]] == ["sentinel-a", "sentinel-b", "landsat", "radar"]
    assert [item["model"] for item in spec["instances"]].count("sentinel-2a") == 2
    assert spec["stations"] == ["Kiruna", "Svalbard", "Matera"]
    assert spec["epoch"] == "2026-10-20T10:00:00Z"
    assert [item["id"] for item in spec["requests"]] == ["kiruna-day", "svalbard-radar"]
    assert spec["requests"][0]["payload"] == "optical"
    assert spec["requests"][1]["payload"] == "sar"
    manifest = scenario_manifest(_EO)
    assert manifest["sha256"]
    assert manifest["requests"] == ["kiruna-day", "svalbard-radar"]
    assert manifest["stations"] == ["Kiruna", "Svalbard", "Matera"]
    wide = load_scenario_v2(_EO16)
    assert wide["horizon_sec"] == 259200
    assert len(wide["instances"]) == 16
    assert len({item["id"] for item in wide["instances"]}) == 16
    assert {item["model"] for item in wide["instances"]} == {"sentinel-2a", "landsat-9", "sar-small"}


def test_duplicate_instance_id_is_rejected(tmp_path: Path):
    path = tmp_path / "dup.yaml"
    path.write_text(
        "version: 2\nepoch: '2026-10-20T00:00:00Z'\nhorizon_sec: 60\ninstances:\n"
        "  - {id: a, model: sentinel-2a, nu_deg: 0}\n"
        "  - {id: a, model: landsat-9, nu_deg: 10}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="повторяется"):
        load_scenario_v2(path)


def test_telemetry_has_no_correct_action_and_noise_changes_the_reading():
    world = MissionWorld(
        {"a": make_engine("cubesat-6u", with_stations=False)},
        epoch="2026-10-20T00:00:00Z",
        horizon_sec=60,
    )
    world.engines["a"].preload_bits(1.0e6)
    truth = world.observe("a")
    noisy = world.observe("a", noise=0.5, rng=np.random.default_rng(1))
    assert "action" not in truth
    assert truth["stored_bits"] == 1.0e6
    assert noisy["stored_bits"] != truth["stored_bits"]


def test_snapshot_then_step_matches_the_uninterrupted_run():
    left = MissionWorld(
        {"a": make_engine("cubesat-6u", with_stations=False)},
        epoch="2026-10-20T00:00:00Z",
        horizon_sec=60,
    )
    right = MissionWorld(
        {"a": make_engine("cubesat-6u", with_stations=False)},
        epoch="2026-10-20T00:00:00Z",
        horizon_sec=60,
    )
    left.step(10.0)
    right.restore(left.snapshot())
    left.step(10.0)
    right.step(10.0)
    assert np.allclose(left.engines["a"].orbit.r, right.engines["a"].orbit.r)
    assert left.engines["a"].time_sec == pytest.approx(right.engines["a"].time_sec)


def test_forecast_does_not_move_the_world_and_estimate_follows_the_seed():
    left = MissionWorld(
        {"a": make_engine("cubesat-6u", with_stations=False)},
        epoch="2026-10-20T00:00:00Z",
        horizon_sec=60,
        seed=7,
    )
    right = MissionWorld(
        {"a": make_engine("cubesat-6u", with_stations=False)},
        epoch="2026-10-20T00:00:00Z",
        horizon_sec=60,
        seed=7,
    )
    left.engines["a"].preload_bits(1.0e6)
    right.engines["a"].preload_bits(1.0e6)
    ahead = left.forecast("a", 10.0)
    assert left.engines["a"].time_sec == 0.0
    assert ahead["kind"] == "forecast"
    assert ahead["time_sec"] == pytest.approx(10.0)
    assert "action" not in left.truth("a")
    assert left.estimate("a", noise=0.5)["stored_bits"] == right.estimate("a", noise=0.5)["stored_bits"]
    left.step(10.0)
    right.step(10.0)
    assert np.allclose(left.engines["a"].orbit.r, right.engines["a"].orbit.r)


def test_sixteen_copies_of_one_model_keep_separate_names(tmp_path: Path):
    rows = "\n".join(
        f"  - {{id: bird-{index}, model: sentinel-2a, nu_deg: {index}}}" for index in range(16)
    )
    path = tmp_path / "sixteen.yaml"
    path.write_text(
        "version: 2\nepoch: '2026-10-20T10:00:00Z'\nhorizon_sec: 60\ninstances:\n" + rows + "\n",
        encoding="utf-8",
    )
    world = world_from_scenario(path)
    assert len(world.engines) == 16
    assert len({engine.spacecraft.config.name for engine in world.engines.values()}) == 16
    assert {item["model"] for item in load_scenario_v2(path)["instances"]} == {"sentinel-2a"}


def test_comments_and_key_order_do_not_change_the_semantic_hash(tmp_path: Path):
    from space_env.mission.scenario_v2 import semantic_hash

    body = "version: 2\nhorizon_sec: 60\ninstances:\n- id: a\n  model: cubesat-6u\nstations: []\nrequests: []\n"
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text("# заметка\n" + body, encoding="utf-8")
    right.write_text(
        "requests: []\nstations: []\ninstances:\n- model: cubesat-6u\n  id: a\nhorizon_sec: 60\nversion: 2\n",
        encoding="utf-8",
    )
    assert semantic_hash(load_scenario_v2(left)) == semantic_hash(load_scenario_v2(right))
    assert scenario_manifest(left)["sha256"] != scenario_manifest(right)["sha256"]
    assert scenario_manifest(left)["semantic_sha256"] == scenario_manifest(right)["semantic_sha256"]
