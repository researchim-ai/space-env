"""Регистрация сред, траектории, куррикулум, оценка, загрузчик сценария, TLE."""
from pathlib import Path

import gymnasium as gym
import pytest

import space_env

from space_env.eval.benchmarks import run_benchmark
from space_env.eval.sft_rubric import score_prediction
from space_env.mission.loader import load_engine
from space_env.rl.curriculum import load_curriculum
from space_env.rl.reinforce import train_linear_reinforce
from space_env.rl.rewards import load_rewards
from space_env.rl.trajectories import write_trajectories

ROOT = Path(__file__).resolve().parents[1]


def test_rewards_and_curriculum_files():
    rewards = load_rewards()
    assert rewards["mission_planner"]["goal_complete"] == pytest.approx(1.0)
    stages = load_curriculum(ROOT / "configs" / "curriculum.yaml")
    assert [stage["name"] for stage in stages] == [
        "resources",
        "anomaly",
        "conjunction",
        "planner",
        "workplan",
        "constellation",
        "conjunction_queue",
        "decay_low",
        "decay_high",
        "phase_burn",
        "reroute",
        "fleet_shift",
        "rendezvous",
        "coverage",
        "pass_fault",
        "debris",
        "disposal",
        "station",
        "orders",
        "sentinel_fault",
        "day_mission",
    ]
    assert rewards["workplan"]["task_done"] == pytest.approx(1.0)


def test_reinforce_refuses_a_multi_discrete_action():
    with pytest.raises(TypeError, match="одно дискретное действие"):
        train_linear_reinforce("space_env/ConstellationOps-v0", episodes=1, max_episode_steps=1)


def test_package_contains_configs_and_training_texts():
    root = Path(space_env.__file__).resolve().parent
    assert (root / "configs" / "curriculum.yaml").is_file()
    assert (root / "configs" / "rewards.yaml").is_file()
    assert (root / "bundled_data" / "sft_pairs.yaml").is_file()
    assert (root / "bundled_data" / "glossary.yaml").is_file()


def test_scenario_yaml_steps():
    engine = load_engine(ROOT / "configs" / "scenarios" / "leo_nominal.yaml", seed=1)
    snap, _events = engine.step(60.0)
    assert snap.time_sec == pytest.approx(60.0)
    assert 0.0 <= snap.battery_soc <= 1.0
    assert snap.contact_station == "Moscow"


def test_short_benchmark_and_rubric():
    summary = run_benchmark(
        "space_env/ResourceManager-v0",
        episodes=1,
        max_episode_steps=3,
        seed=0,
    )
    assert summary["episodes"] == 1
    assert summary["mean_steps"] >= 1
    scored = score_prediction("после AOS начинается сеанс", "AOS это захват сигнала", ["AOS", "LOS"])
    assert scored["prediction_recall"] == pytest.approx(0.5)
    assert scored["reference_recall"] == pytest.approx(0.5)


def test_trajectory_writer_and_reinforce(tmp_path):
    count = write_trajectories(
        tmp_path / "rl.jsonl",
        stages=[{"name": "anomaly", "env": "space_env/AnomalyResponse-v0", "episodes": 1, "max_episode_steps": 3}],
        seed=2,
    )
    assert count >= 1
    lines = (tmp_path / "rl.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert "anomaly" in lines[0]
    result = train_linear_reinforce(
        "space_env/ConjunctionAssessment-v0",
        episodes=3,
        max_episode_steps=4,
        seed=1,
    )
    assert len(result["returns"]) == 3
    assert isinstance(result["mean_last"], float)


def test_gym_make_all_ids():
    for env_id in (
        "space_env/MissionPlanner-v0",
        "space_env/ResourceManager-v0",
        "space_env/AnomalyResponse-v0",
        "space_env/ConjunctionAssessment-v0",
    ):
        env = gym.make(env_id, max_episode_steps=2)
        obs, _info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        env.close()


def test_tle_optional():
    pytest.importorskip("sgp4")
    from space_env.integrations import orbit_from_tle

    line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
    orbit = orbit_from_tle(line1, line2)
    assert np_isfinite_altitude(orbit)


def np_isfinite_altitude(orbit) -> bool:
    import numpy as np

    altitude = orbit.altitude_km()
    return bool(np.isfinite(altitude) and np.linalg.norm(orbit.r) > 6.0e6)


def test_optional_imports_fail_clearly_when_missing():
    import importlib.util

    if importlib.util.find_spec("spiceypy") is None:
        with pytest.raises(ImportError):
            from space_env.integrations.spice_ephem import sun_position_spice_m

            sun_position_spice_m(0.0)
    if importlib.util.find_spec("orekit_jpype") is None:
        with pytest.raises(ImportError):
            from space_env.integrations.orekit_prop import propagate_keplerian_m
            from datetime import datetime, timezone

            propagate_keplerian_m(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0, datetime(2024, 1, 1, tzinfo=timezone.utc), 60.0)
