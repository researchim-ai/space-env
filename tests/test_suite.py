"""Общий прогон всех сред и точка входа Gymnasium."""
import math
import subprocess
import sys

from space_env.envs import registered_env_ids
from space_env.eval.benchmarks import run_registered_benchmarks
from space_env.eval.holdout import compare_policies


def test_suite_covers_every_registered_env():
    rows = run_registered_benchmarks(episodes=1, max_episode_steps=4, seed=0)
    assert [row["env"] for row in rows] == list(registered_env_ids())
    assert len(rows) >= 13
    for row in rows:
        assert row["episodes"] == 1
        assert row["mean_steps"] >= 1
        assert math.isfinite(float(row["mean_reward"]))


def test_holdout_rolls_a_multi_discrete_env():
    summary = compare_policies(
        "space_env/ConstellationOps-v0",
        episodes=1,
        max_episode_steps=3,
        seed=0,
    )
    assert 0.0 <= summary["expert_success_rate"] <= 1.0
    assert math.isfinite(summary["random_mean_reward"])


def test_entry_point_imports_the_registry():
    code = (
        "import importlib.metadata as md\n"
        "eps = md.entry_points()\n"
        "group = eps.select(group='gymnasium.envs') if hasattr(eps, 'select') else eps.get('gymnasium.envs', [])\n"
        "match = [ep for ep in group if ep.name == 'space_env']\n"
        "assert match, 'точка входа gymnasium.envs/space_env не установлена'\n"
        "match[0].load()\n"
        "import gymnasium as gym\n"
        "assert 'space_env/ConjunctionQueue-v0' in gym.registry\n"
    )
    subprocess.check_call([sys.executable, "-c", code])
