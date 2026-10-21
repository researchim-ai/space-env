"""Пять дорожек, перенос сдачи и три постановки бенчмарка."""
import gymnasium as gym

from space_env.eval.tracks import TRACKS, run_benchmark_tracks
from space_env.eval.transfer import acquisition_action, transfer_dump_report
from space_env.rl.expert import expert_action


def _run(env_id: str, steps: int):
    env = gym.make(env_id, max_episode_steps=steps)
    obs, info = env.reset(seed=0)
    done = False
    last = info
    while not done:
        obs, _reward, terminated, truncated, last = env.step(expert_action(env_id, obs))
        done = bool(terminated or truncated)
    env.close()
    return last


def test_station_order_and_sentinel_fault():
    station = _run("space_env/StationContention-v0", 1)
    assert station["success"] and station["delivered_mbit"] == 8.0
    orders = _run("space_env/OrderBook-v0", 1)
    assert orders["success"]
    if orders["sunlit"]:
        assert orders["stored"] == ["HIGH-0", "HIGH-1"]
    fault = _run("space_env/SentinelPassFault-v0", 12)
    assert fault["radio_ready"]
    assert fault["delivered_bits"] > 0.0
    assert fault["success"]


def test_five_tracks_split_train_and_held_out_seeds():
    rows = run_benchmark_tracks(seed=0, held_out_seed=100)
    assert [row["track"] for row in rows] == [item[0] for item in TRACKS]
    for row in rows:
        assert row["train_expert_success"] == 1.0
        assert row["held_expert_success"] == 1.0
    constellation = next(row for row in rows if row["track"] == "constellation")
    assert "learned" in constellation
    assert isinstance(constellation["learned"], bool)


def test_one_dump_policy_runs_on_three_buses():
    assert acquisition_action([0, 0, 1, 0.1, 0, 0, 1]) == 2
    assert acquisition_action([0, 0, 0, 0, 0, 0, 0.2]) == 1
    rows = transfer_dump_report(max_episode_steps=12, seed=0)
    assert [row["vehicle"] for row in rows] == ["sentinel-2a", "landsat-9", "cubesat-6u"]
    assert all(row["delivered_bits"] >= 0.0 for row in rows)
    named = {row["vehicle"]: row["delivered_bits"] for row in rows}
    assert named["sentinel-2a"] > 0.0
