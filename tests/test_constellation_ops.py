"""Пакет доходит по цепочке до аппарата с наибольшим углом места над Фэрбанксом."""
import gymnasium as gym
import numpy as np
import pytest

import space_env.envs


def test_packet_reaches_the_gateway_slot_one_hop_at_a_time():
    env = gym.make("space_env/ConstellationOps-v0", max_episode_steps=8)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4 * 4 + 2,)
    names = list(env.unwrapped._names)
    gateway = names.index(info["gateway"])
    holder = names.index(info["source"])
    assert holder != gateway
    for _ in range(env.unwrapped.count):
        holder = names.index(max(env.unwrapped._fleet.buffer_mb, key=env.unwrapped._fleet.buffer_mb.get))
        if holder == gateway:
            break
        _obs, reward, _term, _trunc, step = env.step(np.array([holder, 2]))
        assert step["route"], "луч к соседу должен открыться на шаге 1.5°"
        assert len(step["route"]) == 1
        if step["route"][0]["to"] == info["gateway"]:
            assert reward > 0.0
    assert env.unwrapped._fleet.buffer_mb[info["gateway"]] >= 8.0 - 1e-6
    assert env.unwrapped._steps >= 1
    env.close()


def test_expert_rollout_and_episode_frames_carry_the_hop():
    from space_env.data_generators.sft import generate_sft_from_constellation
    from space_env.rl.expert import expert_action
    from space_env.rl.trajectories import rollout_episode
    from space_env.viz.export import export_constellation_ops

    env = gym.make("space_env/ConstellationOps-v0", max_episode_steps=8)
    rows = rollout_episode(env, seed=0, policy=lambda obs: expert_action("space_env/ConstellationOps-v0", obs))
    env.close()
    assert isinstance(rows[0]["action"], list)
    assert any(row["success"] for row in rows)
    records = list(generate_sft_from_constellation())
    text = " ".join(item["instruction"] + item["output"] for item in records)
    assert "Фэрбанкс" in text
    assert records[0]["meta"]["gateway"] in text
    document = export_constellation_ops()
    assert document["memory_capacity_bits"] == 8.0e6
    assert any(frame["route"] for frame in document["frames"])
    last = document["frames"][-1]
    assert sum(value or 0.0 for value in (vehicle["memory_bits"] for vehicle in last["vehicles"].values())) == pytest.approx(8.0e6)
