"""Промышленные постановки: дистанция, приоритет, окно с отказом, фрагмент, увод."""
import gymnasium as gym

from space_env.catalog.references import reference
from space_env.rl.expert import expert_action


def _run(env_id: str, steps: int = 8):
    env = gym.make(env_id, max_episode_steps=steps)
    obs, info = env.reset(seed=0)
    done = False
    last = info
    while not done:
        obs, _reward, terminated, truncated, last = env.step(expert_action(env_id, obs))
        done = bool(terminated or truncated)
    env.close()
    return last


def test_new_classes_are_labeled_as_assumptions():
    cubesat = reference("cubesat-6u")
    debris = reference("debris-fragment")
    assert cubesat.norad_id is None and cubesat.assumptions
    assert debris.spacecraft_config().propulsion.delta_v_total_m_s == 0.0


def test_rendezvous_disposal_and_coverage_follow_the_expert():
    held = _run("space_env/RendezvousHold-v0", 1)
    assert held["success"] and abs(held["rate_m_s"]) < 0.05
    covered = _run("space_env/PriorityCoverage-v0", 1)
    assert covered["success"]
    if covered["sunlit"]:
        assert covered["stored"] == "P1"
    lowered = _run("space_env/Disposal-v0", 1)
    assert lowered["success"] and lowered["drop_m"] > 10.0


def test_pass_fault_recovers_inside_the_window():
    done = _run("space_env/PassFault-v0", 6)
    assert done["success"] and done["delivered"] and done["radio_ready"]


def test_debris_burn_is_rejected_and_own_burn_lowers_pc():
    avoided = _run("space_env/DebrisAvoid-v0", 1)
    assert avoided["success"] and avoided["pc"] < 1.0e-4
    env = gym.make("space_env/DebrisAvoid-v0")
    _obs, _info = env.reset(seed=0)
    _obs, _reward, _term, _trunc, info = env.step(2)
    assert info["debris_rejected"] is True
    assert info["success"] is False
    env.close()
