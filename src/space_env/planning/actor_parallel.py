"""Optional PettingZoo ParallelEnv; actor-only spaces and shared team reward."""

import copy
from typing import ClassVar

try:
    from pettingzoo import ParallelEnv
except ImportError as exc:
    raise ImportError("Install space-env[marl] for ActorParallelEnv") from exc

from space_env.planning.actor_gym import ActorMissionEnv


class ActorParallelEnv(ParallelEnv):
    metadata: ClassVar = {
        "render_modes": [],
        "name": "space_env_actor_policy_v1",
        "is_parallelizable": True,
    }

    def __init__(self, reference):
        self.joint = ActorMissionEnv(reference)
        self.possible_agents = list(self.joint.roles)
        self.agents = []
        self.observation_spaces = self.joint.observation_space.spaces
        self.action_spaces = self.joint.action_space.spaces

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, info = self.joint.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        return obs, {
            a: {
                "binding": info["bindings"][a],
                "catalogue": info["catalogues"][a],
                "reward_scope": "joint_team",
                "physical_episode_seed": info["physical_episode_seed"],
            }
            for a in self.agents
        }

    def step(self, actions):
        if not self.agents:
            if actions:
                raise ValueError("parallel actor episode is over")
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("every live actor role must decide")
        before = self.agents[:]
        obs, reward, terminated, truncated, info = self.joint.step(actions)
        if terminated or truncated:
            self.agents = []
        return (
            obs,
            {a: reward for a in before},
            {a: terminated for a in before},
            {a: truncated for a in before},
            {
                a: {
                    "feedback": copy.deepcopy(info["feedback"]),
                    "binding": info["bindings"][a],
                    "reward_scope": "joint_team",
                    "discount": info["discount"],
                }
                for a in before
            },
        )
