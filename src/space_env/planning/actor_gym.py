"""Gymnasium joint dictionary adapter over the role-only live coordinator."""

import copy
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from space_env.planning.actor_online import ActorCoordinator
from space_env.tasks.env import encode, observation_space


class ActorMissionEnv(gym.Env):
    metadata: ClassVar = {"render_modes": []}

    def __init__(self, reference):
        self._reference = copy.deepcopy(reference)
        self.coordinator = ActorCoordinator(reference)
        self.roles = self.coordinator.roles
        self.action_ids = self.coordinator.catalogues
        self._spec = self.coordinator._spec
        self.action_space = spaces.Dict(
            {
                r: spaces.Discrete(len(self.action_ids[r]))
                if self._spec.roles[r].action_slots == 1
                else spaces.MultiDiscrete(
                    [len(self.action_ids[r])] * self._spec.roles[r].action_slots
                )
                for r in self.roles
            }
        )
        observations = {r: observation_space(self._spec) for r in self.roles}
        for role in self.roles:
            observations[role].spaces["action_mask"] = spaces.MultiBinary(
                len(self.action_ids[role])
            )
        self.observation_space = spaces.Dict(observations)

    def _observations(self, values):
        result = {r: encode(values[r], self._spec) for r in self.roles}
        for role in self.roles:
            result[role]["action_mask"] = result[role]["action_mask"][
                : len(self.action_ids[role])
            ].copy()
        return result

    def reset(self, *, seed=None, options=None):
        if isinstance(seed, bool):
            raise TypeError("Gym sampling seed cannot be boolean")
        if options is not None and not isinstance(options, dict):
            raise TypeError("actor reset options must be metadata mapping")
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.coordinator = ActorCoordinator(self._reference)
        values = self.coordinator.observe()
        return self._observations(values), {
            "catalogues": copy.deepcopy(self.action_ids),
            "bindings": {r: v["binding"] for r, v in values.items()},
            "physical_episode_seed": self.coordinator._plan["episode_seed"],
            "seed_scope": "fixed verified physical epoch; seed controls wrapper/action sampling only",
            "reward_scope": "joint_team",
            "ignored_options": sorted(map(str, options or {})),
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid actor Gym joint action")
        choices = {}
        for role in self.roles:
            values = [
                self.action_ids[role][int(i)] for i in np.atleast_1d(action[role])
            ]
            nonwait = list(dict.fromkeys(a for a in values if a != "wait"))
            choices[role] = nonwait or ["wait"]
        current = self.coordinator.observe()
        response = self.coordinator.step(
            choices,
            expected_bindings={r: v["binding"] for r, v in current.items()},
            idempotency_key=f"gym/{self.coordinator.cursor}",
        )
        return (
            self._observations(response["observations"]),
            response["reward"],
            response["terminated"],
            response["truncated"],
            {
                "feedback": response["feedback"],
                "bindings": {
                    r: v["binding"] for r, v in response["observations"].items()
                },
                "reward_scope": "joint_team",
                "discount": response["discount"],
                "elapsed_sec": response["elapsed_sec"],
            },
        )
