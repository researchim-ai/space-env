"""Gym JSON-text bridge sharing the same source-bound decision sessions.

Trainer outcomes live in info, never in the observation. A sampled arbitrary
string is usually an invalid tool command; invalid commands consume call budget.
"""

from __future__ import annotations

import copy
import json
import string
from typing import ClassVar

import gymnasium as gym
from gymnasium import spaces

from space_env.industrial.agent import IndustrialEpisode
from space_env.industrial.campaign import AgentCampaignSession
from space_env.industrial.native_proximity import NativeProximitySession
from space_env.industrial.native_service import NativeServiceSession
from space_env.industrial.operations import OperationsSession
from space_env.industrial.orbits import OrbitSession
from space_env.industrial.proximity import ProximitySession
from space_env.industrial.service import ServiceSession


class IndustrialEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, kind, configuration, *, max_calls=256, task_text=None):
        if (
            kind
            not in {
                "operations",
                "native_campaign",
                "proximity",
                "service",
                "native_service",
                "native_proximity",
                "native_orbits",
            }
            or type(max_calls) is not int
            or not 1 <= max_calls <= 512
        ):
            raise ValueError("unknown industrial environment or invalid budget")
        self.kind, self.configuration = kind, copy.deepcopy(configuration)
        self.max_calls = max_calls
        self.task_text = task_text
        self.action_space = spaces.Text(
            min_length=1, max_length=131072, charset=string.printable
        )
        self.observation_space = spaces.Text(
            min_length=1, max_length=4000000, charset=string.printable
        )
        self._episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("unknown reset options")
        config = copy.deepcopy(self.configuration)
        if seed is not None and self.kind == "proximity":
            config["seed"] = seed
        if seed is not None and self.kind == "native_proximity":
            for index, sensor in enumerate(config["sensors"].values()):
                sensor["seed"] = (seed + index) % (2**63)
        factory = {
            "native_campaign": AgentCampaignSession,
            "operations": OperationsSession,
            "proximity": ProximitySession,
            "service": ServiceSession,
            "native_service": NativeServiceSession,
            "native_proximity": NativeProximitySession,
            "native_orbits": OrbitSession,
        }[self.kind]
        self._episode = IndustrialEpisode(
            factory(config),
            max_calls=self.max_calls,
            task_text=self.task_text,
            split=(config.get("generation") or {}).get("split", "train")
            if self.kind == "native_orbits"
            else "train",
        )
        self._calls = 0
        return self._observation_text(), {
            "seed_changes_scenario": self.kind in {"proximity", "native_proximity"}
        }

    def _observation_text(self):
        observation = self._episode.observe()
        if self.task_text is not None:
            observation["task_text"] = self.task_text
        if self._episode._budget_penalty:
            observation["episode_budget"] = {
                "max_calls": self.max_calls,
                "remaining_calls": max(0, self.max_calls - self._calls),
                "incomplete_penalty": self._episode._budget_penalty,
            }
        return json.dumps(observation, ensure_ascii=True)

    def step(self, action):
        if self._episode is None:
            raise RuntimeError("reset first")
        if not isinstance(action, str) or not self.action_space.contains(action):
            raise ValueError("JSON text action outside declared space")
        self._calls += 1
        response = self._episode.call("gym-" + str(self._calls), action)
        terminal = self._episode._session._done and not getattr(
            self._episode._session, "_truncated", False
        )
        truncated = self._episode.done and not terminal
        return (
            self._observation_text(),
            float(response["reward"]),
            terminal,
            truncated,
            {
                "feedback": response.get("feedback", response.get("error")),
                "trainer_only": self._episode._session.metrics(),
            },
        )
