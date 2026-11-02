"""Optional PettingZoo ParallelEnv over the same atomic task decision."""

from __future__ import annotations

import copy
from typing import ClassVar

import numpy as np
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError as exc:
    raise ImportError(
        "Install space-env[marl] for the PettingZoo task adapter"
    ) from exc

from space_env.tasks.env import encode, observation_space
from space_env.tasks.schema import TaskSpec
from space_env.tasks.session import TaskSession


class MissionParallelEnv(ParallelEnv):
    metadata: ClassVar = {
        "render_modes": [],
        "name": "space_env_runtime_task_v1",
        "is_parallelizable": True,
    }

    def __init__(self, scenario, task, variable_duration=False):
        self._scenario = copy.deepcopy(scenario)
        self._task = TaskSpec.model_validate(copy.deepcopy(task))
        self.variable_duration = bool(variable_duration)
        self.durations = list(self._task.durations_sec or (self._task.decision_sec,))
        self.possible_agents = sorted(self._task.roles)
        self.agents = []
        self.action_ids = ["wait", *[a.id for a in self._task.actions]]
        self.observation_spaces = {}
        for a in self.possible_agents:
            base = observation_space(self._task)
            count = len(self.action_ids)
            slots = self._task.roles[a].action_slots
            base.spaces["action_mask"] = (
                spaces.MultiBinary(count)
                if slots == 1
                else spaces.Tuple(
                    tuple(spaces.MultiBinary(count) for _ in range(slots))
                )
            )
            self.observation_spaces[a] = base
        self.action_spaces = {
            a: (
                spaces.Discrete(len(self.action_ids))
                if self._task.roles[a].action_slots == 1
                else spaces.MultiDiscrete(
                    [len(self.action_ids)] * self._task.roles[a].action_slots
                )
            )
            for a in self.possible_agents
        }
        if self.variable_duration:
            for a in self.possible_agents:
                self.action_spaces[a] = spaces.Dict(
                    {
                        "choices": self.action_spaces[a],
                        "duration": spaces.Discrete(len(self.durations)),
                    }
                )
                self.observation_spaces[a].spaces["action_mask"] = spaces.Dict(
                    {
                        "choices": self.observation_spaces[a].spaces["action_mask"],
                        "duration": spaces.MultiBinary(len(self.durations)),
                    }
                )
        self.session = None
        self._rng = np.random.default_rng()

    def _encode(self, view, agent):
        result = encode(view, self._task)
        mask = result["action_mask"][: len(self.action_ids)].copy()
        slots = self._task.roles[agent].action_slots
        result["action_mask"] = (
            mask if slots == 1 else tuple(mask.copy() for _ in range(slots))
        )
        if self.variable_duration:
            result["action_mask"] = {
                "choices": result["action_mask"],
                "duration": np.ones(len(self.durations), dtype=np.int8),
            }
        return result

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a metadata mapping")
        if isinstance(seed, bool):
            raise TypeError("seed cannot be boolean")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        episode_seed = (
            (
                int(self._rng.integers(0, 2**63 - 1))
                if self._task.seed_policy == "derive-network"
                else self._scenario.get("seed", 0)
            )
            if seed is None
            else int(seed)
        )
        self.session = TaskSession(
            self._scenario,
            self._task.model_dump(mode="json", by_alias=True),
            seed=episode_seed,
        )
        self.agents = self.possible_agents[:]
        for a in self.agents:
            self.action_spaces[a].seed(seed)
        return {a: self._encode(self.session.observe(a), a) for a in self.agents}, {
            a: {
                "seed_policy": self._task.seed_policy,
                "ignored_options": sorted(map(str, options or {})),
            }
            for a in self.agents
        }

    def step(self, actions):
        if not self.agents:
            if actions:
                raise ValueError("parallel episode is over")
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("joint task action requires exactly the live roles")
        if any(not self.action_spaces[a].contains(v) for a, v in actions.items()):
            raise ValueError("invalid role action")
        current = self.agents[:]
        choices = {
            a: [
                self.action_ids[int(i)]
                for i in np.atleast_1d(
                    actions[a]["choices"] if self.variable_duration else actions[a]
                )
            ]
            for a in current
        }
        r = self.session.step(
            choices,
            expected_revision=self.session.revision,
            idempotency_key=f"parallel/{self.session.revision}",
            advance_sec=min(
                self.durations[int(v["duration"])] for v in actions.values()
            )
            if self.variable_duration
            else None,
        )
        obs = {a: self._encode(r["next_role_observations"][a], a) for a in current}
        rewards = {
            a: r["reward"] for a in current
        }  # Identical cooperative team signal, not additive shares.
        term = {a: r["terminated"] for a in current}
        trunc = {a: r["truncated"] for a in current}
        infos = {
            a: {
                "elapsed_sec": r["elapsed_sec"],
                "discount": r["discount"],
                "feedback": (
                    r["feedback"]
                    if r["feedback"]["role"] in {None, a}
                    else {"accepted": False, "code": "joint_rejected"}
                ),
            }
            for a in current
        }
        if r["terminated"] or r["truncated"]:
            self.agents = []
        return obs, rewards, term, trunc, infos

    def state(self):
        if self.session is None:
            raise RuntimeError("reset is required")
        return encode(self.session.observe(), self._task)

    def render(self):
        return None

    def close(self):
        pass
