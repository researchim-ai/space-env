"""Gymnasium adapter: all actions and physical transitions use TaskSession."""

from __future__ import annotations

import copy
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from space_env.tasks.observation import FEATURES, KINDS, RELATIONS
from space_env.tasks.schema import TaskSpec
from space_env.tasks.session import TaskSession


def observation_space(spec):
    n, e, a = spec.max_entities, spec.max_relations, spec.max_actions
    return spaces.Dict(
        {
            "clock": spaces.Box(0.0, 259200.0, (1,), dtype=np.float64),
            "entities": spaces.Box(-1e300, 1e300, (n, len(FEATURES)), dtype=np.float64),
            "entity_mask": spaces.MultiBinary(n),
            "feature_mask": spaces.MultiBinary((n, len(FEATURES))),
            "entity_kind": spaces.Box(0, len(KINDS), (n,), dtype=np.int16),
            "relations": spaces.Box(0, max(n, len(RELATIONS)), (e, 3), dtype=np.int32),
            "relation_mask": spaces.MultiBinary(e),
            "action_mask": spaces.MultiBinary(a),
        }
    )


def encode(view, spec):
    n, e, a = spec.max_entities, spec.max_relations, spec.max_actions
    out = {
        "clock": np.array([view["time_sec"]], dtype=np.float64),
        "entities": np.zeros((n, len(FEATURES)), dtype=np.float64),
        "entity_mask": np.zeros(n, dtype=np.int8),
        "feature_mask": np.zeros((n, len(FEATURES)), dtype=np.int8),
        "entity_kind": np.zeros(n, dtype=np.int16),
        "relations": np.zeros((e, 3), dtype=np.int32),
        "relation_mask": np.zeros(e, dtype=np.int8),
        "action_mask": np.zeros(a, dtype=np.int8),
    }
    for row in view["entities"]:
        i = row["index"]
        out["entity_mask"][i] = 1
        out["entity_kind"][i] = KINDS[row["kind"]]
        for field, value in row["values"].items():
            j = FEATURES.index(field)
            out["entities"][i, j] = value
            out["feature_mask"][i, j] = 1
    for i, row in enumerate(view["relations"]):
        out["relations"][i] = [
            row["source"],
            row["destination"],
            RELATIONS[row["kind"]],
        ]
        out["relation_mask"][i] = 1
    out["action_mask"][: len(view["action_mask"])] = view["action_mask"]
    return out


class MissionTaskEnv(gym.Env):
    metadata: ClassVar = {"render_modes": ["ansi"]}

    def __init__(self, scenario, task, render_mode=None, variable_duration=False):
        if render_mode not in {None, "ansi"}:
            raise ValueError("unsupported task render mode")
        self.render_mode = render_mode
        self._scenario = copy.deepcopy(scenario)
        self._task = TaskSpec.model_validate(copy.deepcopy(task))
        self.variable_duration = bool(variable_duration)
        self.durations = list(self._task.durations_sec or (self._task.decision_sec,))
        self.session = None
        self.action_ids = ["wait", *[a.id for a in self._task.actions]]
        self.action_space = spaces.MultiDiscrete(
            [len(self.action_ids)] * self._task.max_joint_actions
        )
        if self.variable_duration:
            self.action_space = spaces.Dict(
                {
                    "choices": self.action_space,
                    "duration": spaces.Discrete(len(self.durations)),
                }
            )
        self.observation_space = observation_space(self._task)

    def reset(self, *, seed=None, options=None):
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a metadata mapping")
        if isinstance(seed, bool):
            raise TypeError("seed cannot be boolean")
        super().reset(seed=seed)
        episode_seed = (
            (
                int(self.np_random.integers(0, 2**63 - 1))
                if self._task.seed_policy == "derive-network"
                else self._scenario.get("seed", 0)
            )
            if seed is None
            else int(seed)
        )
        self.action_space.seed(seed)
        self.session = TaskSession(
            self._scenario,
            self._task.model_dump(mode="json", by_alias=True),
            seed=episode_seed,
        )
        return encode(self.session.observe(), self._task), {
            "schema": "space-env.task-info/1",
            "seed_policy": self._task.seed_policy,
            "ignored_options": sorted(map(str, options or {})),
        }

    def step(self, action):
        if self.session is None:
            raise RuntimeError("reset is required")
        if not self.action_space.contains(action):
            raise ValueError("invalid task joint action")
        raw = action["choices"] if self.variable_duration else action
        duration = (
            self.durations[int(action["duration"])] if self.variable_duration else None
        )
        choices = [self.action_ids[int(i)] for i in raw]
        r = self.session.step(
            {"@central": choices},
            expected_revision=self.session.revision,
            idempotency_key=f"gym/{self.session.revision}",
            advance_sec=duration,
        )
        return (
            encode(r["next_observation"], self._task),
            r["reward"],
            r["terminated"],
            r["truncated"],
            {
                "elapsed_sec": r["elapsed_sec"],
                "discount": r["discount"],
                "feedback": r["feedback"],
                "reward_components": r["reward_components"],
            },
        )

    def action_masks(self):
        if self.session is None:
            raise RuntimeError("reset is required")
        mask = np.asarray(self.session.observe()["action_mask"], dtype=np.int8)
        choices = tuple(mask.copy() for _ in range(self._task.max_joint_actions))
        return (
            {
                "choices": choices,
                "duration": np.ones(len(self.durations), dtype=np.int8),
            }
            if self.variable_duration
            else choices
        )

    def render(self):
        if self.session is None:
            return None
        view = self.session.observe()
        return f"{view['task_id']} t={view['time_sec']:.6f}s revision={view['revision']} entities={len(view['entities'])}"
