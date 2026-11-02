"""Bounded role-bound structured tools; raw runtime/control artifacts stay private."""

from __future__ import annotations

import copy


def step_schema():
    return {
        "name": "space_env_task_step",
        "description": "Apply declared task action IDs and advance physical mission time once.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_ids", "expected_revision", "idempotency_key"],
            "properties": {
                "action_ids": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {"type": "string"},
                },
                "expected_revision": {"type": "integer", "minimum": 0},
                "idempotency_key": {"type": "string", "minLength": 1},
                "advance_sec": {"type": "number", "exclusiveMinimum": 1e-9},
            },
        },
    }


class TaskTools:
    def __init__(self, session, role="@central"):
        if role not in {"@central", *session.describe()["roles"]}:
            raise ValueError("unknown role")
        self.__session = session
        self.__role = role

    def observe(self):
        return self.__session.observe(self.__role)

    def schema(self):
        return copy.deepcopy(step_schema())

    def joint_schema(self):
        if self.__role != "@central":
            raise PermissionError("joint control requires the central controller")
        return joint_step_schema()

    def joint_step(
        self, choices, *, expected_revision, idempotency_key, advance_sec=None
    ):
        if self.__role != "@central":
            raise PermissionError("joint control requires the central controller")
        r = self.__session.step(
            choices,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            advance_sec=advance_sec,
        )
        return {
            k: r[k]
            for k in [
                "revision",
                "elapsed_sec",
                "reward",
                "discount",
                "terminated",
                "truncated",
                "feedback",
            ]
        } | {"observation": r["next_observation"]}

    def step(self, action_ids, *, expected_revision, idempotency_key, advance_sec=None):
        r = self.__session.step(
            {self.__role: action_ids},
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            advance_sec=advance_sec,
        )
        feedback = r["feedback"]
        if feedback["role"] not in {None, self.__role} and self.__role != "@central":
            feedback = {"accepted": False, "code": "joint_rejected"}
        return {
            k: r[k]
            for k in [
                "revision",
                "elapsed_sec",
                "reward",
                "discount",
                "terminated",
                "truncated",
            ]
        } | {
            "feedback": feedback,
            "observation": r["next_role_observations"][self.__role],
        }


def joint_step_schema():
    from space_env.tasks.schema import TaskDecision

    return {
        "name": "space_env_task_joint_step",
        "description": "Apply one synchronized joint task decision.",
        "parameters": TaskDecision.model_json_schema(),
    }
