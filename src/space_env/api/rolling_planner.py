"""Causal rolling policy state, bound to native history rather than predictions."""

from __future__ import annotations

import copy

from space_env.planning.model import make_input
from space_env.planning.policy import BaselinePolicy
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession

CONFIG = {
    "name": "route_aware_rolling",
    "max_candidates": 64,
    "max_nodes": 256,
    "reclaim_acked_downlinks": True,
    "cooldown_steps": 4,
}


def _feedback(session):
    return (
        session._transitions[-1]["transition"]["feedback"]
        if session._transitions
        else None
    )


def _actual_choice(policy, request):
    # A failed MANUAL choice must not penalize an unexecuted policy proposal.
    policy._last_decision["action_ids"] = sorted(
        a for ids in request["choices"].values() for a in ids
    )


def rebuild(session, *, proposed=False):
    """Bounded bootstrap/recovery; only previously executed native prefix replay."""
    probe = TaskSession(session._original, session._task, seed=session._seed)
    policy = BaselinePolicy(CONFIG)
    for row in session._transitions:
        policy.decide(make_input(probe, _feedback(probe), conditional_claims=True))
        _actual_choice(policy, row["request"])
        result = probe.step(**row["request"])
        if result["feedback"] != row["transition"]["feedback"]:
            raise ValueError("rolling planner history replay changed feedback")
    if proposed:
        policy.decide(make_input(probe, _feedback(probe), conditional_claims=True))
    return policy


def stage(operator):
    """Return a fresh policy and proposal without changing the operator."""
    state = operator.planner_state
    policy = (
        rebuild(operator.session)
        if state is None
        else BaselinePolicy.from_checkpoint(state["policy"])
    )
    decision = policy.decide(
        make_input(
            operator.session, _feedback(operator.session), conditional_claims=True
        )
    )
    return policy, {
        **decision,
        "expected_revision": operator.session.revision,
        "planner": "causal_rolling",
        "information": "current public observations and static resource claims; no forecast or future rollout",
    }


def save(operator, policy, *, executed=False, request=None):
    if executed:
        _actual_choice(policy, request)
    operator.planner_state = {
        "schema": "space-env.operator-rolling/1",
        "task_sha256": digest(
            operator.session._task.model_dump(mode="json", by_alias=True)
        ),
        "executed": executed,
        "policy": copy.deepcopy(policy.checkpoint()),
    }


def validate_shape(value, checkpoint):
    if value is None:
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "task_sha256", "executed", "policy"}
        or value["schema"] != "space-env.operator-rolling/1"
        or type(value["executed"]) is not bool
    ):
        raise ValueError("invalid rolling planner state")
    policy = BaselinePolicy.from_checkpoint(value["policy"])
    n = len(checkpoint["transitions"])
    expected_revision = n - 1 if value["executed"] else n
    if (
        expected_revision < 0
        or policy._revision != expected_revision
        or policy.config.model_dump(mode="json")
        != BaselinePolicy(CONFIG).config.model_dump(mode="json")
        or value["task_sha256"] != digest(checkpoint["task"])
    ):
        raise ValueError("rolling planner state/history binding mismatch")


def validate_history(operator, value):
    if value is None:
        return
    policy = rebuild(operator.session, proposed=not value["executed"])
    if policy.checkpoint() != value["policy"]:
        raise ValueError("rolling planner state differs from native history")
