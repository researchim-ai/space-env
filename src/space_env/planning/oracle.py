"""Trainer-only exhaustive tiny masked action-tree oracle. Never a policy service."""

from __future__ import annotations

import itertools

from space_env.tasks import TaskSession


def tiny_oracle(episode, *, depth=2, max_nodes=64, max_candidates=6):
    for value, bound in [(depth, 6), (max_nodes, 4096), (max_candidates, 8)]:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= bound
        ):
            raise ValueError("tiny oracle budget outside declared bounds")
    root = TaskSession.from_checkpoint(episode)
    best = {"reward": float("-inf"), "actions": []}
    nodes, exhausted = 0, False

    def visit(session, level, reward, path):
        nonlocal nodes, exhausted, best
        if nodes >= max_nodes:
            exhausted = True
            return
        nodes += 1
        if level == depth or session._done:
            if reward > best["reward"]:
                best = {"reward": reward, "actions": path}
            return
        obs = session.observe()
        ids = sorted(
            a["id"] for a in obs["catalogue"] if a["enabled"] and a["id"] != "wait"
        )
        if len(ids) > max_candidates:
            raise ValueError("oracle candidate budget exceeded; no silent pruning")
        checkpoint = session.checkpoint()
        for n in range(min(len(ids), session._task.max_joint_actions) + 1):
            for choices in itertools.combinations(ids, n):
                if nodes >= max_nodes:
                    exhausted = True
                    return
                child = TaskSession.from_checkpoint(checkpoint)
                result = child.step(
                    {"@central": choices or ["wait"]},
                    expected_revision=child.revision,
                    idempotency_key=f"oracle/{level}",
                )
                visit(
                    child,
                    level + 1,
                    reward + result["reward"],
                    path + [list(choices) or ["wait"]],
                )

    visit(root, 0, 0.0, [])
    return {
        "schema": "space-env.tiny-oracle/1",
        "nodes": nodes,
        "budget_exhausted": exhausted,
        "proven_optimal_in_masked_tree": not exhausted,
        "best_reward": None if best["reward"] == float("-inf") else best["reward"],
        "actions": best["actions"],
        "depth": depth,
        "scope": "finite depth, fixed decision interval, current-mask action tree only; not continuous/full-mission optimality",
    }
