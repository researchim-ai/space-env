"""Opt-in bounded partial-goal utility with latched declared-safety failures.

Only the native goal's declared resource/health/keepout constraints are gated.
This is reward accounting, not a collision shield or flight safety certificate.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field

from space_env.runtime.schema import StrictModel

Priority = Annotated[int, Field(ge=1, le=1000, strict=True)]


class GoalUtilitySpec(StrictModel):
    priorities: dict[str, Priority] = Field(default_factory=dict, max_length=8)


def resolve_weights(spec, conditions):
    ids = [row["id"] for row in conditions]
    if len(ids) != len(set(ids)) or not ids or set(spec.priorities) - set(ids):
        raise ValueError("utility priorities must name known unique goal conditions")
    return {name: spec.priorities.get(name, 1) for name in ids}


def evaluate_utility(status, weights, *, previous_violations=()):
    if not weights or any(
        type(w) is not int or not 1 <= w <= 1000 for w in weights.values()
    ):
        raise ValueError("invalid bounded utility weights")
    rows = status["conditions"]
    if (
        len(rows) != len(weights)
        or {r["id"] for r in rows} != set(weights)
        or any(type(r["met"]) is not bool for r in rows)
    ):
        raise ValueError("goal status does not match utility conditions")
    violations = set(previous_violations)
    for failure in status["failures"]:
        if failure == "deadline" or failure.startswith("orbit_window_missed:"):
            continue
        violations.add("native:" + failure)
    for row in status["resources"]:
        recipe = row["recipe"]
        actor = row["actor"]
        for bound, actual, minimum in (
            ("min_propellant_kg", "propellant_kg", True),
            ("max_energy_j", "energy_used_j", False),
        ):
            limit = recipe.get(bound)
            if limit is None:
                continue
            value = row.get(actual)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not math.isfinite(limit)
            ):
                violations.add("unobserved_resource:" + actor + ":" + actual)
            elif (minimum and value < limit - 1e-12) or (
                not minimum and value > limit + 1e-9
            ):
                violations.add("resource:" + actor + ":" + bound)
        if recipe.get("require_healthy") and row.get("healthy") is not True:
            violations.add("health:" + actor)
    total = math.fsum(weights.values())
    completed = math.fsum(weights[r["id"]] for r in rows if r["met"])
    fraction = completed / total
    return {
        "weighted_completed": completed,
        "total_priority": total,
        "completion_fraction": fraction,
        "score": -2.0 if violations else fraction,
        "violations": sorted(violations),
    }
