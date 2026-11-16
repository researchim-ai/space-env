"""Opt-in task acceptance, separate from physical all-goals mission success."""

from __future__ import annotations

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel


class TaskAcceptanceSpec(StrictModel):
    minimum_fraction: float = Field(gt=0, le=1, strict=True)
    mandatory: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.mandatory)) != len(self.mandatory) or any(
            not name.strip() for name in self.mandatory
        ):
            raise ValueError("mandatory condition IDs must be unique and nonblank")
        return self


def validate_acceptance(spec, weights):
    if set(spec.mandatory) - set(weights):
        raise ValueError("mandatory conditions must be declared goal IDs")


def evaluate_acceptance(spec, status, utility, *, finished, truncated=False):
    met = {row["id"] for row in status["conditions"] if row["met"]}
    missing = sorted(set(spec.mandatory) - met)
    safe = not utility["violations"] and utility["score"] >= 0
    threshold = utility["completion_fraction"] >= spec.minimum_fraction
    natural = finished and not truncated
    return {
        "accepted": bool(natural and safe and threshold and not missing),
        "natural_terminal": bool(natural),
        "pending": not finished,
        "native_truncated": bool(truncated),
        "minimum_fraction": spec.minimum_fraction,
        "completion_fraction": utility["completion_fraction"],
        "threshold_met": threshold,
        "mandatory_met": not missing,
        "missing_mandatory": missing,
        "safety_valid": safe,
    }


def acceptance_bound(spec, weights, dispatch_bound):
    """Sufficient contradiction under current known single-epoch dispatch rules."""
    if dispatch_bound is None:
        return None
    count_limit = dispatch_bound["maximum_completed_conditions"]
    if type(count_limit) is not int or not 0 <= count_limit <= len(weights):
        raise ValueError("invalid published dispatch cardinality bound")
    count = len(spec.mandatory)
    upper = None
    if count <= count_limit:
        mandatory = set(spec.mandatory)
        selected = [weights[name] for name in mandatory]
        selected += sorted(
            (value for name, value in weights.items() if name not in mandatory),
            reverse=True,
        )[: count_limit - count]
        upper = sum(selected) / sum(weights.values())
    reason = (
        "mandatory_count_exceeds_dispatch_bound"
        if count > count_limit
        else "threshold_exceeds_dispatch_bound"
        if upper < spec.minimum_fraction
        else None
    )
    return {
        "status": "infeasible_under_current_dispatch_rules"
        if reason
        else "not_ruled_out",
        "reason": reason,
        "maximum_fraction_given_mandatory": upper,
        "scope": "Current published single-epoch dispatch/template rules only. Future released updates may change this bound; noncontradiction is not physical feasibility.",
    }
