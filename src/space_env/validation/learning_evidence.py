"""Paired held-out agent comparison; native replay does not attest model training."""

from __future__ import annotations

import math

from pydantic import Field

from space_env.llm.episode import verify_record
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class LearningDisclosure(StrictModel):
    training_groups: tuple[str, ...] = Field(min_length=1, max_length=10000)
    training_dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    before_weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    after_weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    before_inference_trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    after_inference_trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def paired_statistics(before, after):
    """Exact two-sided McNemar/binomial test on discordant independent groups.

    One binary outcome per group. No claim of power from repeated same-task seeds.
    """
    if not before or set(before) != set(after) or len(before) > 128:
        raise ValueError("paired group matrix must match, with 1..128 groups")
    if any(type(v) is not bool for v in [*before.values(), *after.values()]):
        raise ValueError("outcomes must be boolean")
    improved = sum(not before[k] and after[k] for k in before)
    regressed = sum(before[k] and not after[k] for k in before)
    discordant = improved + regressed
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, i) for i in range(min(improved, regressed) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    n = len(before)
    return {
        "groups": n,
        "before_successes": sum(before.values()),
        "after_successes": sum(after.values()),
        "improved": improved,
        "regressed": regressed,
        "tied": n - discordant,
        "success_rate_difference": (improved - regressed) / n,
        "exact_two_sided_mcnemar_p": p_value,
        "alpha": 0.05,
        "minimum_groups": 20,
        "paired_improvement_detected": n >= 20
        and improved > regressed
        and p_value < 0.05,
        "assumption": "independent, predeclared held-out groups; label separation alone does not establish independence",
    }


def learning_comparison(before_records, after_records, disclosure):
    declared = LearningDisclosure.model_validate(disclosure)
    if declared.before_weights_sha256 == declared.after_weights_sha256:
        raise ValueError("before/after weight identities must differ")

    def index(records):
        if not 1 <= len(records) <= 128:
            raise ValueError("1..128 held-out records required")
        result = {}
        for record in records:
            verify_record(record)
            config = record["configuration"]
            group = config["group"]
            if config["split"] != "test" or config["policy_origin"] != "external_llm":
                raise ValueError(
                    "only held-out external LLM records accepted, never scripted demos"
                )
            if group in result or group in declared.training_groups:
                raise ValueError("duplicate or train/test leaking group")
            result[group] = record
        return result

    before, after = index(before_records), index(after_records)
    if set(before) != set(after):
        raise ValueError("before/after held-out groups differ")
    identities = set()
    for group, record in before.items():
        other = after[group]
        if (
            record["initial_checkpoint"] != other["initial_checkpoint"]
            or record["tools"] != other["tools"]
            or record["configuration"] != other["configuration"]
        ):
            raise ValueError(
                "paired prompts, budgets, tools and initial states must match"
            )
        identity = record["initial_checkpoint"]["sha256"]
        if identity in identities:
            raise ValueError("same initial task relabelled as independent groups")
        identities.add(identity)
    statistics = paired_statistics(
        {g: r["metrics"]["success"] for g, r in before.items()},
        {g: r["metrics"]["success"] for g, r in after.items()},
    )
    return {
        "schema": "space-env.learning-comparison/1",
        "statistics": statistics,
        "disclosure": declared.model_dump(mode="json"),
        "records": {
            g: {"before": before[g]["sha256"], "after": after[g]["sha256"]}
            for g in sorted(before)
        },
        "native_replay_verified": True,
        "training_proven": False,
        "scope": "Paired native outcomes only. Model, training-data and inference hashes are submitted declarations; inference/training authenticity and causal benefit are not attested.",
        "sha256_inputs": digest({"before": before_records, "after": after_records}),
    }
