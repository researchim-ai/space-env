"""Versioned synthetic distributions over declared native mission templates."""

from __future__ import annotations

import copy
from typing import Literal

from pydantic import Field, model_validator

from space_env.planning.benchmark import TeacherGates
from space_env.planning.model import PolicyConfig
from space_env.runtime.schema import FiniteScalar, StrictModel

UNITS = {
    "product_bits_scale": "ratio",
    "memory_scale": "ratio",
    "battery_scale": "ratio",
    "rate_scale": "ratio",
    "deadline_scale": "ratio",
    "processing_scale": "ratio",
    "acquisition_scale": "ratio",
    "station_terminals": "count",
    "outage_duration_sec": "s",
    "relay_ttl_sec": "s",
}


def parameter_registry():
    return copy.deepcopy(UNITS)


class Distribution(StrictModel):
    parameter: Literal[
        "product_bits_scale",
        "memory_scale",
        "battery_scale",
        "rate_scale",
        "deadline_scale",
        "processing_scale",
        "acquisition_scale",
        "station_terminals",
        "outage_duration_sec",
        "relay_ttl_sec",
    ]
    target: str = Field(default="*", min_length=1, max_length=128)
    unit: Literal["ratio", "count", "s"]
    mode: Literal["choice", "uniform"] = "choice"
    values: tuple[FiniteScalar, ...] = Field(default=(), max_length=64)
    low: FiniteScalar | None = None
    high: FiniteScalar | None = None
    source: str = Field(min_length=1, max_length=1024)
    parameter_status: Literal["synthetic"] = "synthetic"

    @model_validator(mode="after")
    def bounds(self):
        if self.unit != UNITS[self.parameter]:
            raise ValueError("parameter unit mismatch")
        if self.mode == "choice":
            if (
                not self.values
                or len(set(self.values)) != len(self.values)
                or self.low is not None
                or self.high is not None
            ):
                raise ValueError("choice requires unique values and no bounds")
            values = self.values
        else:
            if (
                self.values
                or self.low is None
                or self.high is None
                or self.low >= self.high
            ):
                raise ValueError("uniform requires ordered bounds and no values")
            values = [self.low, self.high]
        if self.parameter == "station_terminals":
            if self.mode != "choice" or any(
                v != int(v) or not 1 <= v <= 128 for v in values
            ):
                raise ValueError("terminal count requires integer choices in [1,128]")
        elif self.unit == "ratio":
            if any(not 1e-6 <= v <= 1e6 for v in values):
                raise ValueError("ratio outside execution guard bounds")
        elif any(not 0 < v <= 86400 for v in values):
            raise ValueError("duration outside execution guard bounds")
        return self


class FamilyTemplate(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")
    family: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "ground",
        "eo_processing",
        "relay",
        "recovery",
        "shared_terminal",
        "resource_limited",
    ]
    source: str = Field(min_length=1, max_length=1024)
    scenario: dict
    task: dict
    samples: int = Field(default=1, ge=1, le=32, strict=True)
    split: Literal["train", "holdout", "ood"] | None = None
    distributions: tuple[Distribution, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def distinct_parameters(self):
        keys = [(p.parameter, p.target) for p in self.distributions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate distribution target")
        for p in self.distributions:
            if (
                p.target == "*"
                and sum(q.parameter == p.parameter for q in self.distributions) > 1
            ):
                raise ValueError("wildcard and specific distributions cannot overlap")
        return self


class QualityGates(StrictModel):
    max_spool_bytes: int = Field(
        default=536870912, ge=65536, le=4294967296, strict=True
    )
    max_row_bytes: int = Field(default=33554432, ge=1024, le=134217728, strict=True)
    min_rl_rows: int = Field(default=1, ge=0, le=100000, strict=True)
    min_sft_rows: int = Field(default=1, ge=0, le=100000, strict=True)
    min_unique_missions: int = Field(default=1, ge=1, le=64, strict=True)
    max_error_episodes: int = Field(default=0, ge=0, le=320, strict=True)
    sft_min_completion: float = Field(default=1.0, ge=0, le=1, strict=True)
    sft_max_rejections: int = Field(default=0, ge=0, le=10000, strict=True)
    require_teacher_gates: bool = Field(default=True, strict=True)
    max_rows: int = Field(default=100000, ge=1, le=1000000, strict=True)
    max_bundle_bytes: int = Field(
        default=268435456, ge=1024, le=4294967296, strict=True
    )


class FactorySpec(StrictModel):
    schema_id: Literal["space-env.factory/1"] = Field(
        default="space-env.factory/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)
    parameter_status: Literal["synthetic"] = "synthetic"
    seed_policy: Literal["preserve", "derive-network"] = "derive-network"
    runtime_audit: Literal["full", "compact"] = "compact"
    workers: int = Field(default=1, ge=1, le=8, strict=True)
    templates: tuple[FamilyTemplate, ...] = Field(min_length=1, max_length=32)
    policies: tuple[PolicyConfig, ...] = Field(
        default=(
            PolicyConfig(name="priority"),
            PolicyConfig(name="earliest_deadline"),
            PolicyConfig(),
        ),
        min_length=1,
        max_length=5,
    )
    teacher_gates: TeacherGates | None = None
    quality: QualityGates = Field(default_factory=QualityGates)
    export_roles: tuple[str, ...] = Field(
        default=("@central",), min_length=1, max_length=65
    )
    sft_policies: tuple[str, ...] = ("route_aware_rolling",)
    max_generated_bytes: int = Field(
        default=67108864, ge=1024, le=536870912, strict=True
    )

    @model_validator(mode="after")
    def budget(self):
        if (
            len({t.id for t in self.templates}) != len(self.templates)
            or sum(t.samples for t in self.templates) > 64
        ):
            raise ValueError("duplicate template or case budget exceeded")
        names = {p.name for p in self.policies}
        if (
            len(names) != len(self.policies)
            or set(self.sft_policies) - names
            or len(set(self.sft_policies)) != len(self.sft_policies)
        ):
            raise ValueError("duplicate/unknown export policy")
        if len(set(self.export_roles)) != len(self.export_roles) or any(
            not r.strip() for r in self.export_roles
        ):
            raise ValueError("invalid export roles")
        if self.quality.min_sft_rows and not self.sft_policies:
            raise ValueError("SFT minimum requires an SFT policy")
        if self.quality.min_sft_rows and "@central" not in self.export_roles:
            raise ValueError("SFT minimum requires central-role export")
        if self.teacher_gates:
            g = self.teacher_gates
            if (
                g.candidate not in names
                or not g.baselines
                or len(set(g.baselines)) != len(g.baselines)
                or g.candidate in g.baselines
                or set(g.baselines) - names
            ):
                raise ValueError(
                    "teacher gates require distinct configured candidate/baselines"
                )
        return self
