"""Bounded evidence and uncertainty contracts; labels are not authentication."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar, StrictModel


class EvidenceSource(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "manufacturer_specification",
        "laboratory_measurement",
        "flight_measurement",
        "synthetic_fixture",
    ]
    uri: str = Field(min_length=1, max_length=2048)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_scope: str = Field(min_length=1, max_length=1024)
    locator: str = Field(min_length=1, max_length=1024)
    retrieved_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    redistribution: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def nonblank(self):
        if any(
            not getattr(self, k).strip()
            for k in ("id", "uri", "content_scope", "locator", "redistribution")
        ):
            raise ValueError("evidence fields must be nonblank")
        if self.kind != "synthetic_fixture" and not self.uri.startswith("https://"):
            raise ValueError("non-synthetic evidence requires an HTTPS source URI")
        date.fromisoformat(self.retrieved_on)
        if self.kind != "synthetic_fixture":
            parsed = urlsplit(self.uri)
            if (
                not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(
                    "evidence URI must have a host and contain no embedded credentials"
                )
        return self


MODELS = {
    "pv_efficiency": {
        "x_unit": "W",
        "y_unit": "W",
        "parameter_unit": "ratio",
        "meaning": "constant effective conversion efficiency at declared conditions",
    },
    "inverse_inertia": {
        "x_unit": "N*m*s",
        "y_unit": "rad/s",
        "parameter_unit": "1/(kg*m^2)",
        "meaning": "inverse principal-axis inertia from independently known net angular impulse",
    },
    "inverse_heat_capacity": {
        "x_unit": "J",
        "y_unit": "K",
        "parameter_unit": "K/J",
        "meaning": "inverse lumped heat capacity from independently known net absorbed energy",
    },
}


class Observation(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    experiment_id: str = Field(min_length=1, max_length=128)
    split: Literal["train", "holdout"]
    x: FiniteScalar = Field(ge=-1e12, le=1e12)
    y: FiniteScalar = Field(ge=-1e12, le=1e12)
    sigma_y: float = Field(ge=1e-12, le=1e12, strict=True)

    @model_validator(mode="after")
    def independent_summary(self):
        if not self.id.strip() or not self.experiment_id.strip():
            raise ValueError("measurement and experiment IDs must be nonblank")
        if self.x != 0 and abs(self.x) < 1e-12:
            raise ValueError("nonzero calibration excitation below numerical guard")
        return self


class CalibrationGates(StrictModel):
    parameter_low: float = Field(gt=0, le=1e12, strict=True)
    parameter_high: float = Field(gt=0, le=1e12, strict=True)
    max_train_standardized_rmse: float = Field(gt=0, le=1e6, strict=True)
    max_holdout_standardized_rmse: float = Field(gt=0, le=1e6, strict=True)
    max_holdout_absolute_error: float = Field(gt=0, le=1e12, strict=True)
    min_train: int = Field(default=4, ge=3, le=256, strict=True)
    min_holdout: int = Field(default=3, ge=2, le=256, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.parameter_low >= self.parameter_high:
            raise ValueError("ordered positive parameter bounds required")
        return self


class CalibrationSpec(StrictModel):
    schema_version: Literal["space-env.scalar-calibration/1"] = (
        "space-env.scalar-calibration/1"
    )
    id: str = Field(min_length=1, max_length=128)
    component_id: str = Field(min_length=1, max_length=128)
    model: Literal["pv_efficiency", "inverse_inertia", "inverse_heat_capacity"]
    x_unit: Literal["W", "N*m*s", "J"]
    y_unit: Literal["W", "rad/s", "K"]
    experiment_model: Literal[
        "constant-effective-efficiency",
        "isolated-principal-axis-impulse",
        "isolated-known-net-heat",
    ]
    frame: Literal["scalar", "BODY"]
    axis: Literal["x", "y", "z"] | None = None
    source: EvidenceSource
    conditions: str = Field(min_length=12, max_length=2048)
    input_error_assumption: Literal["negligible-independent-x"]
    noise_assumption: Literal["independent-known-gaussian-sigma-y"]
    observations: tuple[Observation, ...] = Field(min_length=5, max_length=512)
    gates: CalibrationGates

    @model_validator(mode="after")
    def experiment_contract(self):
        if any(not s.strip() for s in (self.id, self.component_id, self.conditions)):
            raise ValueError("calibration identity/conditions must be nonblank")
        if self.source.kind == "manufacturer_specification":
            raise ValueError("a datasheet is not a calibration measurement series")
        expected = {
            "pv_efficiency": ("constant-effective-efficiency", "scalar"),
            "inverse_inertia": ("isolated-principal-axis-impulse", "BODY"),
            "inverse_heat_capacity": ("isolated-known-net-heat", "scalar"),
        }[self.model]
        if (self.experiment_model, self.frame) != expected or (
            (self.axis is not None) != (self.model == "inverse_inertia")
        ):
            raise ValueError(
                "experiment assumptions/frame/axis do not match the identifiable scalar model"
            )
        model = MODELS[self.model]
        if (self.x_unit, self.y_unit) != (model["x_unit"], model["y_unit"]):
            raise ValueError("calibration model/unit mismatch")
        if self.model == "pv_efficiency" and (
            self.gates.parameter_high > 1 or any(o.x < 0 for o in self.observations)
        ):
            raise ValueError("PV excitation/efficiency outside physical support")
        if len({r.id for r in self.observations}) != len(self.observations):
            raise ValueError("duplicate measurement ID")
        # One independent aggregate per experiment. A time-series run must not
        # contribute adjacent/overlapping samples to train and holdout.
        if len({r.experiment_id for r in self.observations}) != len(self.observations):
            raise ValueError(
                "experiment reuse/overlap; one aggregate per independent experiment required"
            )
        for split, minimum in (
            ("train", self.gates.min_train),
            ("holdout", self.gates.min_holdout),
        ):
            if sum(r.split == split for r in self.observations) < minimum:
                raise ValueError("too few independent train/holdout experiments")
        return self


class SensitivityParameter(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    target: Literal[
        "wheel_torque_scale",
        "wheel_momentum_scale",
        "body_inertia_scale",
        "battery_capacity_scale",
        "solar_efficiency_scale",
        "thermal_capacity_scale",
    ]
    actor: str = Field(min_length=1, max_length=128)
    unit: Literal["ratio"] = "ratio"
    mode: Literal["uniform", "truncated_normal"]
    low: float = Field(gt=0, le=1e6, strict=True)
    high: float = Field(gt=0, le=1e6, strict=True)
    mean: FiniteScalar | None = None
    sigma: float | None = Field(default=None, gt=0, le=1e6, strict=True)
    latent_group: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    status: Literal["assumed_sensitivity"] = "assumed_sensitivity"
    rationale: str = Field(min_length=12, max_length=1024)

    @model_validator(mode="after")
    def ordered(self):
        if (
            self.low >= self.high
            or not self.actor.strip()
            or not self.rationale.strip()
        ):
            raise ValueError(
                "sensitivity requires ordered bounds and nonblank provenance"
            )
        if self.mode == "uniform" and (self.mean is not None or self.sigma is not None):
            raise ValueError("uniform has no normal parameters")
        if self.mode == "truncated_normal" and (
            self.mean is None or self.sigma is None or not math.isfinite(self.mean)
        ):
            raise ValueError("truncated normal requires finite mean and sigma")
        return self


class EnsembleSpec(StrictModel):
    schema_version: Literal["space-env.parameter-ensemble/1"] = (
        "space-env.parameter-ensemble/1"
    )
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    seed: int = Field(ge=0, le=2**63 - 1, strict=True)
    samples: int = Field(ge=1, le=64, strict=True)
    parameters: tuple[SensitivityParameter, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique(self):
        if len({p.id for p in self.parameters}) != len(self.parameters) or len(
            {(p.actor, p.target) for p in self.parameters}
        ) != len(self.parameters):
            raise ValueError("duplicate parameter ID or effective target")
        return self
