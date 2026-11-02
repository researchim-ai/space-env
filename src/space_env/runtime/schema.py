"""Strict scenario/command envelopes; physical definitions remain backend-specific."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from space_env.simulation.snapshot import canonical_json

FiniteScalar = Annotated[float, Field(strict=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class SourceRecord(StrictModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    uri: str = ""
    license: str = "unspecified"


class RuntimeLimits(StrictModel):
    max_operations: int = Field(default=10000, ge=1, le=100000, strict=True)
    max_commands: int = Field(default=10000, ge=1, le=10000, strict=True)
    max_advance_sec: float = Field(default=3600.0, gt=1e-9, le=259200.0, strict=True)


class Fidelity(StrictModel):
    trajectory: Literal[
        "fixed-gcrs-ephemeris",
        "controlled-gcrs",
        "propagated-gcrs-coast",
        "coupled-gcrs-thrust",
        "legacy-dynamic",
    ]
    resources: Literal["explicit-surrogate", "coupled-bus-surrogate", "legacy-training"]


class ScenarioSpec(StrictModel):
    schema_version: Literal[3] = 3
    id: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    backend: Literal[
        "reference-fixed",
        "reference-dynamic",
        "reference-lifecycle",
        "reference-spacecraft",
        "legacy-training",
    ]
    physics_profile: Literal[
        "reference-fixed-downlink/1",
        "reference-dynamic-downlink/1",
        "reference-lifecycle/1",
        "reference-spacecraft-bus/1",
        "legacy-training/1",
    ]
    actor_order: tuple[str, ...] = Field(min_length=1)
    definition: dict
    seed: int = Field(default=0, ge=0, strict=True)
    observation: Literal["ideal-local/1"] = "ideal-local/1"
    objective: Literal["on_time_bits", "delivered_bits", "fulfilled_requests"]
    fidelity: Fidelity
    required_capabilities: tuple[str, ...] = ()
    sources: tuple[SourceRecord, ...] = ()
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def contract(self):
        expected = {
            "reference-fixed": ("reference-fixed-downlink/1", "on_time_bits"),
            "reference-dynamic": ("reference-dynamic-downlink/1", "on_time_bits"),
            "reference-lifecycle": ("reference-lifecycle/1", "fulfilled_requests"),
            "reference-spacecraft": (
                "reference-spacecraft-bus/1",
                "fulfilled_requests",
            ),
            "legacy-training": ("legacy-training/1", "delivered_bits"),
        }
        if (self.physics_profile, self.objective) != expected[self.backend]:
            raise ValueError(
                "backend/profile/objective mismatch; no implicit conversion"
            )
        fidelity = {
            "reference-fixed": ("fixed-gcrs-ephemeris", "explicit-surrogate"),
            "reference-dynamic": ("controlled-gcrs", "explicit-surrogate"),
            "reference-lifecycle": ("controlled-gcrs", "explicit-surrogate"),
            "reference-spacecraft": (
                "coupled-gcrs-thrust"
                if any(
                    self.definition.get(k) is not None
                    for k in ("actuation", "unloading")
                )
                else "propagated-gcrs-coast",
                "coupled-bus-surrogate",
            ),
            "legacy-training": ("legacy-dynamic", "legacy-training"),
        }
        if (self.fidelity.trajectory, self.fidelity.resources) != fidelity[
            self.backend
        ]:
            raise ValueError("fidelity does not match declared backend")
        if len(self.actor_order) != len(set(self.actor_order)) or any(
            not n.strip() for n in self.actor_order
        ):
            raise ValueError("actor order must be unique and nonempty")
        if not self.id.strip() or not self.lineage.strip():
            raise ValueError("scenario ID and lineage must be nonblank")
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique")
        canonical_json(self.model_dump(mode="json"))
        return self


class ImpulseSpec(StrictModel):
    frame: Literal["GCRS"]
    delta_v_m_s: tuple[FiniteScalar, FiniteScalar, FiniteScalar]

    @model_validator(mode="after")
    def finite_delta_v(self):
        if (
            not all(math.isfinite(v) for v in self.delta_v_m_s)
            or math.hypot(*self.delta_v_m_s) <= 0
        ):
            raise ValueError("impulse requires a finite nonzero delta-v")
        return self


class BurnSpec(StrictModel):
    frame: Literal["GCRS"]
    direction_gcrs: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    thrust_n: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def unit_direction(self):
        if not math.isclose(
            math.hypot(*self.direction_gcrs), 1.0, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("finite burn direction must be a unit GCRS vector")
        return self


class SlewSpec(StrictModel):
    frame: Literal["GCRS"]
    quaternion_wxyz: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]

    @model_validator(mode="after")
    def unit_quaternion(self):
        if not math.isclose(
            math.hypot(*self.quaternion_wxyz), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("slew requires unit body-to-GCRS quaternion wxyz")
        return self


class ForecastBinding(StrictModel):
    scenario_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    trajectory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: int = Field(ge=0, strict=True)
    geometry_revision: int = Field(ge=0, strict=True)
    actor_id: str = Field(min_length=1)


class BodyBurnSpec(StrictModel):
    frame: Literal["BODY"]
    direction_body: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    thrust_n: float = Field(gt=0, strict=True)
    firing_sec: float = Field(gt=1e-6, le=259200, strict=True)

    @model_validator(mode="after")
    def unit_direction(self):
        if not math.isclose(
            math.hypot(*self.direction_body), 1.0, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("body burn direction must be a unit BODY vector")
        return self


class WheelTorqueSpec(StrictModel):
    frame: Literal["BODY"]
    motor_torque_nm: tuple[FiniteScalar, FiniteScalar, FiniteScalar]


class MomentumUnloadSpec(StrictModel):
    frame: Literal["BODY"]
    target_momentum_nms: tuple[FiniteScalar, FiniteScalar, FiniteScalar] = (
        0.0,
        0.0,
        0.0,
    )
    tolerance_nms: float = Field(default=1e-4, gt=0, strict=True)


class RelaySpec(StrictModel):
    ttl_sec: float = Field(gt=1e-6, le=259200, strict=True)


class RuntimeCommand(StrictModel):
    command_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    kind: Literal[
        "wait",
        "downlink",
        "processing",
        "imaging",
        "impulse",
        "finite_burn",
        "transform",
        "slew",
        "body_burn",
        "wheel_torque",
        "momentum_unload",
        "relay",
    ]
    duration_sec: float = Field(ge=0, strict=True)
    destination_id: str = ""
    product_id: str = ""
    request_id: str = ""
    job_id: str = ""
    impulse: ImpulseSpec | None = None
    burn: BurnSpec | None = None
    forecast: ForecastBinding | None = None
    slew: SlewSpec | None = None
    body_burn: BodyBurnSpec | None = None
    wheel_torque: WheelTorqueSpec | None = None
    momentum_unload: MomentumUnloadSpec | None = None
    relay: RelaySpec | None = None

    @model_validator(mode="after")
    def fields(self):
        if self.kind == "impulse":
            if self.duration_sec != 0 or self.impulse is None or self.burn is not None:
                raise ValueError(
                    "impulse requires zero duration and impulse specification"
                )
        elif self.duration_sec <= 1e-9:
            raise ValueError("control window must exceed 1ns")
        if self.kind == "finite_burn" and self.burn is None:
            raise ValueError("finite burn requires burn specification")
        if (self.kind != "impulse" and self.impulse is not None) or (
            self.kind != "finite_burn" and self.burn is not None
        ):
            raise ValueError("maneuver specification does not match command kind")
        if not self.command_id.strip() or not self.actor_id.strip():
            raise ValueError("command and actor IDs must be nonblank")
        if self.kind not in {"downlink", "relay"} and self.destination_id:
            raise ValueError("destination is only valid for downlink or relay")
        if self.kind not in {"downlink", "processing", "relay"} and self.product_id:
            raise ValueError("product selection requires downlink, processing or relay")
        if (self.kind == "relay") != (self.relay is not None):
            raise ValueError("relay specification must match command kind")
        if self.kind == "relay" and (not self.product_id.strip() or not self.destination_id.strip()):
            raise ValueError("relay requires product_id and destination actor")
        if (self.kind == "slew") != (self.slew is not None):
            raise ValueError("slew specification must match command kind")
        if (self.kind == "momentum_unload") != (self.momentum_unload is not None):
            raise ValueError("momentum unloading specification must match command kind")
        if (self.kind == "wheel_torque") != (self.wheel_torque is not None):
            raise ValueError("wheel torque specification must match command kind")
        if (self.kind == "body_burn") != (self.body_burn is not None):
            raise ValueError("body burn specification must match command kind")
        if self.kind != "transform" and self.job_id:
            raise ValueError("job_id requires transform command")
        if self.kind == "transform" and not self.job_id.strip():
            raise ValueError("transform requires job_id")
        if self.kind != "imaging" and self.request_id:
            raise ValueError("request selection requires imaging")
        return self


class AccessPolicy(StrictModel):
    truth: bool = Field(default=False, strict=True)
    forecast: bool = Field(default=False, strict=True)


def load_scenario(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return ScenarioSpec.model_validate(value).model_dump(mode="json")


def reference_scenario(spec: dict, *, id: str, lineage: str, seed: int = 0) -> dict:
    return ScenarioSpec(
        id=id,
        lineage=lineage,
        backend="reference-fixed",
        physics_profile="reference-fixed-downlink/1",
        actor_order=tuple(spec["ephemerides"]),
        definition=spec,
        seed=seed,
        fidelity=Fidelity(
            trajectory="fixed-gcrs-ephemeris", resources="explicit-surrogate"
        ),
        objective="on_time_bits",
    ).model_dump(mode="json")


def legacy_scenario(world, *, id: str, lineage: str, seed: int = 0) -> dict:
    from space_env.simulation.snapshot import world_payload

    return ScenarioSpec(
        id=id,
        lineage=lineage,
        backend="legacy-training",
        physics_profile="legacy-training/1",
        actor_order=tuple(world.engines),
        definition=world_payload(world),
        seed=seed,
        fidelity=Fidelity(trajectory="legacy-dynamic", resources="legacy-training"),
        objective="delivered_bits",
    ).model_dump(mode="json")


def scenario_json_schema() -> dict:
    return json.loads(canonical_json(ScenarioSpec.model_json_schema()))


def dynamic_reference_scenario(
    spec: dict, *, id: str, lineage: str, seed: int = 0
) -> dict:
    return ScenarioSpec(
        id=id,
        lineage=lineage,
        backend="reference-dynamic",
        physics_profile="reference-dynamic-downlink/1",
        actor_order=tuple(spec["orbits"]),
        definition=spec,
        seed=seed,
        fidelity=Fidelity(trajectory="controlled-gcrs", resources="explicit-surrogate"),
        objective="on_time_bits",
    ).model_dump(mode="json")


def lifecycle_scenario(spec: dict, *, id: str, lineage: str, seed: int = 0) -> dict:
    return ScenarioSpec(
        id=id,
        lineage=lineage,
        backend="reference-lifecycle",
        physics_profile="reference-lifecycle/1",
        actor_order=tuple(spec["orbits"]),
        definition=spec,
        seed=seed,
        fidelity=Fidelity(trajectory="controlled-gcrs", resources="explicit-surrogate"),
        objective="fulfilled_requests",
    ).model_dump(mode="json")


def spacecraft_scenario(spec: dict, *, id: str, lineage: str, seed: int = 0) -> dict:
    return ScenarioSpec(
        id=id,
        lineage=lineage,
        backend="reference-spacecraft",
        physics_profile="reference-spacecraft-bus/1",
        actor_order=tuple(spec["orbits"]),
        definition=spec,
        seed=seed,
        fidelity=Fidelity(
            trajectory="coupled-gcrs-thrust"
            if any(spec.get(k) is not None for k in ("actuation", "unloading"))
            else "propagated-gcrs-coast",
            resources="coupled-bus-surrogate",
        ),
        objective="fulfilled_requests",
    ).model_dump(mode="json")
