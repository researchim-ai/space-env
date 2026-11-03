"""Curated manufacturer reference points and explicit mixed-evidence adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from pydantic import Field, model_validator

from space_env.calibration.schema import EvidenceSource
from space_env.reference.operations import VehicleResources
from space_env.reference.rigid_attitude import RigidAttitudeModel
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest

CATALOG = Path(__file__).with_name("data") / "hardware_v1.json"
CATALOG_SHA256 = "08a653c1b5e61f1a4f75d0e7b0b4511cca385841e930beac8c6744f81279d1e3"


class WheelAssumptions(StrictModel):
    inertia_kg_m2: tuple[tuple[FiniteScalar, FiniteScalar, FiniteScalar], ...]
    efficiency: float = Field(gt=0, le=1, strict=True)
    idle_power_w: float = Field(ge=0, le=1e8, strict=True)
    torque_loss_w_per_nm2: float = Field(gt=0, le=1e12, strict=True)
    rationale: str = Field(min_length=12, max_length=2048)

    @model_validator(mode="after")
    def explicit_assumptions(self):
        if len(self.rationale.strip()) < 12:
            raise ValueError("nonblank wheel-model assumption rationale required")
        return self


def hardware_catalog():
    raw = CATALOG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
        raise ValueError("packaged hardware catalog bytes differ from frozen inventory")
    data = json.loads(raw)
    if data.get("schema") != "space-env.hardware-reference-points/1" or data.get(
        "sha256"
    ) != digest({k: v for k, v in data.items() if k != "sha256"}):
        raise ValueError("invalid hardware catalog digest")
    for source in data["sources"].values():
        EvidenceSource.model_validate(source)
    return copy.deepcopy(data)


def hardware_profile(id):
    data = hardware_catalog()
    if not isinstance(id, str) or id not in data["profiles"]:
        raise ValueError("unknown hardware reference profile")
    profile = copy.deepcopy(data["profiles"][id])
    return {
        "id": id,
        **profile,
        "source": data["sources"][profile["source_id"]],
        "catalog_sha256": data["sha256"],
    }


def verify_source_bytes(value, content):
    source = EvidenceSource.model_validate(value)
    if not isinstance(content, bytes) or len(content) > 16_000_000:
        raise ValueError("source verification requires <=16MB bytes")
    observed = hashlib.sha256(content).hexdigest()
    return {
        "passed": observed == source.content_sha256,
        "observed_sha256": observed,
        "expected_sha256": source.content_sha256,
        "independent_authentication": False,
    }


def wheel_model(profile_ids, assumptions):
    """Three ideal orthogonal wheels; no imported full power/speed/friction curve."""
    assumptions = WheelAssumptions.model_validate(assumptions)
    if not isinstance(profile_ids, (tuple, list)) or len(profile_ids) != 3:
        raise ValueError("exactly three explicit orthogonal wheel profile IDs required")
    profiles = [hardware_profile(id) for id in profile_ids]
    if any(p["kind"] != "reaction_wheel" for p in profiles):
        raise ValueError("wheel adapter requires reaction-wheel reference points")
    facts = [p["values_si"] for p in profiles]
    spin = [f["rated_momentum_nms"] / f["speed_at_rated_momentum_rad_s"] for f in facts]
    model = RigidAttitudeModel(
        model_id="published-wheel-points/mixed-1",
        source="; ".join(p["id"] for p in profiles) + "; " + assumptions.rationale,
        inertia_kg_m2=assumptions.inertia_kg_m2,
        wheel_inertia_kg_m2=tuple(spin),
        max_momentum_nms=tuple(f["rated_momentum_nms"] for f in facts),
        max_torque_nm=tuple(f["torque_at_rated_momentum_nm"] for f in facts),
        efficiency=assumptions.efficiency,
        idle_power_w=assumptions.idle_power_w,
        torque_loss_w_per_nm2=assumptions.torque_loss_w_per_nm2,
    )
    result = {
        "schema": "space-env.wheel-reference-adapter/1",
        "model": model.model_dump(mode="json"),
        "profile_ids": list(profile_ids),
        "profile_evidence": profiles,
        "assumptions": assumptions.model_dump(mode="json"),
        "parameter_evidence": {
            "max_momentum_nms": "scenario cap chosen equal to published rated point, not hardware maximum",
            "max_torque_nm": "scenario cap chosen equal to torque at rated momentum; constant torque envelope is assumed",
            "wheel_inertia_kg_m2": "derived h/omega from published rated point; constant spin inertia approximation",
            "inertia_kg_m2": "caller assumption, not manufacturer spacecraft inertia",
            "efficiency/idle_power/torque_loss": "caller assumptions; published steady-at-speed power is not idle power",
        },
        "qualification_status": "mixed-published-derived-assumed",
        "industrial_readiness": False,
    }
    result["sha256"] = digest(result)
    return result


def solar_resources(profile_id, base, *, area_m2, packing_factor, assumption):
    """Effective constant-efficiency resource model, not a measured I-V/aging model."""
    profile = hardware_profile(profile_id)
    if profile["kind"] != "solar_reference_point":
        raise ValueError("solar adapter requires a solar reference point")
    for value in (area_m2, packing_factor):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("solar assembly inputs must be finite scalars")
    if (
        not 0 < area_m2 <= 1e6
        or not 0 < packing_factor <= 1
        or not isinstance(assumption, str)
        or len(assumption.strip()) < 12
        or len(assumption) > 2048
    ):
        raise ValueError(
            "positive area, packing factor and explicit operating-condition assumption required"
        )
    vehicle = VehicleResources.model_validate(base).model_dump(mode="json")
    vehicle.update(
        solar_area_m2=float(area_m2),
        solar_efficiency=profile["values_si"]["efficiency"] * packing_factor,
    )
    vehicle = VehicleResources.model_validate(vehicle)
    from space_env.core.constants import SOLAR_CONSTANT

    result = {
        "schema": "space-env.solar-reference-adapter/1",
        "vehicle_resources": vehicle.model_dump(mode="json"),
        "profile_evidence": profile,
        "assumptions": {
            "packing_factor": float(packing_factor),
            "reference_area_m2": float(area_m2),
            "operating_conditions": assumption,
            "irradiance_mapping": "constant efficiency linear model, not interpolation between datasheet calibrations",
            "native_solar_constant_w_m2": SOLAR_CONSTANT,
        },
        "qualification_status": "mixed-published-derived-assumed",
        "industrial_readiness": False,
    }
    result["sha256"] = digest(result)
    return result
