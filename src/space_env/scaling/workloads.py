"""Synthetic multi-plane fixed-ephemeris workloads, not flight-calibrated missions."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_geodetic
from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.runtime import reference_scenario
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class FleetWorkloadSpec(StrictModel):
    id: str = Field(default="multi-plane", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    actors: int = Field(default=16, ge=1, le=64, strict=True)
    products: int = Field(default=100, ge=1, le=1023, strict=True)
    planes: int = Field(default=4, ge=1, le=16, strict=True)
    horizon_sec: float | None = Field(default=None, gt=0, le=21600, strict=True)
    decision_sec: float = Field(default=2.0, gt=0, le=3600, strict=True)
    product_bits: float = Field(default=100000.0, ge=1.0, le=1e9, strict=True)
    mode: Literal["nominal", "ground_outage"] = "nominal"
    repeats: int = Field(default=2, ge=1, le=10, strict=True)

    @model_validator(mode="after")
    def feasible_envelope(self):
        if self.planes > self.actors:
            raise ValueError("planes cannot exceed actors")
        if self.horizon_sec is not None and self.horizon_sec < self.decision_sec:
            raise ValueError("horizon must include at least one decision")
        horizon = self.horizon_sec or self.decision_sec * (
            math.ceil(self.products / self.actors) + 1
        )
        if horizon > 21600 or math.ceil(horizon / self.decision_sec) >= 10000:
            raise ValueError("fleet horizon/decision budget exceeded")
        return self


def fleet_workload(value):
    cfg = FleetWorkloadSpec.model_validate(value)
    horizon = cfg.horizon_sec or cfg.decision_sec * (
        math.ceil(cfg.products / cfg.actors) + 1
    )
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    radius = 7_000_000.0
    names = [f"sat-{i:03}" for i in range(cfg.actors)]
    ephemerides, stations = {}, []
    for plane in range(cfg.planes):
        raan = 2 * math.pi * plane / cfg.planes
        inclination = math.radians(50 + plane * 0.5)
        x = np.array([math.cos(raan), math.sin(raan), 0.0])
        y = np.array(
            [
                -math.sin(raan) * math.cos(inclination),
                math.cos(raan) * math.cos(inclination),
                math.sin(inclination),
            ]
        )
        lat, lon, _ = ecef_to_geodetic(context.matrix(epoch) @ (radius * x))
        members = list(range(plane, cfg.actors, cfg.planes))
        stations.append(
            {
                "name": f"Ground-{plane}",
                "lat_deg": float(lat),
                "lon_deg": float(lon),
                "terminals": len(members),
            }
        )
        for slot, i in enumerate(members):
            # Distinct phase and altitude, including within one plane.
            phase = 0.003 * (slot - (len(members) - 1) / 2)
            r = radius + 100 * i
            initial = ReferenceState(
                epoch,
                r * (math.cos(phase) * x + math.sin(phase) * y),
                math.sqrt(MU_EARTH / r) * (-math.sin(phase) * x + math.cos(phase) * y),
            )
            trajectory = propagate(
                initial,
                horizon,
                context,
                model=ForceModel(j2=True),
                sample_sec=min(4.0, horizon),
                max_step_sec=min(4.0, horizon),
            )
            ephemerides[names[i]] = {**trajectory, "sha256": digest(trajectory)}
    products = [
        {
            "id": f"product-{i:04}",
            "actor": names[i % cfg.actors],
            "size_bits": cfg.product_bits,
            "deadline_sec": horizon,
            "destination": f"Ground-{(i % cfg.actors) % cfg.planes}",
        }
        for i in range(cfg.products)
    ]
    # Eight independently selectable actor commands per role; joint limit stays 64.
    groups = [names[i : i + 8] for i in range(0, cfg.actors, 8)]
    roles = {
        f"control-{i}": {"actors": group, "action_slots": len(group)}
        for i, group in enumerate(groups)
    }
    owner = {a: role for role, group in roles.items() for a in group["actors"]}
    native = {
        "schema": "space-env.reference-downlink-spec/1",
        "ephemerides": {a: ephemerides[a] for a in names},
        "vehicles": {
            a: {
                "memory_bits": cfg.product_bits
                * (math.ceil(cfg.products / cfg.actors) + 1),
                "battery_wh": 100.0,
                "initial_soc": 1.0,
            }
            for a in names
        },
        "stations": stations,
        "products": products,
        "outages": [
            {"station": g["name"], "start_sec": 0.0, "end_sec": horizon}
            for g in stations
        ]
        if cfg.mode == "ground_outage"
        else [],
        "horizon_sec": horizon,
    }
    task = {
        "id": cfg.id,
        "runtime_audit": "compact",
        "roles": roles,
        "max_joint_actions": cfg.actors,
        "max_actions": cfg.products + 1,
        "max_entities": 2 * cfg.products + cfg.actors + cfg.planes + 32,
        "max_relations": 5 * cfg.products + 4 * cfg.actors + 64,
        "decision_sec": cfg.decision_sec,
        "max_decision_sec": cfg.decision_sec,
        "max_decisions": math.ceil(horizon / cfg.decision_sec) + 1,
        "actions": [
            {
                "id": "send-" + p["id"],
                "role": owner[p["actor"]],
                "kind": "submit",
                "commands": [
                    {
                        "command_id": p["id"],
                        "actor_id": p["actor"],
                        "kind": "downlink",
                        "duration_sec": cfg.decision_sec,
                        "product_id": p["id"],
                        "destination_id": p["destination"],
                    }
                ],
            }
            for p in products
        ],
    }
    return {
        "id": cfg.id,
        "family": "synthetic-multi-plane-" + cfg.mode,
        "source": "Distinct J2-propagated GCRS initial states and planes; fixed ephemeris resource workload, not controlled-orbit/coupled-bus or calibrated mission qualification",
        "scenario": reference_scenario(native, id=cfg.id, lineage="scale-" + cfg.id),
        "task": task,
        "policy": {"name": "priority", "max_candidates": max(32, cfg.actors)},
        "repeats": cfg.repeats,
        "timeout_sec": 600,
        "min_completion_fraction": 1.0 if cfg.mode == "nominal" else 0.0,
    }
