"""Versioned heterogeneous EO tasks derived from a public synthetic formation.

Rotated sites/orbits are distinct geometries, not independent blind holdouts.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, field_validator

from space_env.core.attitude import align_z
from space_env.reference import EOPSeries, Epoch, FrameContext
from space_env.reference.area_coverage import AreaOfInterest
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks.choice_missions import _opaque, choice_mission
from space_env.tasks.schema import TaskSpec


class HeterogeneousMissionSpec(StrictModel):
    profile: Literal["balanced", "bulk_tradeoff"] = "balanced"
    longitude_shift_deg: Literal[-12, 0, 12] = 0
    catalogue_seed: int = Field(default=0, ge=0, le=1000000, strict=True)

    @field_validator("longitude_shift_deg", mode="before")
    @classmethod
    def integer_meridian(cls, value):
        if type(value) is not int:
            raise ValueError("longitude shift must be an integer, never bool/float")
        return value


def _longitude(value, shift):
    return (value + shift + 180.0) % 360.0 - 180.0


def heterogeneous_mission(value=None):
    cfg = HeterogeneousMissionSpec.model_validate({} if value is None else value)
    case = choice_mission({"cameras": 2, "catalogue_seed": cfg.catalogue_seed})
    d, task = case["scenario"]["definition"], case["task"]
    requests, jobs = d["lifecycle"]["requests"], d["lifecycle"]["jobs"]
    # Literal costs, before policy execution. No feasibility/outcome search.
    costs = (
        [
            (7, 1.0, 600000, 200000, 1.0),
            (5, 2.0, 1200000, 300000, 3.0),
            (11, 3.0, 1800000, 600000, 5.0),
            (3, 1.0, 400000, 150000, 1.0),
        ]
        if cfg.profile == "balanced"
        else [
            (11, 3.0, 3200000, 2800000, 8.0),
            (6, 1.0, 300000, 150000, 1.0),
            (6, 1.0, 300000, 150000, 1.0),
            (9, 2.0, 1000000, 500000, 3.0),
        ]
    )
    jobs_by_input = {j["inputs"][0]: j for j in jobs}
    controls = {}
    for request, (priority, acquisition, raw, final, work) in zip(
        requests, costs, strict=True
    ):
        request.update(
            priority=priority,
            acquisition_sec=acquisition,
            raw_size_bits=raw,
            deadline_sec=60.0,
        )
        job = jobs_by_input[request["raw_product_id"]]
        job.update(work_sec=work, workspace_bits=100000)
        job["outputs"][0]["size_bits"] = final
        controls[request["id"]] = {
            "imaging": acquisition,
            "transform": float(math.ceil(work + job["switching_sec"])),
            "relay": float(math.ceil(final / 250000) + 2),
            "downlink": float(math.ceil(final / 250000) + 2),
        }
    # All jobs fit in isolation; simultaneous storage/custody still consumes memory.
    for actor in d["operations"]["vehicles"].values():
        actor["memory_bits"] = 8000000.0
    if cfg.profile == "bulk_tradeoff":
        d["operations"]["outages"] = [
            {"station": "Ground", "start_sec": 0.0, "end_sec": 18.0},
            {"station": "Ground", "start_sec": 42.0, "end_sec": 60.0},
        ]
    shift = cfg.longitude_shift_deg
    epoch = Epoch.utc(next(iter(d["orbits"].values()))["initial"]["epoch_utc"])
    terrestrial = FrameContext(EOPSeries.packaged()).matrix(epoch)
    angle = math.radians(shift)
    c, s = math.cos(angle), math.sin(angle)
    rotation = (
        terrestrial.T
        @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        @ terrestrial
    )
    for actor, orbit in d["orbits"].items():
        for key in ("r_m", "v_m_s"):
            orbit["initial"][key] = (
                rotation @ np.asarray(orbit["initial"][key])
            ).tolist()
        d["spacecraft_bus"][actor]["attitude"]["initial_quaternion_wxyz"] = align_z(
            -np.asarray(orbit["initial"]["r_m"])
        ).tolist()
    for row in [*d["acquisition_targets"], *d["operations"]["stations"]]:
        row["lon_deg"] = _longitude(row["lon_deg"], shift)
    areas = {}
    for area in d["lifecycle"]["areas_of_interest"]:
        area["anchor_lon_deg"] = _longitude(area["anchor_lon_deg"], shift)
        area["shell"] = [[_longitude(lon, shift), lat] for lon, lat in area["shell"]]
        if area.get("holes"):
            area["holes"] = [
                [[_longitude(lon, shift), lat] for lon, lat in ring]
                for ring in area["holes"]
            ]
        areas[area["id"]] = digest(
            AreaOfInterest.model_validate(area).model_dump(mode="json")
        )
    for cloud in d["lifecycle"]["cloud_masks"]:
        cloud["aoi_sha256"] = areas[cloud["aoi_id"]]
    for action in task["actions"]:
        for command in action.get("commands", []):
            if command["kind"] in controls[action["request_ref"]]:
                command["duration_sec"] = controls[action["request_ref"]][
                    command["kind"]
                ]
                if command["kind"] == "relay":
                    command["relay"]["ttl_sec"] = 60.0
        body = {k: v for k, v in action.items() if k != "id"}
        for command in body.get("commands", []):
            command.pop("command_id", None)
        action["id"] = _opaque("act", digest(body), cfg.catalogue_seed)
        for command in action.get("commands", []):
            command["command_id"] = action["id"]
    task["actions"].sort(key=lambda a: a["id"])
    task.update(objective="sla", id="heterogeneous-" + cfg.profile)
    case.update(
        id=f"{cfg.profile}-meridian-{shift}",
        configuration=cfg.model_dump(mode="json"),
        schema="space-env.heterogeneous-mission/1",
    )
    case["task"] = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    case["scenario"].update(id=case["id"], lineage="heterogeneous-formation/v1")
    case["scenario"]["metadata"] = {
        "configuration": cfg.model_dump(mode="json"),
        "scope": "public transformed synthetic geometry; ideal local telemetry, not independent/OOD or flight-calibrated data",
    }
    case["scope"] = (
        "heterogeneous acquisition/CPU/data costs, SLA reward, unallocated camera and gateway choices; no full-mission optimality claim"
    )
    return case
