"""Constructive two-wave EO campaign with a continuous coupled resource runtime.

The second target is placed under a predeclared fixed boresight at a nominal
epoch. This is an explicit synthetic opportunity, not blind customer demand.
No propagation, future schedule or target placement is supplied to the policy.
"""

from __future__ import annotations

import copy
import math

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
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.area_coverage import AreaOfInterest, ray_ellipsoid
from space_env.reference.spacecraft_bus import rotate_body
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest


class LongCampaignSpec(StrictModel):
    horizon_sec: FiniteScalar = Field(default=7200.0, ge=120, le=21600)
    release_sec: FiniteScalar | None = Field(default=None, ge=20)
    battery_wh: FiniteScalar = Field(default=100.0, ge=0.1, le=1000)
    sample_sec: FiniteScalar = Field(default=30.0, gt=0, le=60)
    max_step_sec: FiniteScalar = Field(default=60.0, gt=0, le=120)
    execution_step_sec: FiniteScalar = Field(default=5.0, ge=0.01, le=5)
    ground_interruption: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def window(self):
        if self.release_sec is not None and self.release_sec + 30 >= self.horizon_sec:
            raise ValueError("late-wave delivery window must precede endurance end")
        return self


def long_campaign(configuration=None):
    from space_env.tasks.modeling_suite import compile_suite

    spec = LongCampaignSpec.model_validate(
        {} if configuration is None else configuration
    )
    case = copy.deepcopy(
        next(c for c in compile_suite()["cases"] if c["id"] == "delivery")
    )
    scenario, task = case["scenario"], case["task"]
    definition = scenario["definition"]
    initial = definition["orbits"]["b"]["initial"]
    radius = math.hypot(*initial["r_m"])
    release = (
        spec.release_sec
        if spec.release_sec is not None
        else 2 * math.pi * math.sqrt(radius**3 / MU_EARTH)
    )
    if release + 30 >= spec.horizon_sec:
        raise ValueError("choose an explicit earlier release for a short campaign")
    definition["horizon_sec"] = spec.horizon_sec
    for actor, orbit in definition["orbits"].items():
        orbit["integration"] = {
            "sample_sec": spec.sample_sec,
            "max_step_sec": spec.max_step_sec,
        }
        definition["operations"]["vehicles"][actor]["battery_wh"] = spec.battery_wh
    definition["lifecycle"]["execution_step_sec"] = spec.execution_step_sec
    definition["lifecycle"]["payloads"]["b"] = copy.deepcopy(
        definition["lifecycle"]["payloads"]["a"]
    )

    # Target construction always uses the SAME finer arc, independent of the
    # executor settings, so convergence pairs do not silently change demand.
    epoch = Epoch.utc(initial["epoch_utc"])
    context = FrameContext(
        EOPSeries(
            **{k: v for k, v in definition["eop"].items() if k != "interpolation"}
        )
    )
    arc = ReferenceEphemeris.from_result(
        propagate(
            ReferenceState(epoch, initial["r_m"], initial["v_m_s"]),
            release + 2.0,
            context,
            model=ForceModel(j2=True),
            sample_sec=1.0,
            max_step_sec=5.0,
        )
    )
    matrix = context.matrix(epoch.shifted(release + 1.0))
    direction = rotate_body(
        definition["spacecraft_bus"]["b"]["attitude"]["initial_quaternion_wxyz"],
        [0.0, 0.0, 1.0],
    )
    point = ray_ellipsoid(
        matrix @ arc.state(release + 1.0)[:3], matrix @ np.asarray(direction)
    )
    if point is None:
        raise ValueError("declared late boresight misses Earth")
    lat, lon, _ = ecef_to_geodetic(point)
    lat, lon = float(lat), float(lon)
    definition["acquisition_targets"].append(
        {"id": "late-site", "lat_deg": lat, "lon_deg": lon}
    )
    definition["operations"]["stations"].append(
        {"name": "Late-Ground", "lat_deg": lat, "lon_deg": lon, "terminals": 1}
    )
    ring = [
        [lon - 0.005, lat - 0.005],
        [lon + 0.005, lat - 0.005],
        [lon + 0.005, lat + 0.005],
        [lon - 0.005, lat + 0.005],
        [lon - 0.005, lat - 0.005],
    ]
    area = AreaOfInterest(
        id="late-area",
        anchor_lon_deg=lon,
        anchor_lat_deg=lat,
        shell=ring,
        source="Constructed fixed-boresight opportunity at declared late-wave midpoint",
    )
    definition["lifecycle"]["areas_of_interest"].append(area.model_dump(mode="json"))
    definition["lifecycle"]["cloud_masks"].append(
        {
            "id": "late-clear",
            "aoi_id": area.id,
            "aoi_sha256": digest(area.model_dump(mode="json")),
            "bounds_m": [-2000.0, -2000.0, 2000.0, 2000.0],
            "state": "clear",
            "source_status": "synthetic",
            "source": "declared constructive clear sky",
            "valid_from_sec": release,
            "valid_until_sec": release + 30.0,
        }
    )
    request = copy.deepcopy(definition["lifecycle"]["requests"][1])
    request.update(
        id="late-b",
        target_id="late-site",
        eligible_actors=["b"],
        release_sec=release,
        deadline_sec=release + 30.0,
        raw_product_id="late-raw-b",
        delivery_product_id="late-final-b",
        destination_id="Late-Ground",
        coverage={"aoi_id": area.id, "min_fraction": 0.95},
    )
    job = copy.deepcopy(definition["lifecycle"]["jobs"][0])
    job.update(id="late-job-b", inputs=["late-raw-b"])
    job["outputs"][0]["id"] = "late-final-b"
    events = [
        {
            "id": "late-arrival",
            "at_sec": release,
            "type": "request_added",
            "payload": {"request": request, "jobs": [job]},
        }
    ]
    if spec.ground_interruption:
        events.extend(
            [
                {
                    "id": "late-outage",
                    "at_sec": release + 0.5,
                    "type": "station_outage_added",
                    "payload": {
                        "station": "Late-Ground",
                        "start_sec": release + 0.5,
                        "end_sec": release + 8.0,
                    },
                },
                {
                    "id": "late-recovery",
                    "at_sec": release + 6.0,
                    "type": "station_recovered",
                    "payload": {"station": "Late-Ground"},
                },
            ]
        )
    initial_goal = task.pop("goal")
    initial_goal["deadline_sec"] = 20.0
    task.update(max_decision_sec=300.0, max_decisions=512, environment_events=events)
    for name, kind, duration, fields in [
        ("late-image", "imaging", 2.0, {"request_id": "late-b"}),
        ("late-process", "transform", 2.0, {"job_id": "late-job-b"}),
        (
            "late-deliver",
            "downlink",
            3.0,
            {"product_id": "late-final-b", "destination_id": "Late-Ground"},
        ),
    ]:
        task["actions"].append(
            {
                "id": name,
                "role": "b",
                "kind": "submit",
                "request_ref": "late-b",
                "commands": [
                    {
                        "command_id": name,
                        "actor_id": "b",
                        "kind": kind,
                        "duration_sec": duration,
                        **fields,
                    }
                ],
            }
        )
    limits = {
        actor: {
            "require_healthy": True,
            "max_energy_j": spec.horizon_sec * 16.0 + 2000.0,
        }
        for actor in scenario["actor_order"]
    }
    campaign = {
        "id": "two-wave-endurance",
        "phases": [
            {"id": "initial-delivery", "goal": initial_goal},
            {
                "id": "late-delivery",
                "goal": {
                    "conditions": [
                        {
                            "id": "late-delivery",
                            "kind": "delivery",
                            "request_ids": ["late-b"],
                        }
                    ],
                    "deadline_sec": release + 30.0,
                    "limits": limits,
                },
            },
            {
                "id": "delivery-record-endurance",
                "goal": {
                    "conditions": [
                        {
                            "id": "retained-deliveries",
                            "kind": "delivery",
                            "request_ids": [
                                *initial_goal["conditions"][0]["request_ids"],
                                "late-b",
                            ],
                            "not_before_sec": spec.horizon_sec - 0.001,
                        }
                    ],
                    "deadline_sec": spec.horizon_sec,
                    "limits": limits,
                },
            },
        ],
    }
    return {
        "configuration": {"scenario": scenario, "task": task, "campaign": campaign},
        "public_contract": {
            "minimum_completed_requests": 4,
            "completion_action_ids": [],
            "adaptive_wait": True,
            "max_wait_sec": 300.0,
        },
        "setup": {
            "horizon_sec": spec.horizon_sec,
            "release_sec": release,
            "battery_wh": spec.battery_wh,
            "scope": "synthetic constructive two-wave opportunity; two active spacecraft; battery-only RC/J2 profile; no ISL or flight calibration",
            "settings": spec.model_dump(mode="json"),
        },
    }
