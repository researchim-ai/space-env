"""Declared synthetic coupled formation campaigns; no flight or global-orbit claim."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.attitude import align_z
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
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class CampaignSpec(StrictModel):
    id: str = Field(default="coupled-campaign", pattern=r"^[A-Za-z0-9_-]{1,60}$")
    cameras: int = Field(default=4, ge=1, le=12, strict=True)
    relays: int = Field(default=2, ge=1, le=4, strict=True)
    waves: int = Field(default=3, ge=1, le=16, strict=True)
    wave_gap_sec: float = Field(default=12.0, ge=4, le=60, strict=True)
    horizon_sec: float = Field(default=90.0, ge=20, le=300, strict=True)
    mode: Literal["nominal", "ground_recovery", "ack_loss", "ground_outage", "ttl"] = (
        "nominal"
    )
    seed: int = Field(default=0, ge=0, lt=2**31, strict=True)
    repeats: int = Field(default=2, ge=1, le=4, strict=True)
    ground_control_sec: float = Field(default=4.0, ge=2, le=8, strict=True)
    ack_release_actions: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def envelope(self):
        if self.cameras + self.relays > 16:
            raise ValueError("coupled formation profile is bounded to 16 actors")
        if (self.waves - 1) * self.wave_gap_sec + 16 > self.horizon_sec:
            raise ValueError(
                "last wave needs at least 16 seconds of declared service horizon"
            )
        if (
            self.cameras * self.waves * 6
            + (self.relays if self.ack_release_actions else 0)
            > 1023
        ):
            raise ValueError(
                "campaign exceeds native individual-action catalogue budget"
            )
        return self


class CampaignGates(StrictModel):
    actors: int = Field(ge=2, le=16, strict=True)
    requests: int = Field(ge=1, le=200, strict=True)
    min_acquired: int = Field(ge=1, le=200, strict=True)
    min_jobs_completed: int = Field(ge=1, le=200, strict=True)
    min_custody_completed: int = Field(ge=0, le=200, strict=True)
    min_late_completed: int = Field(ge=0, le=200, strict=True)
    min_ground_senders: int = Field(ge=0, le=16, strict=True)
    require_expired_transfers: bool = False
    require_ack_retries: bool = False


def campaign_workload(value):
    cfg = CampaignSpec.model_validate(value)
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/operator_relay.json"
        ).read_text()
    )
    scenario = copy.deepcopy(fixture["scenario"])
    d = scenario["definition"]
    ops = d["operations"]
    life = d["lifecycle"]
    cameras = [f"camera-{i:02}" for i in range(cfg.cameras)]
    relays = [f"relay-{i:02}" for i in range(cfg.relays)]
    actors = cameras + relays
    template = {
        key: copy.deepcopy(d[key])
        for key in ["orbits", "spacecraft_bus", "acquisition_sensors", "execution"]
    }
    initial = template["orbits"]["a"]["initial"]
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc(initial["epoch_utc"])
    scenario.update(
        id=cfg.id,
        lineage=f"coupled-formation/{cfg.cameras}/{cfg.relays}/{cfg.waves}",
        actor_order=actors,
        seed=cfg.seed,
    )
    versioned = cfg.ground_control_sec != 4.0 or cfg.ack_release_actions
    configuration = cfg.model_dump(mode="json")
    if not versioned:
        configuration.pop("ground_control_sec")
        configuration.pop("ack_release_actions")
    scenario["metadata"] = {
        "family": "coupled-formation/2" if versioned else "coupled-formation/1",
        "parameter_status": "synthetic",
        "configuration": configuration,
        "scope": "distinct closely spaced formation arcs; fixed initial boresights and 5-degree half-angle cameras; not global multi-plane or flight qualification",
    }
    d["horizon_sec"] = cfg.horizon_sec
    d["orbits"], d["spacecraft_bus"], d["acquisition_sensors"] = {}, {}, {}
    d["execution"] = {"mode": "resource-lanes/1", "actors": {}}
    ops["vehicles"] = {}
    tx = copy.deepcopy(ops["radio_network"]["transmitters"]["a"])
    ops["radio_network"]["transmitters"] = {}
    life.update(payloads={}, requests=[], jobs=[], areas_of_interest=[], cloud_masks=[])
    d["acquisition_targets"] = []
    arcs = {}
    for i, actor in enumerate(actors):
        angle = (i - (len(actors) - 1) / 2) * 0.0007
        tilt = (i % 3 - 1) * 0.0003
        cz, sz, cy, sy = (
            math.cos(angle),
            math.sin(angle),
            math.cos(tilt),
            math.sin(tilt),
        )
        rotation = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        )
        orbit = copy.deepcopy(template["orbits"]["a"])
        r = rotation @ np.asarray(initial["r_m"])
        v = rotation @ np.asarray(initial["v_m_s"])
        orbit["initial"].update(r_m=r.tolist(), v_m_s=v.tolist())
        d["orbits"][actor] = orbit
        bus = copy.deepcopy(template["spacecraft_bus"]["a"])
        bus["attitude"]["initial_quaternion_wxyz"] = align_z(-r).tolist()
        bus["attitude"]["payload_tolerance_deg"] = 5.0
        d["spacecraft_bus"][actor] = bus
        d["acquisition_sensors"][actor] = copy.deepcopy(
            template["acquisition_sensors"]["a"]
        )
        d["execution"]["actors"][actor] = copy.deepcopy(
            template["execution"]["actors"]["a"]
        )
        ops["vehicles"][actor] = {
            **copy.deepcopy(
                fixture["scenario"]["definition"]["operations"]["vehicles"]["a"]
            ),
            "memory_bits": 8000000.0,
        }
        ops["radio_network"]["transmitters"][actor] = copy.deepcopy(tx)
        life["payloads"][actor] = {
            "power_w": 10.0,
            "footprint": {"half_angle_x_deg": 5.0, "half_angle_y_deg": 5.0},
            "imaging_model": copy.deepcopy(
                fixture["scenario"]["definition"]["lifecycle"]["payloads"]["a"][
                    "imaging_model"
                ]
            ),
        }
        arcs[actor] = ReferenceEphemeris.from_result(
            propagate(
                ReferenceState(epoch, r, v),
                cfg.horizon_sec,
                context,
                model=ForceModel(j2=True),
                sample_sec=1.0,
                max_step_sec=5.0,
            )
        )
    # One actual station/terminal and one RF aggregate pool serve every relay.
    ops["stations"][0]["terminals"] = 1
    ops["outages"] = []
    if cfg.mode == "ground_outage":
        ops["outages"] = [
            {"station": "Ground", "start_sec": 0.0, "end_sec": cfg.horizon_sec}
        ]
    elif cfg.mode == "ground_recovery":
        ops["outages"] = [{"station": "Ground", "start_sec": 4.0, "end_sec": 20.0}]
    receiver = copy.deepcopy(
        fixture["scenario"]["definition"]["operations"]["dtn_network"]["links"][0][
            "receiver"
        ]
    )
    ops["dtn_network"].update(
        seed=cfg.seed, endpoints={a: {"power_w": 2.0} for a in actors}, links=[]
    )
    for actor in cameras:
        for relay in relays:
            ops["dtn_network"]["links"].append(
                {
                    "source": actor,
                    "destination": relay,
                    "receiver": copy.deepcopy(receiver),
                    "acquisition_sec": 0.05,
                    "ack_loss_probability": 0.25 if cfg.mode == "ack_loss" else 0.0,
                }
            )
    roles = {a: {"actors": [a], "action_slots": 2} for a in actors}
    roles["ground"] = {"stations": ["Ground"]}
    actions = []

    def command(id, actor, kind, duration, **kw):
        actions.append(
            {
                "id": id,
                "role": actor,
                "kind": "submit",
                "commands": [
                    {
                        "command_id": id,
                        "actor_id": actor,
                        "kind": kind,
                        "duration_sec": float(duration),
                        **kw,
                    }
                ],
            }
        )

    for wave in range(cfg.waves):
        release = wave * cfg.wave_gap_sec
        for i, actor in enumerate(cameras):
            stem = f"w{wave:02}-{actor}"
            target = stem + "-target"
            aoi = stem + "-area"
            # Inverse footprint construction is an explicit synthetic target-placement
            # rule, computed before execution. Policies receive no propagated future.
            provider = arcs[actor]
            t = release + 1.0
            matrix = context.matrix(epoch.shifted(t))
            r = provider.state(t)[:3]
            direction = -np.asarray(d["orbits"][actor]["initial"]["r_m"])
            point = ray_ellipsoid(matrix @ r, matrix @ direction)
            if point is None:
                raise ValueError("declared fixed boresight misses Earth")
            lat, lon, _ = ecef_to_geodetic(point)
            lon, lat = float(lon), float(lat)
            d["acquisition_targets"].append(
                {
                    "id": target,
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "available_from_sec": release,
                    "available_until_sec": cfg.horizon_sec,
                }
            )
            ring = [
                [lon - 0.005, lat - 0.005],
                [lon + 0.005, lat - 0.005],
                [lon + 0.005, lat + 0.005],
                [lon - 0.005, lat + 0.005],
                [lon - 0.005, lat - 0.005],
            ]
            life["areas_of_interest"].append(
                {
                    "id": aoi,
                    "anchor_lon_deg": lon,
                    "anchor_lat_deg": lat,
                    "shell": ring,
                    "source_status": "synthetic",
                    "source": "predeclared initial-boresight intersection at nominal acquisition midpoint",
                }
            )
            # Quality is explicit: absent optical cloud truth is unknown, not clear.
            declared_area = AreaOfInterest.model_validate(life["areas_of_interest"][-1])
            life["cloud_masks"].append(
                {
                    "id": stem + "-clear",
                    "aoi_id": aoi,
                    "aoi_sha256": digest(declared_area.model_dump(mode="json")),
                    "bounds_m": [-2000.0, -2000.0, 2000.0, 2000.0],
                    "state": "clear",
                    "valid_from_sec": 0.0,
                    "valid_until_sec": cfg.horizon_sec,
                    "source_status": "synthetic",
                    "source": "predeclared synthetic clear-cell truth",
                }
            )
            raw, final, job = stem + "-raw", stem + "-final", stem + "-cpu"
            request = {
                "id": stem,
                "target_id": target,
                "eligible_actors": [actor],
                "release_sec": float(release),
                "deadline_sec": cfg.horizon_sec,
                "acquisition_sec": 2.0,
                "raw_product_id": raw,
                "raw_size_bits": 1000000,
                "delivery_product_id": final,
                "destination_id": "Ground",
                "coverage": {"aoi_id": aoi, "min_fraction": 0.95},
                "quality": {
                    "kind": "optical",
                    "max_gsd_m": 25.0,
                    "min_sun_elevation_deg": 5.0,
                },
            }
            life["requests"].append(request)
            life["jobs"].append(
                {
                    "id": job,
                    "inputs": [raw],
                    "outputs": [{"id": final, "size_bits": 500000}],
                    "power_w": 5.0,
                    "switching_sec": 0.5,
                    "transformer": "geometry-features/1",
                    "work_sec": 1.5,
                    "workspace_bits": 100000,
                }
            )
            command(stem + "-capture", actor, "imaging", 2, request_id=stem)
            command(stem + "-compute", actor, "transform", 2, job_id=job)
            choices = [relays[i % cfg.relays]]
            if cfg.relays > 1:
                choices.append(relays[(i + 1) % cfg.relays])
            for relay in choices:
                command(
                    stem + "-hop-" + relay,
                    actor,
                    "relay",
                    4,
                    product_id=final,
                    destination_id=relay,
                    relay={"ttl_sec": 0.01 if cfg.mode == "ttl" else cfg.horizon_sec},
                )
                command(
                    stem + "-ground-" + relay,
                    relay,
                    "downlink",
                    cfg.ground_control_sec,
                    product_id=final,
                    destination_id="Ground",
                )
    if cfg.ack_release_actions:
        actions.extend(
            {
                "id": "release-acked-" + relay,
                "role": relay,
                "kind": "cancel_actor",
                "actor": relay,
            }
            for relay in relays
        )
    n = cfg.cameras * cfg.waves
    task = {
        "id": "coupled-campaign-task",
        "roles": roles,
        "actions": actions,
        "decision_sec": 2.0,
        "max_decision_sec": 2.0,
        "max_decisions": math.ceil(cfg.horizon_sec / 2) + 1,
        "max_entities": 4096,
        "max_relations": 16384,
        "max_actions": len(actions) + 1,
        "max_joint_actions": 2 * len(actors) + 1,
        "runtime_audit": "compact",
        "policy_origin": "scripted",
    }
    negative = cfg.mode in {"ground_outage", "ttl"}
    gates = CampaignGates(
        actors=len(actors),
        requests=n,
        min_acquired=1,
        min_jobs_completed=1,
        min_custody_completed=0 if cfg.mode == "ttl" else 1,
        min_late_completed=0 if negative or cfg.waves == 1 else 1,
        min_ground_senders=0 if negative else min(cfg.relays, cfg.cameras),
        require_expired_transfers=cfg.mode == "ttl",
        require_ack_retries=cfg.mode == "ack_loss",
    )
    result = {
        "id": cfg.id,
        "family": "coupled-formation",
        "source": "synthetic declared EO/CPU/DTN/RF/thermal formation campaign; not calibrated flight data",
        "scenario": scenario,
        "task": task,
        "policy": {
            "name": "route_aware_rolling",
            "max_candidates": 64,
            "max_nodes": 1024,
            **({"reclaim_acked_downlinks": True} if cfg.ack_release_actions else {}),
        },
        "repeats": cfg.repeats,
        "timeout_sec": 1800,
        "min_completion_fraction": 0.0 if negative else 1.0,
        "max_completion_fraction": 0.0 if negative else None,
        "campaign_gates": gates.model_dump(mode="json"),
    }
    from space_env.scaling.benchmark import Workload

    return Workload.model_validate(result).model_dump(mode="json")


def campaign_evidence(scenario, report, gates):
    """Evaluator-only witnesses from native final reports; never policy input."""
    gates = CampaignGates.model_validate(gates)
    life = report.get("lifecycle", {})
    dtn = report.get("dtn", {})
    bus = report.get("spacecraft_bus", {})
    artifacts = life.get("artifacts", {})
    requests = life.get("requests", {})
    jobs = life.get("jobs", {})
    transitions = dtn.get("transfers", {})
    acquired = sum(p.get("source") == "acquired" for p in artifacts.values())
    completed_jobs = sum(j.get("status") == "completed" for j in jobs.values())
    custody = sum(t.get("status") == "completed" for t in transitions.values())
    late = sum(
        r.get("release_sec", 0) > 0 and r.get("status") == "completed"
        for r in requests.values()
    )
    senders = {r["actor"] for r in life.get("acknowledgements", [])}
    losses = sum(row.get("kind") == "ack_lost" for row in dtn.get("history", []))
    attempts = {}
    for row in dtn.get("history", []):
        if row.get("kind") == "serialized":
            key = (row["transfer_id"], row["fragment"])
            attempts[key] = attempts.get(key, 0) + 1
    retries = sum(max(0, count - 1) for count in attempts.values())
    initial = [v["initial"] for v in scenario["definition"]["orbits"].values()]
    distinct = len({digest(x) for x in initial})
    checks = {
        "actor_count": len(scenario["actor_order"]) == gates.actors,
        "distinct_initial_states": distinct == gates.actors,
        "request_count": len(requests) == gates.requests,
        "no_preloaded_products": not scenario["definition"]["operations"]["products"],
        "coupled_profiles": bool(bus.get("states"))
        and bool(report.get("radio_network"))
        and bool(dtn),
        "acquisition": acquired >= gates.min_acquired,
        "cpu_jobs": completed_jobs >= gates.min_jobs_completed,
        "custody": custody >= gates.min_custody_completed,
        "later_arrivals_completed": late >= gates.min_late_completed,
        "shared_ground_senders": len(senders) >= gates.min_ground_senders,
        "expiry_witness": not gates.require_expired_transfers
        or any(t.get("status") == "expired" for t in transitions.values()),
        "ack_retry_witness": not gates.require_ack_retries
        or (losses > 0 and retries > 0),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "acquired": acquired,
        "jobs_completed": completed_jobs,
        "custody_completed": custody,
        "later_arrivals_completed": late,
        "ground_senders": sorted(senders),
        "distinct_initial_states": distinct,
        "ack_loss_events": losses,
        "retransmitted_fragments": retries,
        "scope": "native evaluator witnesses; geometry is deliberately synthetic and formation-local",
    }
