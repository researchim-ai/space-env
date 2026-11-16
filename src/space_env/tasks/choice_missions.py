"""Unallocated EO/CPU/custody missions with opaque, permuted action catalogues.

Synthetic close formations. Every camera may acquire each request; two gateways
are alternatives. No precomputed assignment, route or reference schedule.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Literal

from pydantic import Field

from space_env.reference.area_coverage import AreaOfInterest
from space_env.runtime.schema import StrictModel
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.snapshot import digest
from space_env.tasks.schema import TaskSpec


class ChoiceMissionSpec(StrictModel):
    family: Literal["open", "memory", "ground_window", "link_blackout"] = "open"
    catalogue_seed: int = Field(default=0, ge=0, le=1000000, strict=True)
    cameras: int = Field(default=3, ge=2, le=4, strict=True)
    waves: int = Field(default=2, ge=2, le=3, strict=True)


def _opaque(namespace, value, salt):
    return (
        namespace
        + "-"
        + hashlib.sha256((str(salt) + "\0" + value).encode()).hexdigest()[:16]
    )


def choice_mission(value=None):
    cfg = ChoiceMissionSpec.model_validate({} if value is None else value)
    # Seed affects labels only. Physics/data remain matched across label variants.
    physical = {
        "id": "choice-" + cfg.family,
        "cameras": cfg.cameras,
        "relays": 2,
        "waves": cfg.waves,
        "wave_gap_sec": 12.0,
        "horizon_sec": 60.0,
        "seed": 0,
        "repeats": 1,
        "ack_release_actions": True,
    }
    base = campaign_workload(physical)
    scenario = copy.deepcopy(base["scenario"])
    d = scenario["definition"]
    life = d["lifecycle"]
    cameras = [a for a in scenario["actor_order"] if a.startswith("camera-")]
    gateways = [a for a in scenario["actor_order"] if a.startswith("relay-")]
    # Names never encode release-wave, owner or stage. Alias permutations are not
    # new independent geometries and do not change physical stochastic inputs.
    aliases = {}
    for row in life["requests"]:
        for key in ("id", "target_id", "raw_product_id", "delivery_product_id"):
            aliases[row[key]] = _opaque("obj", row[key], cfg.catalogue_seed)
    for row in life["jobs"]:
        aliases[row["id"]] = _opaque("obj", row["id"], cfg.catalogue_seed)
    for name in ("areas_of_interest", "cloud_masks"):
        for row in life[name]:
            aliases[row["id"]] = _opaque("obj", row["id"], cfg.catalogue_seed)

    def rename(value):
        if isinstance(value, dict):
            return {aliases.get(k, k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return aliases.get(value, value) if isinstance(value, str) else value

    scenario = rename(scenario)
    d, actions = scenario["definition"], []
    life = d["lifecycle"]
    areas = {
        a["id"]: digest(AreaOfInterest.model_validate(a).model_dump(mode="json"))
        for a in life["areas_of_interest"]
    }
    for cloud in life["cloud_masks"]:
        cloud["aoi_sha256"] = areas[cloud["aoi_id"]]
    priorities = [1, 8, 3, 13, 5, 21, 2, 17, 4, 9, 6, 11]
    for i, request in enumerate(life["requests"]):
        request["eligible_actors"] = list(cameras)
        request["priority"] = priorities[i]
        request["deadline_sec"] = min(
            60.0, request["release_sec"] + 28.0 + 8.0 * (i % 3)
        )
    if cfg.family == "memory":
        for actor in cameras:
            d["operations"]["vehicles"][actor]["memory_bits"] = 1600000.0
    if cfg.family == "ground_window":
        d["operations"]["outages"] = [
            {"station": "Ground", "start_sec": 0.0, "end_sec": 18.0},
            {"station": "Ground", "start_sec": 30.0, "end_sec": 60.0},
        ]
    if cfg.family == "link_blackout":
        for link in d["operations"]["dtn_network"]["links"]:
            if link["destination"] == gateways[0]:
                link["outages"] = [[8.0, 60.0]]

    def add(actor, kind, duration, request=None, **parameters):
        command = {
            "actor_id": actor,
            "kind": kind,
            "duration_sec": float(duration),
            **parameters,
        }
        identifier = _opaque("act", digest(command), cfg.catalogue_seed)
        command["command_id"] = identifier
        action = {
            "id": identifier,
            "role": actor,
            "kind": "submit",
            "commands": [command],
        }
        if request is not None:
            action["request_ref"] = request
        actions.append(action)

    jobs = {j["inputs"][0]: j for j in life["jobs"]}
    for request in life["requests"]:
        rid = request["id"]
        for actor in cameras:
            add(
                actor,
                "slew",
                4.0,
                rid,
                slew={"frame": "GCRS", "target_id": request["target_id"]},
            )
            add(actor, "imaging", 2.0, rid, request_id=rid)
            add(
                actor,
                "transform",
                2.0,
                rid,
                job_id=jobs[request["raw_product_id"]]["id"],
            )
            for gateway in gateways:
                add(
                    actor,
                    "relay",
                    4.0,
                    rid,
                    product_id=request["delivery_product_id"],
                    destination_id=gateway,
                    relay={"ttl_sec": 30.0},
                )
        for gateway in gateways:
            add(
                gateway,
                "downlink",
                4.0,
                rid,
                product_id=request["delivery_product_id"],
                destination_id="Ground",
            )
    for actor in scenario["actor_order"]:
        actions.append(
            {
                "id": _opaque("act", "cancel:" + actor, cfg.catalogue_seed),
                "role": actor,
                "kind": "cancel_actor",
                "actor": actor,
            }
        )
    actions.sort(key=lambda a: a["id"])
    if len({a["id"] for a in actions}) != len(actions):
        raise ValueError("opaque action collision")
    task = copy.deepcopy(base["task"])
    task["roles"].pop("ground")
    for role in task["roles"].values():
        role["action_slots"] = 1
    task.update(
        id="choice-" + cfg.family,
        actions=actions,
        max_actions=len(actions) + 1,
        max_joint_actions=len(scenario["actor_order"]),
        pointing_masks=True,
        policy_origin="external",
        max_decisions=31,
    )
    task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    scenario["id"] = "choice-" + cfg.family
    # Label variants share lineage; no train/test separation by alias seed.
    scenario["lineage"] = f"choice-formation/{cfg.cameras}/{cfg.waves}"
    scenario["metadata"] = {
        "family": "unallocated-choice/1",
        "configuration": cfg.model_dump(mode="json"),
        "scope": "public synthetic close-formation regression; scripted outages and released requests, not blind/OOD or sensor-estimated operation",
    }
    return {
        "id": cfg.family + "-labels-" + str(cfg.catalogue_seed),
        "scenario": scenario,
        "task": task,
        "configuration": cfg.model_dump(mode="json"),
        "scope": "choose among all camera owners and two single-hop gateway paths; templates remain bounded, not arbitrary flight software commands",
    }


def choice_suite():
    cases = [
        choice_mission({"family": family, "catalogue_seed": seed})
        for family in ("open", "memory", "ground_window", "link_blackout")
        for seed in (0, 19)
    ]
    return {
        "schema": "space-env.choice-suite/1",
        "cases": cases,
        "scope": "eight public regression recipes: four resource conditions, two alias permutations each; one formation geometry, not eight independent missions",
    }
