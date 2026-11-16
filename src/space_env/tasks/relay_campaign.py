"""Long synthetic camera/relay campaign on native RF and custody executors.

Late deliveries require S-band camera-to-relay custody followed by X-band ground
delivery. The two ground senders compete for one declared receiver resource pool.
"""

from __future__ import annotations

import copy
import json
import math
from importlib.resources import files

import numpy as np
from pydantic import Field

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
from space_env.runtime.schema import FiniteScalar
from space_env.simulation.snapshot import digest
from space_env.tasks.long_campaign import LongCampaignSpec, long_campaign


class RelayCampaignSpec(LongCampaignSpec):
    network_sample_sec: FiniteScalar = Field(default=5.0, ge=0.01, le=5)
    ack_loss_probability: FiniteScalar = Field(default=0.0, ge=0, le=1)
    bundle_ttl_sec: FiniteScalar = Field(default=30.0, ge=0.001, le=3600)
    ground_terminals: int = Field(default=1, ge=1, le=2, strict=True)
    ground_rate_bps: FiniteScalar = Field(default=250000.0, ge=250000, le=500000)


def relay_campaign(configuration=None):
    spec = RelayCampaignSpec.model_validate(
        {} if configuration is None else configuration
    )
    base = {name: getattr(spec, name) for name in LongCampaignSpec.model_fields}
    case = long_campaign(base)
    cfg = case["configuration"]
    scenario, task = cfg["scenario"], cfg["task"]
    definition = scenario["definition"]
    ops, life = definition["operations"], definition["lifecycle"]
    fixture = json.loads(
        files("space_env").joinpath("configs/operator_relay.json").read_text()
    )
    template = fixture["scenario"]["definition"]["operations"]
    for relay, angle in [("r", 0.002), ("s", 0.007)]:
        orbit = copy.deepcopy(definition["orbits"]["a"])
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        for key in ["r_m", "v_m_s"]:
            orbit["initial"][key] = (
                rotation @ np.asarray(orbit["initial"][key])
            ).tolist()
        definition["orbits"][relay] = orbit
        definition["spacecraft_bus"][relay] = copy.deepcopy(
            definition["spacecraft_bus"]["a"]
        )
        definition["acquisition_sensors"][relay] = copy.deepcopy(
            definition["acquisition_sensors"]["a"]
        )
        definition["execution"]["actors"][relay] = copy.deepcopy(
            definition["execution"]["actors"]["a"]
        )
        ops["vehicles"][relay] = copy.deepcopy(ops["vehicles"]["a"])
        life["payloads"][relay] = {"power_w": 10.0}
        task["roles"][relay] = {"actors": [relay], "action_slots": 1}
        scenario["actor_order"].append(relay)
    radio = copy.deepcopy(template["radio_network"])
    radio.update(sample_sec=spec.network_sample_sec, weather={}, transmitters={})
    tx = template["radio_network"]["transmitters"]["a"]
    for actor in scenario["actor_order"]:
        radio["transmitters"][actor] = copy.deepcopy(tx)
        if actor in {"r", "s"}:
            radio["transmitters"][actor]["frequency_hz"] = 8200000000.0
            radio["transmitters"][actor]["modes"] = [copy.deepcopy(tx["modes"][0])]
    ground = copy.deepcopy(radio["receivers"]["Ground"])
    ground.update(
        min_frequency_hz=8100000000.0,
        max_frequency_hz=8300000000.0,
        aggregate_rate_bps=spec.ground_rate_bps,
    )
    radio["receivers"]["Late-Ground"] = ground
    next(s for s in ops["stations"] if s["name"] == "Late-Ground")["terminals"] = (
        spec.ground_terminals
    )
    ops["radio_network"] = radio
    dtn = copy.deepcopy(template["dtn_network"])
    receiver = copy.deepcopy(dtn["links"][0]["receiver"])
    dtn.update(
        sample_sec=spec.network_sample_sec,
        max_retries=8,
        endpoints={a: {"power_w": 2.0} for a in scenario["actor_order"]},
        links=[
            {
                "source": a,
                "destination": b,
                "receiver": copy.deepcopy(receiver),
                "acquisition_sec": 0.05,
                "ack_loss_probability": spec.ack_loss_probability,
            }
            for a, b in [("a", "r"), ("b", "s")]
        ],
    )
    ops["dtn_network"] = dtn
    release = case["setup"]["release_sec"]
    late = next(e for e in task["environment_events"] if e["type"] == "request_added")
    request = copy.deepcopy(late["payload"]["request"])
    # A separate constructed opportunity for camera a preserves its original
    # footprint and pointing thresholds throughout the complete acquisition.
    initial = definition["orbits"]["a"]["initial"]
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
    bore = rotate_body(
        definition["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"],
        [0.0, 0.0, 1.0],
    )
    point = ray_ellipsoid(
        matrix @ arc.state(release + 1.0)[:3], matrix @ np.asarray(bore)
    )
    if point is None:
        raise ValueError("camera a declared boresight misses Earth")
    lat, lon, _ = ecef_to_geodetic(point)
    lat, lon = float(lat), float(lon)
    definition["acquisition_targets"].append(
        {"id": "late-site-a", "lat_deg": lat, "lon_deg": lon}
    )
    area = AreaOfInterest(
        id="late-area-a",
        anchor_lon_deg=lon,
        anchor_lat_deg=lat,
        shell=[
            [lon - 0.005, lat - 0.005],
            [lon + 0.005, lat - 0.005],
            [lon + 0.005, lat + 0.005],
            [lon - 0.005, lat + 0.005],
            [lon - 0.005, lat - 0.005],
        ],
        source="Constructed camera-a opportunity at declared midpoint",
    )
    life["areas_of_interest"].append(area.model_dump(mode="json"))
    life["cloud_masks"].append(
        {
            "id": "late-a-clear",
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
    request.update(
        id="late-a",
        eligible_actors=["a"],
        raw_product_id="late-raw-a",
        delivery_product_id="late-final-a",
        target_id="late-site-a",
        coverage={"aoi_id": area.id, "min_fraction": 0.95},
    )
    job = copy.deepcopy(late["payload"]["jobs"][0])
    job.update(id="late-job-a", inputs=["late-raw-a"])
    job["outputs"][0]["id"] = "late-final-a"
    task["environment_events"].append(
        {
            "id": "late-a-arrival",
            "at_sec": release,
            "type": "request_added",
            "payload": {"request": request, "jobs": [job]},
        }
    )
    task["environment_events"].sort(key=lambda e: e["at_sec"])
    # Hold the receiver closed until both independent camera pipelines can
    # have queued products. Reopening is an external event, hidden until now.
    if spec.ground_interruption:
        next(e for e in task["environment_events"] if e["id"] == "late-outage")[
            "payload"
        ]["end_sec"] = release + 20.0
        next(e for e in task["environment_events"] if e["id"] == "late-recovery")[
            "at_sec"
        ] = release + 12.0
        task["environment_events"].sort(key=lambda e: e["at_sec"])
    task["actions"] = [a for a in task["actions"] if a["id"] != "late-deliver"]

    def action(name, actor, kind, duration, ref, **kwargs):
        task["actions"].append(
            {
                "id": name,
                "role": actor,
                "kind": "submit",
                "request_ref": ref,
                "commands": [
                    {
                        "command_id": name,
                        "actor_id": actor,
                        "kind": kind,
                        "duration_sec": duration,
                        **kwargs,
                    }
                ],
            }
        )

    action("late-a-image", "a", "imaging", 2.0, "late-a", request_id="late-a")
    action("late-a-process", "a", "transform", 2.0, "late-a", job_id="late-job-a")
    for camera, relay in [("a", "r"), ("b", "s")]:
        ref = "late-" + camera
        pid = "late-final-" + camera
        action(
            "late-" + camera + "-point",
            camera,
            "slew",
            4.0,
            ref,
            slew={
                "frame": "GCRS",
                "target_id": "late-site-a" if camera == "a" else "late-site",
            },
        )
        action(
            "late-" + camera + "-hop",
            camera,
            "relay",
            4.0,
            ref,
            product_id=pid,
            destination_id=relay,
            relay={"ttl_sec": spec.bundle_ttl_sec},
        )
        action(
            "late-" + relay + "-ground",
            relay,
            "downlink",
            6.0,
            ref,
            product_id=pid,
            destination_id="Late-Ground",
        )
    task["runtime_audit"] = "compact"
    task["pointing_masks"] = True
    for phase in cfg["campaign"]["phases"][1:]:
        phase["goal"]["conditions"][0]["request_ids"].append("late-a")
        for relay in ["r", "s"]:
            phase["goal"]["limits"][relay] = copy.deepcopy(phase["goal"]["limits"]["a"])
    case["public_contract"]["minimum_completed_requests"] = 5
    case["public_contract"]["request_pointing"] = True
    case["public_brief"] = (
        f"Operate the four spacecraft until t={spec.horizon_sec} seconds. Deliver the three initial requests by t=20 "
        "and the two incoming requests by their announced deadlines. Use the declared camera/relay catalogue. "
        "Preserve successful delivery records to the horizon without resets. At late/endurance checks no spacecraft "
        f"fault or safe-mode history is permitted; per-actor energy cap is {spec.horizon_sec * 16 + 2000} J from mission origin. "
        f"The late station has {spec.ground_terminals} terminals and a {spec.ground_rate_bps} bit/s aggregate rate pool. "
        "Future event times are not provided."
    )
    case["setup"].update(
        scope="constructive four-spacecraft EO/CPU/ISL/RF-ground campaign; synthetic packet, RF and bus parameters; no flight calibration",
        relay_settings=spec.model_dump(mode="json"),
    )
    return case
