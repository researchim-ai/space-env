"""Synthetic fault-tolerant camera/relay campaigns with native custody."""

import copy
from typing import Literal

from pydantic import Field

from space_env.runtime.schema import FiniteScalar
from space_env.tasks.relay_campaign import RelayCampaignSpec, relay_campaign


class RoutingCampaignSpec(RelayCampaignSpec):
    route_delay_sec: FiniteScalar = Field(default=0, ge=0, le=3600)
    max_route_age_sec: FiniteScalar = Field(default=3600, ge=0, le=3600)
    fault_profile: Literal[
        "none",
        "before_transfer",
        "mid_transfer",
        "after_custody_recover",
        "after_custody_permanent",
        "all_relays_down",
    ] = "none"


def routing_campaign(configuration=None):
    spec = RoutingCampaignSpec.model_validate(configuration or {})
    case = relay_campaign({k: getattr(spec, k) for k in RelayCampaignSpec.model_fields})
    cfg = case["configuration"]
    task = cfg["task"]
    network = cfg["scenario"]["definition"]["operations"]["dtn_network"]
    original = copy.deepcopy(network["links"])
    for camera, destination in [("a", "s"), ("b", "r")]:
        edge = copy.deepcopy(next(e for e in original if e["source"] == camera))
        edge["destination"] = destination
        network["links"].append(edge)
        hop = copy.deepcopy(
            next(a for a in task["actions"] if a["id"] == f"late-{camera}-hop")
        )
        hop["id"] = f"late-{camera}-via-{destination}"
        hop["commands"][0].update(command_id=hop["id"], destination_id=destination)
        task["actions"].append(hop)
        ground = copy.deepcopy(
            next(a for a in task["actions"] if a["id"] == f"late-{destination}-ground")
        )
        ground.update(
            id=f"late-{camera}-ground-{destination}", request_ref=f"late-{camera}"
        )
        ground["commands"][0].update(
            command_id=ground["id"], product_id=f"late-final-{camera}"
        )
        task["actions"].append(ground)
    release = case["setup"]["release_sec"]
    profile = spec.fault_profile
    faults = []
    if profile != "none":
        offset = {"before_transfer": 3, "all_relays_down": 3, "mid_transfer": 4.4}.get(
            profile, 6
        )
        actors = ["r", "s"] if profile == "all_relays_down" else ["r"]
        for actor in actors:
            faults.append(
                {
                    "id": f"radio-fault-{actor}",
                    "at_sec": release + offset,
                    "type": "spacecraft_fault",
                    "payload": {"actor": actor, "subsystem": "radio"},
                }
            )
        if profile == "after_custody_recover":
            faults.append(
                {
                    "id": "radio-recovery-r",
                    "at_sec": release + 18,
                    "type": "spacecraft_fault_cleared",
                    "payload": {"actor": "r", "subsystem": "radio"},
                }
            )
    task["environment_events"].extend(faults)
    task["environment_events"].sort(key=lambda e: e["at_sec"])
    for phase in cfg["campaign"]["phases"][1:]:
        for actor in ("r", "s"):
            phase["goal"]["limits"][actor]["require_healthy"] = False
    case["public_contract"]["adaptive_routing"] = True
    case["public_contract"]["max_route_age_sec"] = spec.max_route_age_sec
    if spec.route_delay_sec > 0:
        cfg["route_telemetry"] = {"delay_sec": spec.route_delay_sec}
    case["public_brief"] = (
        f"Operate all four spacecraft until t={spec.horizon_sec} seconds. Deliver the three initial "
        "requests by t=20 and both incoming requests by their announced deadlines. "
        "Use current centralized relay telemetry to choose among declared routes. Relay faults "
        "are tolerated; cameras must remain healthy. Preserve delivery records and native custody "
        f"without resets. Each actor has an energy cap of {spec.horizon_sec * 16 + 2000} J from origin. "
        f"Late-Ground has {spec.ground_terminals} terminals and {spec.ground_rate_bps} bit/s aggregate capacity. "
        "Future fault and recovery times are not provided."
    )
    case["setup"]["routing_settings"] = spec.model_dump(mode="json")
    if spec.route_delay_sec > 0:
        case["public_brief"] += (
            f" Route telemetry is sampled at decision boundaries and transported with {spec.route_delay_sec} s delay. "
            f"Reject route samples older than {spec.max_route_age_sec} s. "
            "The native catalogue, masks, demand and operational feedback remain current."
        )
    return case
