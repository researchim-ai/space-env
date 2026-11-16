"""Five-spacecraft campaign with a longer alternative around a blocked ISL."""

import copy
import math

import numpy as np
from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar
from space_env.tasks.multihop_campaign import MultihopCampaignSpec, multihop_campaign


class DetourCampaignSpec(MultihopCampaignSpec):
    forward_link_available: bool = Field(default=False, strict=True)
    backup_link_available: bool = Field(default=True, strict=True)
    forward_recovery_delay_sec: FiniteScalar | None = Field(default=None, gt=0, le=30)

    @model_validator(mode="after")
    def recovery_contract(self):
        if self.forward_link_available and self.forward_recovery_delay_sec is not None:
            raise ValueError(
                "recovery delay requires initially unavailable forward link"
            )
        return self


def detour_campaign(configuration=None):
    spec = DetourCampaignSpec.model_validate(
        {} if configuration is None else configuration
    )
    case = multihop_campaign(
        {k: getattr(spec, k) for k in MultihopCampaignSpec.model_fields}
    )
    cfg = case["configuration"]
    cfg["publish_detour_graph"] = True
    definition = cfg["scenario"]["definition"]
    ops, task = definition["operations"], cfg["task"]
    for key in ("orbits", "spacecraft_bus", "acquisition_sensors"):
        definition[key]["t"] = copy.deepcopy(definition[key]["r"])
    angle = 0.003
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    for key in ("r_m", "v_m_s"):
        initial = definition["orbits"]["t"]["initial"]
        initial[key] = (rotation @ np.asarray(initial[key])).tolist()
    definition["execution"]["actors"]["t"] = copy.deepcopy(
        definition["execution"]["actors"]["r"]
    )
    definition["lifecycle"]["payloads"]["t"] = copy.deepcopy(
        definition["lifecycle"]["payloads"]["r"]
    )
    ops["vehicles"]["t"] = copy.deepcopy(ops["vehicles"]["r"])
    ops["radio_network"]["transmitters"]["t"] = copy.deepcopy(
        ops["radio_network"]["transmitters"]["r"]
    )
    cfg["scenario"]["actor_order"].append("t")
    task["roles"]["t"] = {"actors": ["t"], "action_slots": 1}
    net = ops["dtn_network"]
    if spec.forward_recovery_delay_sec is not None:
        next(e for e in net["links"] if e["source"] == "r" and e["destination"] == "s")[
            "outages"
        ] = [[0, case["setup"]["release_sec"] + spec.forward_recovery_delay_sec]]
    net["endpoints"]["t"] = copy.deepcopy(net["endpoints"]["r"])
    template = next(a for a in task["actions"] if a["id"] == "late-a-forward")
    for source, destination in [("r", "t"), ("t", "s"), ("t", "r")]:
        link = copy.deepcopy(
            next(
                e
                for e in net["links"]
                if e["source"] == "r" and e["destination"] == "s"
            )
        )
        link.update(source=source, destination=destination, outages=[])
        if not spec.backup_link_available and (source, destination) == ("t", "s"):
            link["outages"] = [[0, spec.horizon_sec]]
        net["links"].append(link)
        action = copy.deepcopy(template)
        action.update(id=f"late-a-{source}-{destination}", role=source)
        action["commands"][0].update(
            command_id=action["id"], actor_id=source, destination_id=destination
        )
        task["actions"].append(action)
    for phase in cfg["campaign"]["phases"][1:]:
        phase["goal"]["limits"]["t"] = copy.deepcopy(phase["goal"]["limits"]["r"])
    case["public_contract"]["require_route_progress"] = False
    case["public_brief"] = case["public_brief"].split(" Camera a must use")[0].replace(
        "four spacecraft", "five spacecraft"
    ) + (
        " Operate five spacecraft. Camera a can use a->r->s or a->r->t->s; camera b uses b->s. "
        "Choose the shortest currently RF-reachable continuation among admissible first hops, "
        "never revisit a previous custody holder. Current central RF graph and confirmed custody "
        "paths are public even when the separate route channel is delayed. Future link/resource "
        "availability is unknown; native admission remains authoritative."
    )
    case["setup"]["detour_settings"] = spec.model_dump(mode="json")
    return case
